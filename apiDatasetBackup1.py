"""
API Unificada — NAS100 + US30 + GER40 + BTCUSD + AUDUSD
Inicio: uvicorn apiDataset:app --host 172.16.0.4 --port 80 --reload
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import requests as http_requests

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "DataFiles")

# ── Configuración de modelos por símbolo ──────────────────────
# features_type:
#   "standard" → NAS100/US30/GER40/BTCUSD: 5m OHLCV + EMA/MACD/ADX
#   "audusd"   → AUDUSD: 5m+15m OHLCV + RSI + Stochastic (sin EMA/MACD/ADX)
MODELOS_CONFIG = {
    "NAS100": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_NAS100_v4.pth",
        "scaler_file":   "scaler_NAS100_v4.pkl",
        "cols_file":     "input_columns_NAS100_v4.pkl",
        "data_file":     "Data_Entrenamiento_NAS100.xlsx",
        "min_threshold": 0.92,
        "features_type": "standard",
    },
    "US30": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_US30_v4.pth",
        "scaler_file":   "scaler_US30_v4.pkl",
        "cols_file":     "input_columns_US30_v4.pkl",
        "data_file":     "Data_Entrenamiento_US30.xlsx",
        "min_threshold": 0.92,
        "features_type": "standard",
    },
    "GER40": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GER40_v4.pth",
        "scaler_file":   "scaler_GER40_v4.pkl",
        "cols_file":     "input_columns_GER40_v4.pkl",
        "data_file":     "Data_Entrenamiento_GER40.xlsx",
        "min_threshold": 0.92,
        "features_type": "standard",
    },
    "BTCUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_BTCUSD_v4.pth",
        "scaler_file":   "scaler_BTCUSD_v4.pkl",
        "cols_file":     "input_columns_BTCUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_BTCUSD.xlsx",
        "min_threshold": 0.92,
        "features_type": "standard",
    },
    "AUDUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_AUDUSD_v4.pth",
        "scaler_file":   "scaler_AUDUSD_v4.pkl",
        "cols_file":     "input_columns_AUDUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_AUDUSD.xlsx",
        "min_threshold": 0.92,
        "features_type": "audusd",  # 19 features — URL fija del EA
    },
    "GBPAUD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GBPAUD_v4.pth",
        "scaler_file":   "scaler_GBPAUD_v4.pkl",
        "cols_file":     "input_columns_GBPAUD_v4.pkl",
        "data_file":     "Data_Entrenamiento_GBPAUD.xlsx",
        "min_threshold": 0.98,
        "features_type": "audusd",  # misma URL que AUDUSD — 19 features
    },
    "EURUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_EURUSD_v4.pth",
        "scaler_file":   "scaler_EURUSD_v4.pkl",
        "cols_file":     "input_columns_EURUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_EURUSD.xlsx",
        "min_threshold": 0.92,
        "features_type": "audusd",  # misma URL que AUDUSD/GBPAUD — 19 features
    },
}

# ── Cache anti-duplicados por vela ────────────────────────────
ULTIMA_VELA = {}  # { "NAS100": "202602250100", ... }

# ── Arquitectura del modelo ───────────────────────────────────

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()

class TradingModelV4(nn.Module):
    def __init__(self, input_size, dropout=0.4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
        )
        self.regressor  = nn.Linear(64, 1)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        h = self.shared(x)
        return self.regressor(h), self.classifier(h)

    def predict_with_uncertainty(self, x, n_samples=30):
        self.eval(); enable_dropout(self)
        pp, tp = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                p, t = self.forward(x)
                pp.append(p.unsqueeze(0))
                tp.append(torch.softmax(t, dim=1).unsqueeze(0))
        ps = torch.cat(pp, 0); ts = torch.cat(tp, 0)
        pm = ps.mean(0); tm = ts.mean(0)
        conf = tm.max(1).values - ts.std(0).max(1).values
        return pm, tm, conf

# ── Normalización ─────────────────────────────────────────────

def norm_scalar(value, p1, p99):
    if p99 - p1 == 0: return 0.0
    return float(np.clip(2 * (value - p1) / (p99 - p1) - 1, -1.5, 1.5))

def norm_series(col, p1, p99):
    if p99 - p1 == 0: return col * 0
    return (2 * (col - p1) / (p99 - p1) - 1).clip(-1.5, 1.5)

def denorm_scalar(value, min_val, max_val):
    return float((value + 1) * (max_val - min_val) / 2 + min_val)

def denorm_array(arr, min_val, max_val):
    return (arr + 1) * (max_val - min_val) / 2 + min_val

# ── Dataset ───────────────────────────────────────────────────

class DS(Dataset):
    def __init__(self, X, yp, yt):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.yp = torch.tensor(yp, dtype=torch.float32)
        self.yt = torch.tensor(yt, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.yp[i], self.yt[i]

# ── Normalización de dataset según features_type ─────────────

def normalizar_dataset(df, s, features_type):
    pips = s.get("pips_factor", 1)
    df['profit_original'] = df['profit'].fillna(0) * pips
    df['profit_norm'] = norm_series(df['profit_original'], s['p1_profit'], s['p99_profit'])
    for c in ['precioopen5','precioclose5','preciohigh5','preciolow5']:
        df[c] = norm_series(df[c], s['p1_precio5'], s['p99_precio5'])
    df['volume5'] = norm_series(df['volume5'], s['p1_vol5'], s['p99_vol5'])
    for c in ['rsi5','iStochaMain5','iStochaSign5']:
        df[c] = df[c] / 100.0

    if features_type == "audusd":
        for c in ['precioopen15','precioclose15','preciohigh15','preciolow15']:
            df[c] = norm_series(df[c], s['p1_precio15'], s['p99_precio15'])
        df['volume15'] = norm_series(df['volume15'], s['p1_vol15'], s['p99_vol15'])
        for c in ['rsi15','iStochaMain15','iStochaSign15']:
            df[c] = df[c] / 100.0
    else:
        for c in ['ema550','ema5200','ema50_prev','ema5200_prev',
                   'macdLine5','signalLine5','macdLine_prev5','signalLine_prev5',
                   'adx5','diPlus5','diMinus5']:
            df[c] = norm_series(df[c], s[f'p1_{c}'], s[f'p99_{c}'])

# ── Cálculo de umbral dinámico ────────────────────────────────

def calcular_umbral_optimo(model, SC, COLS, data_file, features_type="standard"):
    data = pd.read_excel(os.path.join(DATA_DIR, data_file))
    data.columns = data.columns.str.strip()
    data["profit"] = pd.to_numeric(data["profit"], errors="coerce")
    data = data[data['tipo'].isin(['BUY','SELL']) & data["profit"].notna()].copy()
    data['fecha'] = pd.to_datetime(
        data['fecha'].astype(str).str.replace(',', '-', regex=False),
        format="%Y-%m-%d %H:%M", errors='coerce')
    data = data[data['fecha'].notna()].sort_values('fecha').reset_index(drop=True)

    test_data = data.iloc[int(len(data) * 0.85):].copy()
    test_data['dia_semana'] = test_data['fecha'].dt.weekday / 6.0
    test_data['hora']       = test_data['fecha'].dt.hour    / 23.0
    test_data['minuto']     = test_data['fecha'].dt.minute  / 55.0

    normalizar_dataset(test_data, SC, features_type)
    test_data['tipo_encoded'] = test_data['tipo'].apply(lambda x: 1 if x == 'BUY' else 0)
    test_data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    test_data.fillna(0, inplace=True)

    X  = test_data[COLS].values
    yp = test_data['profit_norm'].values.reshape(-1, 1)
    yt = test_data['tipo_encoded'].values
    loader = DataLoader(DS(X, yp, yt), batch_size=32, shuffle=False)

    all_conf, all_real_p, all_pred_p = [], [], []
    for Xb, ypb, ytb in loader:
        pm, tm, conf = model.predict_with_uncertainty(Xb)
        all_conf.extend(conf.cpu().numpy())
        all_real_p.extend(ypb.cpu().numpy().flatten())
        all_pred_p.extend(pm.cpu().numpy().flatten())

    conf   = np.array(all_conf)
    real_p = denorm_array(np.array(all_real_p), SC['min_profit'], SC['max_profit'])
    pred_p = denorm_array(np.array(all_pred_p), SC['min_profit'], SC['max_profit'])
    total  = len(conf)

    best_thr, best_pf, best_n = 0.35, 0.0, total
    for thr in np.arange(0.35, 0.92, 0.01):
        mask  = conf > thr
        n_ops = mask.sum()
        if n_ops < total * 0.10: break
        fr = real_p[mask]
        winners, losers = fr[fr > 0], fr[fr < 0]
        if len(losers) == 0 or len(winners) == 0: continue
        pf = abs(winners.sum() / losers.sum())
        if pf > best_pf:
            best_pf, best_thr, best_n = pf, round(float(thr), 2), int(n_ops)

    p_inf = round(float(np.percentile(pred_p, 10)), 6)
    p_sup = round(float(np.percentile(pred_p, 90)), 6)
    print(f"    Umbral: {best_thr:.2f}  →  {best_n} ops ({best_n/total*100:.1f}%)  |  PF: {best_pf:.2f}  |  P10: {p_inf:.2f}  P90: {p_sup:.2f}")
    return best_thr, best_pf, best_n, total, p_inf, p_sup

# ── Cargar todos los modelos al arrancar ──────────────────────

MODELOS = {}

for symbol, cfg in MODELOS_CONFIG.items():
    print(f"Cargando modelo {symbol}...")
    try:
        scaler_path = os.path.join(cfg["model_dir"], cfg["scaler_file"])
        cols_path   = os.path.join(cfg["model_dir"], cfg["cols_file"])
        model_path  = os.path.join(cfg["model_dir"], cfg["model_file"])

        with open(scaler_path, "rb") as f: SC   = pickle.load(f)
        with open(cols_path,   "rb") as f: COLS = pickle.load(f)

        model = TradingModelV4(input_size=len(COLS), dropout=0.4)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        ft = cfg.get("features_type", "standard")
        print(f"  Calculando umbral {symbol}...")
        thr, pf, n, total, p_inf, p_sup = calcular_umbral_optimo(model, SC, COLS, cfg["data_file"], ft)

        min_thr = cfg.get("min_threshold", 0.35)
        thr = max(thr, min_thr)
        if thr == min_thr:
            print(f"  Umbral ajustado al mínimo configurado: {thr:.2f}")

        MODELOS[symbol] = {
            "model":         model,
            "SC":            SC,
            "COLS":          COLS,
            "threshold":     thr,
            "pf":            pf,
            "n_ops":         n,
            "total":         total,
            "p_inf":         p_inf,
            "p_sup":         p_sup,
            "features_type": ft,
        }
        print(f"  {symbol} listo.\n")
    except Exception as e:
        print(f"  ERROR cargando {symbol}: {e}\n")

print(f"API lista — modelos cargados: {list(MODELOS.keys())}\n")

# ── App ───────────────────────────────────────────────────────

app = FastAPI(title="Trading API Unificada")

@app.get("/predict")
def predict(
    symbol: str   = Query(...),
    fecha:  str   = Query(...),
    # 5m — todos los modelos
    o5:  float = Query(...), c5:  float = Query(...),
    h5:  float = Query(...), l5:  float = Query(...), v5: float = Query(...),
    r5:  float = Query(...),
    m5:  float = Query(...), s5:  float = Query(...),
    # 15m — AUDUSD los usa, los demás los ignoran
    o15: float = Query(None), c15: float = Query(None),
    h15: float = Query(None), l15: float = Query(None), v15: float = Query(None),
    r15: float = Query(None), m15: float = Query(None), s15: float = Query(None),
    # EMA/MACD/ADX 5m — NAS100/US30/GER40/BTCUSD
    ema550:           float = Query(None), ema5200:          float = Query(None),
    ema50_prev:       float = Query(None), ema5200_prev:     float = Query(None),
    macdLine5:        float = Query(None), signalLine5:      float = Query(None),
    macdLine_prev5:   float = Query(None), signalLine_prev5: float = Query(None),
    adx5: float = Query(None), diPlus5: float = Query(None), diMinus5: float = Query(None),
    # EMA/MACD/ADX 15m — aceptados pero ignorados
    ema5015: float = Query(None), ema20015: float = Query(None),
    ema50_prev15: float = Query(None), ema200_prev15: float = Query(None),
    macdLine15: float = Query(None), signalLine15: float = Query(None),
    macdLine_prev15: float = Query(None), signalLine_prev15: float = Query(None),
    adx15: float = Query(None), diPlus15: float = Query(None), diMinus15: float = Query(None),
    # Param extra AUDUSD (c5d duplicado de c5, se ignora)
    c5d: float = Query(None),
):
    sym = symbol.upper()
    if sym not in MODELOS:
        return JSONResponse({
            "valor_profit":  0.0,
            "RESULTADO":     "IGNORE",
            "percentil_inf": 0.0,
            "percentil_sup": 0.0,
        })

    m         = MODELOS[sym]
    s         = m["SC"]
    MODEL     = m["model"]
    COLS      = m["COLS"]
    THRESHOLD = m["threshold"]
    FT        = m["features_type"]

    try:
        dt = datetime.fromisoformat(fecha)
    except ValueError:
        raise HTTPException(400, f"Fecha invalida: '{fecha}'. Formato: 2026-01-19T01:15:48")

    # ── Filtro anti-duplicados por vela de 5m ─────────────────
    minuto_vela = dt.replace(second=0, microsecond=0)
    minuto_vela = minuto_vela.replace(minute=(dt.minute // 5) * 5)
    clave_vela  = f"{sym}_{minuto_vela.strftime('%Y%m%d%H%M')}"

    if ULTIMA_VELA.get(sym) == clave_vela:
        return JSONResponse({
            "valor_profit":  0.0,
            "RESULTADO":     "IGNORE",
            "percentil_inf": m["p_inf"],
            "percentil_sup": m["p_sup"],
        })
    ULTIMA_VELA[sym] = clave_vela

    # ── Construir features según tipo ─────────────────────────
    if FT == "audusd":
        pips = s.get("pips_factor", 10000)
        use_rel = s.get("use_relative_prices", False)
        if use_rel:
            # Precios relativos al open en pips — igual que en entrenamiento
            c5_r  = (c5  - o5)  * pips;  h5_r  = (h5  - o5)  * pips;  l5_r  = (l5  - o5)  * pips
            o15_v = o15 or 0.0; c15_v = c15 or 0.0; h15_v = h15 or 0.0; l15_v = l15 or 0.0
            c15_r = (c15_v - o15_v) * pips; h15_r = (h15_v - o15_v) * pips; l15_r = (l15_v - o15_v) * pips
        features = {
            "dia_semana":     dt.weekday() / 6.0,
            "hora":           dt.hour      / 23.0,
            "minuto":         dt.minute    / 55.0,
            "precioopen5":    0.0,
            "precioclose5":   norm_scalar(c5_r,  s["p1_precio5"],  s["p99_precio5"]) if use_rel else norm_scalar(c5,  s["p1_precio5"],  s["p99_precio5"]),
            "preciohigh5":    norm_scalar(h5_r,  s["p1_precio5"],  s["p99_precio5"]) if use_rel else norm_scalar(h5,  s["p1_precio5"],  s["p99_precio5"]),
            "preciolow5":     norm_scalar(l5_r,  s["p1_precio5"],  s["p99_precio5"]) if use_rel else norm_scalar(l5,  s["p1_precio5"],  s["p99_precio5"]),
            "volume5":        norm_scalar(v5,  s["p1_vol5"],  s["p99_vol5"]),
            "precioopen15":   0.0,
            "precioclose15":  norm_scalar(c15_r, s["p1_precio15"], s["p99_precio15"]) if use_rel else norm_scalar(c15_v, s["p1_precio15"], s["p99_precio15"]),
            "preciohigh15":   norm_scalar(h15_r, s["p1_precio15"], s["p99_precio15"]) if use_rel else norm_scalar(h15_v, s["p1_precio15"], s["p99_precio15"]),
            "preciolow15":    norm_scalar(l15_r, s["p1_precio15"], s["p99_precio15"]) if use_rel else norm_scalar(l15_v, s["p1_precio15"], s["p99_precio15"]),
            "volume15":       norm_scalar(v15 or 0, s["p1_vol15"], s["p99_vol15"]),
            "rsi5":           r5  / 100.0,
            "iStochaMain5":   m5  / 100.0,
            "iStochaSign5":   s5  / 100.0,
            "rsi15":          (r15 or 0) / 100.0,
            "iStochaMain15":  (m15 or 0) / 100.0,
            "iStochaSign15":  (s15 or 0) / 100.0,
        }
    else:
        features = {
            "dia_semana":    dt.weekday() / 6.0,
            "hora":          dt.hour      / 23.0,
            "minuto":        dt.minute    / 55.0,
            "precioopen5":   norm_scalar(o5, s["p1_precio5"], s["p99_precio5"]),
            "precioclose5":  norm_scalar(c5, s["p1_precio5"], s["p99_precio5"]),
            "preciohigh5":   norm_scalar(h5, s["p1_precio5"], s["p99_precio5"]),
            "preciolow5":    norm_scalar(l5, s["p1_precio5"], s["p99_precio5"]),
            "volume5":       norm_scalar(v5, s["p1_vol5"],    s["p99_vol5"]),
            "rsi5":          r5 / 100.0,
            "iStochaMain5":  m5 / 100.0,
            "iStochaSign5":  s5 / 100.0,
            "ema550":           norm_scalar(ema550 or 0,         s["p1_ema550"],          s["p99_ema550"]),
            "ema5200":          norm_scalar(ema5200 or 0,        s["p1_ema5200"],         s["p99_ema5200"]),
            "ema50_prev":       norm_scalar(ema50_prev or 0,     s["p1_ema50_prev"],      s["p99_ema50_prev"]),
            "ema5200_prev":     norm_scalar(ema5200_prev or 0,   s["p1_ema5200_prev"],    s["p99_ema5200_prev"]),
            "macdLine5":        norm_scalar(macdLine5 or 0,      s["p1_macdLine5"],       s["p99_macdLine5"]),
            "signalLine5":      norm_scalar(signalLine5 or 0,    s["p1_signalLine5"],     s["p99_signalLine5"]),
            "macdLine_prev5":   norm_scalar(macdLine_prev5 or 0, s["p1_macdLine_prev5"],  s["p99_macdLine_prev5"]),
            "signalLine_prev5": norm_scalar(signalLine_prev5 or 0,s["p1_signalLine_prev5"],s["p99_signalLine_prev5"]),
            "adx5":    norm_scalar(adx5 or 0,    s["p1_adx5"],    s["p99_adx5"]),
            "diPlus5": norm_scalar(diPlus5 or 0, s["p1_diPlus5"], s["p99_diPlus5"]),
            "diMinus5":norm_scalar(diMinus5 or 0,s["p1_diMinus5"],s["p99_diMinus5"]),
        }

    vector   = np.array([features[col] for col in COLS], dtype=np.float32)
    x_tensor = torch.tensor(vector).unsqueeze(0)

    pred_profit, tipo_probs, confidence = MODEL.predict_with_uncertainty(x_tensor, n_samples=30)

    confidence = float(confidence.item())
    class_idx  = int(tipo_probs.argmax(dim=1).item())
    signal     = "BUY" if class_idx == 1 else "SELL"
    profit_est = denorm_scalar(pred_profit.item(), s["min_profit"], s["max_profit"])
    valid      = confidence >= THRESHOLD

    return JSONResponse({
        "valor_profit":  round(profit_est, 6),
        "RESULTADO":     signal if valid else "IGNORE",
        "percentil_inf": m["p_inf"],
        "percentil_sup": m["p_sup"],
    })

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "modelos": {
            sym: {
                "threshold":  m["threshold"],
                "pf":         round(m["pf"], 2),
                "ops":        f"{m['n_ops']}/{m['total']}",
                "p_inf":      m["p_inf"],
                "p_sup":      m["p_sup"],
            }
            for sym, m in MODELOS.items()
        }
    }

# ── Conexión BD ───────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host="severtraderdb.postgres.database.azure.com",
        database="postgres",
        user="Neotradingai",
        password="TraderResponsable2022@",
        cursor_factory=RealDictCursor,
        sslmode="require",
        connect_timeout=5
    )

# ── /moneda ───────────────────────────────────────────────────

@app.get("/moneda")
def insert_moneda(par: int, date: str, time: str,
                  open: float, high: float, low: float, close: float, volume: float):
    try:
        conn = get_connection(); cur = conn.cursor()
        cur.execute("""INSERT INTO "DataTrader1m"
            ("Date","Time","Open","High","Low","Close","Volume","par_id")
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (date, time, open, high, low, close, volume, par))
        conn.commit(); cur.close(); conn.close()
        return {"message": "Success!"}
    except Exception as e:
        return {"error": str(e)}

# ── /operaciones ──────────────────────────────────────────────

@app.get("/operaciones")
def insert_or_update_operacion(
    Date: str, Ticket: int, Symbol: str, Lotes: float, Type: str,
    DateOpen: str, DateClose: str, Price: float, ClosePrice: float,
    Magic: int, SL: float, TP: float, Profit: float, account_id: int
):
    try:
        conn = get_connection(); cur = conn.cursor()
        cur.execute('SELECT "id" FROM "Account" WHERE "id" = %s', (account_id,))
        if not cur.fetchone(): return {"error": "Account does not exist"}
        cur.execute('SELECT "Ticket" FROM "Operation" WHERE "Ticket" = %s', (Ticket,))
        if cur.fetchone():
            cur.execute("""UPDATE "Operation" SET
                "Lotes"=%s,"DateOpen"=%s,"DateClose"=%s,"Price"=%s,
                "ClosePrice"=%s,"Magic"=%s,"SL"=%s,"TP"=%s,"Profit"=%s
                WHERE "Ticket"=%s""",
                (Lotes, DateOpen, DateClose, Price, ClosePrice, Magic, SL, TP, Profit, Ticket))
        else:
            cur.execute("""INSERT INTO "Operation"
                ("Date","Ticket","account_id","Symbol","Lotes","Type",
                 "DateOpen","DateClose","Price","ClosePrice","Magic","SL","TP","Profit")
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (Date, Ticket, account_id, Symbol, Lotes, Type,
                 DateOpen, DateClose, Price, ClosePrice, Magic, SL, TP, Profit))
        conn.commit(); cur.close(); conn.close()
        return {"message": "Successful!"}
    except Exception as e:
        return {"error": str(e)}

# ── /detallebalance ───────────────────────────────────────────

@app.get("/detallebalance")
def insert_detailbalance(
    Date: str, Time: str, Balance: float, Equity: float,
    FreeMargin: float, FreeMarginMode: float, Flotante: float,
    Operations: int, FracFlotante: float, account_id: int
):
    try:
        conn = get_connection(); cur = conn.cursor()
        cur.execute("""INSERT INTO "DetailBalance"
            ("Date","Time","Balance","Equity","FreeMargin","FreeMarginMode",
             "Flotante","Operations","FracFlotante","account_id")
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (Date, Time, Balance, Equity, FreeMargin, FreeMarginMode,
             Flotante, Operations, FracFlotante, account_id))
        conn.commit(); cur.close(); conn.close()
        return {"message": "Successful!"}
    except Exception as e:
        return {"error": str(e)}

# ── /vix ──────────────────────────────────────────────────────

@app.get("/vix")
def get_vix():
    try:
        r = http_requests.get(
            "https://query2.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1m",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        precio = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return {"VIX": precio}
    except Exception as e:
        return {"error": f"No se pudo obtener el VIX: {str(e)}"}