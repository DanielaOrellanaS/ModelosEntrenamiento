"""
API Unificada — NAS100 + US30 + GER40 + BTCUSD + AUDUSD + GBPAUD + EURUSD + GBPUSD
Inicio: uvicorn apiDataset:app --host 192.168.100.73 --port 80 --reload

CAMBIOS v6:
  - min_threshold diferenciado: índices=0.92, forex=0.50
    (AUDUSD/GBPAUD/EURUSD/GBPUSD nunca alcanzan 0.92 por naturaleza del activo)
  - Cache de velas corregido: ahora guarda por clave completa sym+vela
    en lugar de solo por símbolo, evitando que peticiones históricas
    devuelvan el resultado de la última vela procesada
  - valor_profit en pips para forex (÷ pips_factor antes de responder)
    para que el EA reciba el mismo orden de magnitud que los índices
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import RealDictCursor
import requests as http_requests
from collections import OrderedDict

API_VERSION   = "v6"
STARTUP_TIME  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "DataFiles")
CACHE_DIR = os.path.join(BASE_DIR, "Trading_Modelv4")

# ── Configuración de modelos ──────────────────────────────────
# CLAVE: min_threshold separado para índices (0.92) y forex (0.50)
# Los modelos forex generan confianzas más bajas por la naturaleza del
# precio (movimientos pequeños, alta incertidumbre relativa).
# 0.50 deja que el umbral dinámico calculado en training sea el que mande.

MODELOS_CONFIG = {
    "NAS100": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_NAS100_v4.pth",
        "scaler_file":   "scaler_NAS100_v4.pkl",
        "cols_file":     "input_columns_NAS100_v4.pkl",
        "data_file":     "Data_Entrenamiento_NAS100.xlsx",
        "min_threshold": 0.92,   # índice — alta confianza requerida
    },
    "US30": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_US30_v4.pth",
        "scaler_file":   "scaler_US30_v4.pkl",
        "cols_file":     "input_columns_US30_v4.pkl",
        "data_file":     "Data_Entrenamiento_US30.xlsx",
        "min_threshold": 0.92,
    },
    "GER40": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GER40_v4.pth",
        "scaler_file":   "scaler_GER40_v4.pkl",
        "cols_file":     "input_columns_GER40_v4.pkl",
        "data_file":     "Data_Entrenamiento_GER40.xlsx",
        "min_threshold": 0.92,
    },
    "BTCUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_BTCUSD_v4.pth",
        "scaler_file":   "scaler_BTCUSD_v4.pkl",
        "cols_file":     "input_columns_BTCUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_BTCUSD.xlsx",
        "min_threshold": 0.92,
    },
    # ── FOREX: min_threshold = 0.50 ──────────────────────────────
    # Deja que el umbral óptimo calculado en entrenamiento sea el que
    # decida. Si el mejor umbral fue 0.62, la API usará 0.62.
    # Con 0.92 TODOS los forex salían IGNORE siempre.
    "AUDUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_AUDUSD_v4.pth",
        "scaler_file":   "scaler_AUDUSD_v4.pkl",
        "cols_file":     "input_columns_AUDUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_AUDUSD.xlsx",
        "min_threshold": 0.50,
    },
    "GBPAUD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GBPAUD_v4.pth",
        "scaler_file":   "scaler_GBPAUD_v4.pkl",
        "cols_file":     "input_columns_GBPAUD_v4.pkl",
        "data_file":     "Data_Entrenamiento_GBPAUD.xlsx",
        "min_threshold": 0.50,
    },
    "EURUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_EURUSD_v4.pth",
        "scaler_file":   "scaler_EURUSD_v4.pkl",
        "cols_file":     "input_columns_EURUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_EURUSD.xlsx",
        "min_threshold": 0.50,
    },
    "GBPUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GBPUSD_v4.pth",
        "scaler_file":   "scaler_GBPUSD_v4.pkl",
        "cols_file":     "input_columns_GBPUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_GBPUSD.xlsx",
        "min_threshold": 0.50,
    },
}

# ── Cache de resultados por vela ──────────────────────────────
# Clave: "SYM_YYYYMMDDHHSS" — única por símbolo+vela
# Capacidad máxima: 200 entradas (LRU)
# Antes se usaba solo sym como clave → peticiones de velas distintas
# devolvían el resultado de la última vela procesada.
CACHE_VELAS: OrderedDict = OrderedDict()
CACHE_MAX = 200

def cache_get(clave):
    if clave in CACHE_VELAS:
        CACHE_VELAS.move_to_end(clave)
        return CACHE_VELAS[clave]
    return None

def cache_set(clave, valor):
    CACHE_VELAS[clave] = valor
    CACHE_VELAS.move_to_end(clave)
    if len(CACHE_VELAS) > CACHE_MAX:
        CACHE_VELAS.popitem(last=False)

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

    def predict_with_uncertainty(self, x, n_samples=30, seed=42):
        self.eval(); enable_dropout(self)
        pp, tp = [], []
        with torch.no_grad():
            for i in range(n_samples):
                torch.manual_seed(seed + i)
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

def detectar_tipo_modelo(SC):
    return "standard"

# ── Dataset ───────────────────────────────────────────────────

class DS(Dataset):
    def __init__(self, X, yp, yt):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.yp = torch.tensor(yp, dtype=torch.float32)
        self.yt = torch.tensor(yt, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.yp[i], self.yt[i]

# ── Normalización de dataset para calcular umbral ────────────

def normalizar_dataset(df, s, model_type):
    pips = s.get("pips_factor", 1)
    df['profit_original'] = df['profit'].fillna(0) * pips
    df['profit_norm'] = norm_series(df['profit_original'], s['p1_profit'], s['p99_profit'])

    df['volume5'] = norm_series(df['volume5'], s['p1_vol5'], s['p99_vol5'])
    for c in ['rsi5', 'iStochaMain5', 'iStochaSign5']:
        df[c] = df[c] / 100.0

    for c in ['precioopen5','precioclose5','preciohigh5','preciolow5']:
        df[c] = norm_series(df[c], s['p1_precio5'], s['p99_precio5'])
    for c in ['ema550','ema5200','ema50_prev','ema5200_prev',
               'macdLine5','signalLine5','macdLine_prev5','signalLine_prev5',
               'adx5','diPlus5','diMinus5']:
        df[c] = norm_series(df[c], s[f'p1_{c}'], s[f'p99_{c}'])

# ── Cache de umbrales en disco ────────────────────────────────

def _cache_path(symbol):
    return os.path.join(CACHE_DIR, f"threshold_cache_{symbol}.json")

def _load_threshold_cache(symbol, model_path):
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        current_mtime = os.path.getmtime(model_path)
        if abs(data.get("model_mtime", 0) - current_mtime) < 1.0:
            print(f"  [{symbol}] Umbral cargado desde cache: {data['threshold']:.2f}  PF: {data['pf']:.2f}")
            return data
    except Exception:
        pass
    return None

def _save_threshold_cache(symbol, model_path, thr, pf, n, total, p_inf, p_sup):
    path = _cache_path(symbol)
    try:
        with open(path, "w") as f:
            json.dump({
                "model_mtime": os.path.getmtime(model_path),
                "threshold": thr, "pf": pf, "n": n,
                "total": total, "p_inf": p_inf, "p_sup": p_sup,
            }, f)
    except Exception as e:
        print(f"  [{symbol}] No se pudo guardar cache: {e}")

# ── Cálculo de umbral dinámico ────────────────────────────────

def calcular_umbral_optimo(model, SC, COLS, data_file, model_type, n_samples=10):
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

    normalizar_dataset(test_data, SC, model_type)
    test_data['tipo_encoded'] = test_data['tipo'].apply(lambda x: 1 if x == 'BUY' else 0)
    test_data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    test_data.fillna(0, inplace=True)

    X  = test_data[COLS].values
    yp = test_data['profit_norm'].values.reshape(-1, 1)
    yt = test_data['tipo_encoded'].values
    loader = DataLoader(DS(X, yp, yt), batch_size=64, shuffle=False)

    all_conf, all_real_p, all_pred_p = [], [], []
    for Xb, ypb, ytb in loader:
        pm, tm, conf = model.predict_with_uncertainty(Xb, n_samples=n_samples)
        all_conf.extend(conf.cpu().numpy())
        all_real_p.extend(ypb.cpu().numpy().flatten())
        all_pred_p.extend(pm.cpu().numpy().flatten())

    conf   = np.array(all_conf)
    real_p = denorm_array(np.array(all_real_p), SC['min_profit'], SC['max_profit'])
    pred_p = denorm_array(np.array(all_pred_p), SC['min_profit'], SC['max_profit'])
    total  = len(conf)

    # FIX: Para forex (pips_factor > 1), los modelos generan confianzas más bajas
    # por la naturaleza de los movimientos pequeños. Limitamos el umbral máximo
    # a 0.70 para evitar que queden 0 operaciones en producción.
    # Para índices mantenemos el rango completo hasta 0.95.
    pips_factor = SC.get("pips_factor", 1)
    is_forex    = pips_factor > 1
    thr_max     = 0.70 if is_forex else 0.95

    # FIX: Además de maximizar PF, penalizamos umbrales que dejan muy pocas ops.
    # score = PF * log(n_ops) — balanceo entre calidad y cobertura.
    # Esto evita que un umbral de 0.87 con 3 ops ganadoras "gane" sobre
    # un umbral de 0.52 con 150 ops y PF=1.4.
    best_thr, best_score, best_pf, best_n = 0.35, 0.0, 0.0, total
    for thr in np.arange(0.35, thr_max + 0.01, 0.01):
        mask  = conf > thr
        n_ops = mask.sum()
        if n_ops < total * 0.05: break   # mínimo 5% de operaciones
        fr = real_p[mask]
        winners, losers = fr[fr > 0], fr[fr < 0]
        if len(losers) == 0 or len(winners) == 0: continue
        pf    = abs(winners.sum() / losers.sum())
        score = pf * np.log(n_ops + 1)  # penaliza umbrales con pocas ops
        if score > best_score:
            best_score, best_pf, best_thr, best_n = score, pf, round(float(thr), 2), int(n_ops)

    p_inf = round(float(np.percentile(pred_p, 10)), 6)
    p_sup = round(float(np.percentile(pred_p, 90)), 6)
    tipo_activo = "FOREX" if is_forex else "ÍNDICE"
    print(f"  [{tipo_activo}] Umbral óptimo: {best_thr:.2f}  →  {best_n} ops ({best_n/total*100:.1f}%)  |  PF: {best_pf:.2f}  |  P10: {p_inf:.4f}  P90: {p_sup:.4f}")
    return best_thr, best_pf, best_n, total, p_inf, p_sup

# ── Carga de un modelo individual ────────────────────────────

def cargar_modelo(symbol, cfg):
    print(f"[{symbol}] Cargando...")
    try:
        model_path = os.path.join(cfg["model_dir"], cfg["model_file"])

        with open(os.path.join(cfg["model_dir"], cfg["scaler_file"]), "rb") as f: SC   = pickle.load(f)
        with open(os.path.join(cfg["model_dir"], cfg["cols_file"]),   "rb") as f: COLS = pickle.load(f)

        mt = detectar_tipo_modelo(SC)
        pips = SC.get("pips_factor", 1)
        print(f"  [{symbol}] Tipo: {mt}  |  pips_factor: {pips}")

        model = TradingModelV4(input_size=len(COLS), dropout=0.4)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        cached = _load_threshold_cache(symbol, model_path)
        if cached:
            thr   = cached["threshold"]
            pf    = cached["pf"]
            n     = cached["n"]
            total = cached["total"]
            p_inf = cached["p_inf"]
            p_sup = cached["p_sup"]
        else:
            print(f"  [{symbol}] Calculando umbral...")
            thr, pf, n, total, p_inf, p_sup = calcular_umbral_optimo(model, SC, COLS, cfg["data_file"], mt)
            _save_threshold_cache(symbol, model_path, thr, pf, n, total, p_inf, p_sup)

        min_thr = cfg.get("min_threshold", 0.35)
        thr_final = max(thr, min_thr)
        if thr_final == min_thr and thr < min_thr:
            print(f"  [{symbol}] Umbral óptimo {thr:.2f} < mínimo {min_thr:.2f} → usando {thr_final:.2f}")
        else:
            print(f"  [{symbol}] Umbral final: {thr_final:.2f}")

        print(f"  [{symbol}] Listo ✓")
        return symbol, {
            "model":      model,
            "SC":         SC,
            "COLS":       COLS,
            "threshold":  thr_final,
            "pf":         pf,
            "n_ops":      n,
            "total":      total,
            "p_inf":      p_inf,
            "p_sup":      p_sup,
            "model_type": mt,
            "pips_factor": pips,
        }
    except Exception as e:
        import traceback
        print(f"  [{symbol}] ERROR: {e}")
        traceback.print_exc()
        return symbol, None

# ── Cargar todos los modelos en paralelo ─────────────────────

MODELOS = {}

print("\nCargando modelos en paralelo...\n")
with ThreadPoolExecutor(max_workers=len(MODELOS_CONFIG)) as executor:
    futures = {executor.submit(cargar_modelo, sym, cfg): sym
               for sym, cfg in MODELOS_CONFIG.items()}
    for future in as_completed(futures):
        symbol, resultado = future.result()
        if resultado is not None:
            MODELOS[symbol] = resultado

print(f"\nAPI lista {API_VERSION} — modelos cargados: {sorted(MODELOS.keys())}\n")

# ── App ───────────────────────────────────────────────────────

app = FastAPI(title=f"Trading API Unificada {API_VERSION}")

@app.get("/predict")
def predict(
    symbol: str   = Query(...),
    fecha:  str   = Query(...),
    # 5m — todos los modelos
    o5:  float = Query(...), c5:  float = Query(...),
    h5:  float = Query(...), l5:  float = Query(...), v5: float = Query(...),
    r5:  float = Query(...), m5:  float = Query(...), s5: float = Query(...),
    # 15m — aceptados pero no usados en modelos standard
    o15: float = Query(None), c15: float = Query(None),
    h15: float = Query(None), l15: float = Query(None), v15: float = Query(None),
    r15: float = Query(None), m15: float = Query(None), s15: float = Query(None),
    # EMA/MACD/ADX 5m
    ema550:            float = Query(None), ema5200:           float = Query(None),
    ema50_prev:        float = Query(None), ema5200_prev:      float = Query(None),
    macdLine5:         float = Query(None), signalLine5:       float = Query(None),
    macdLine_prev5:    float = Query(None), signalLine_prev5:  float = Query(None),
    adx5:  float = Query(None), diPlus5:  float = Query(None), diMinus5:  float = Query(None),
    # Params extra — aceptados pero ignorados
    c5d:             float = Query(None),
    ema5015:         float = Query(None), ema20015:          float = Query(None),
    ema50_prev15:    float = Query(None), ema200_prev15:     float = Query(None),
    macdLine15:      float = Query(None), signalLine15:      float = Query(None),
    macdLine_prev15: float = Query(None), signalLine_prev15: float = Query(None),
    adx15: float = Query(None), diPlus15: float = Query(None), diMinus15: float = Query(None),
):
    sym = symbol.upper()
    if sym not in MODELOS:
        return JSONResponse({
            "valor_profit": 0.0, "RESULTADO": "IGNORE",
            "percentil_inf": 0.0, "percentil_sup": 0.0,
            "error": f"Símbolo '{sym}' no cargado",
        })

    m         = MODELOS[sym]
    s         = m["SC"]
    MODEL     = m["model"]
    COLS      = m["COLS"]
    THRESHOLD = m["threshold"]
    PIPS      = m["pips_factor"]

    try:
        dt = datetime.fromisoformat(fecha)
    except ValueError:
        raise HTTPException(400, f"Fecha invalida: '{fecha}'. Formato esperado: 2026-01-19T01:15:48")

    # ── Cache por clave sym+vela ──────────────────────────────
    # Corrige el bug anterior donde sym solo como clave hacía que
    # peticiones de velas históricas devolvieran la última vela procesada.
    minuto_vela = dt.replace(second=0, microsecond=0)
    minuto_vela = minuto_vela.replace(minute=(dt.minute // 5) * 5)
    clave_vela  = f"{sym}_{minuto_vela.strftime('%Y%m%d%H%M')}"

    cached = cache_get(clave_vela)
    if cached:
        return JSONResponse(cached)

    # ── FIX: corrección automática de h/l swapped desde el EA ────
    h5_fixed, l5_fixed = h5, l5
    if h5 == o5 and l5 == c5 and o5 != c5:
        h5_fixed = max(o5, c5)
        l5_fixed = min(o5, c5)

    h15_fixed, l15_fixed = h15, l15
    if h15 is not None and l15 is not None and o15 is not None and c15 is not None:
        if h15 == o15 and l15 == c15 and o15 != c15:
            h15_fixed = max(o15, c15)
            l15_fixed = min(o15, c15)

    h5, l5   = h5_fixed, l5_fixed
    h15, l15 = h15_fixed, l15_fixed

    # ── Construir features ────────────────────────────────────
    e550v   = ema550       if ema550       is not None else 0.0
    e5200v  = ema5200      if ema5200      is not None else 0.0
    e50pv   = ema50_prev   if ema50_prev   is not None else 0.0
    e5200pv = ema5200_prev if ema5200_prev is not None else 0.0
    mac5v   = macdLine5        if macdLine5        is not None else 0.0
    sig5v   = signalLine5      if signalLine5      is not None else 0.0
    mac5pv  = macdLine_prev5   if macdLine_prev5   is not None else 0.0
    sig5pv  = signalLine_prev5 if signalLine_prev5 is not None else 0.0
    adx5v   = adx5    if adx5    is not None else 0.0
    dip5v   = diPlus5 if diPlus5 is not None else 0.0
    dim5v   = diMinus5 if diMinus5 is not None else 0.0

    features = {
        "dia_semana":       dt.weekday() / 6.0,
        "hora":             dt.hour      / 23.0,
        "minuto":           dt.minute    / 55.0,
        "precioopen5":      norm_scalar(o5,      s["p1_precio5"],         s["p99_precio5"]),
        "precioclose5":     norm_scalar(c5,      s["p1_precio5"],         s["p99_precio5"]),
        "preciohigh5":      norm_scalar(h5,      s["p1_precio5"],         s["p99_precio5"]),
        "preciolow5":       norm_scalar(l5,      s["p1_precio5"],         s["p99_precio5"]),
        "volume5":          norm_scalar(v5,      s["p1_vol5"],            s["p99_vol5"]),
        "rsi5":             r5 / 100.0,
        "iStochaMain5":     m5 / 100.0,
        "iStochaSign5":     s5 / 100.0,
        "ema550":           norm_scalar(e550v,   s["p1_ema550"],          s["p99_ema550"]),
        "ema5200":          norm_scalar(e5200v,  s["p1_ema5200"],         s["p99_ema5200"]),
        "ema50_prev":       norm_scalar(e50pv,   s["p1_ema50_prev"],      s["p99_ema50_prev"]),
        "ema5200_prev":     norm_scalar(e5200pv, s["p1_ema5200_prev"],    s["p99_ema5200_prev"]),
        "macdLine5":        norm_scalar(mac5v,   s["p1_macdLine5"],       s["p99_macdLine5"]),
        "signalLine5":      norm_scalar(sig5v,   s["p1_signalLine5"],     s["p99_signalLine5"]),
        "macdLine_prev5":   norm_scalar(mac5pv,  s["p1_macdLine_prev5"],  s["p99_macdLine_prev5"]),
        "signalLine_prev5": norm_scalar(sig5pv,  s["p1_signalLine_prev5"],s["p99_signalLine_prev5"]),
        "adx5":    norm_scalar(adx5v, s["p1_adx5"],    s["p99_adx5"]),
        "diPlus5": norm_scalar(dip5v, s["p1_diPlus5"], s["p99_diPlus5"]),
        "diMinus5":norm_scalar(dim5v, s["p1_diMinus5"],s["p99_diMinus5"]),
    }

    vector   = np.array([features[col] for col in COLS], dtype=np.float32)
    x_tensor = torch.tensor(vector).unsqueeze(0)

    pred_profit, tipo_probs, confidence = MODEL.predict_with_uncertainty(x_tensor, n_samples=30)

    confidence = float(confidence.item())
    class_idx  = int(tipo_probs.argmax(dim=1).item())
    signal     = "BUY" if class_idx == 1 else "SELL"

    # profit_est está en pips si PIPS > 1 (forex), en puntos si PIPS = 1 (índices)
    profit_est_raw = denorm_scalar(pred_profit.item(), s["min_profit"], s["max_profit"])
    # Para forex: devolver en pips (dividir por pips_factor) para que sea comparable
    # Para índices: PIPS=1, no hay cambio
    profit_est = profit_est_raw / PIPS if PIPS > 1 else profit_est_raw

    valid = confidence >= THRESHOLD

    resultado = {
        "valor_profit":  round(profit_est, 6),
        "RESULTADO":     signal if valid else "IGNORE",
        "percentil_inf": m["p_inf"],
        "percentil_sup": m["p_sup"],
        "confidence":    round(confidence, 4),   # útil para debug
        "threshold":     round(THRESHOLD, 4),    # útil para debug
    }

    cache_set(clave_vela, resultado)
    return JSONResponse(resultado)

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "api_version":  API_VERSION,
        "startup_time": STARTUP_TIME,
        "cache_size":   len(CACHE_VELAS),
        "modelos": {
            sym: {
                "threshold":   m["threshold"],
                "pf":          round(m["pf"], 2),
                "ops":         f"{m['n_ops']}/{m['total']}",
                "p_inf":       m["p_inf"],
                "p_sup":       m["p_sup"],
                "model_type":  m["model_type"],
                "pips_factor": m["pips_factor"],
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