"""
API Unificada — NAS100 + US30 + GER40 + BTCUSD + AUDUSD + GBPAUD + EURUSD + GBPUSD
Inicio: uvicorn apiDataset:app --host 192.168.100.73 --port 80 --reload
CAMBIOS v6:
  - min_threshold diferenciado: índices=0.92, forex=0.50
CAMBIOS v6.6:
  - Cooldown por símbolo: después de una señal válida, las siguientes N velas
    devuelven IGNORE aunque superen el threshold (sin alterar thresholds).
    Forex/BTCUSD: 3 velas cooldown | Índices: 2 velas cooldown.
  - Campo "cooldown_activo" en respuesta para debug (True = señal bloqueada)
  - Los thresholds, confianzas y log forex no cambian en absoluto.
  - Cache de velas redondeado a vela de 5m (fix peticiones por segundo del EA)
  - Thresholds forex recalibrados según confianzas reales post-reentrenamiento
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
import threading

API_VERSION   = "v6.6"
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
    # ── FOREX: thresholds calibrados con datos reales de AnalisisForex.jsonl ────
    # Metodología: threshold debe estar entre P90 y max para capturar solo
    # picos reales de confianza, no el ruido base (~0.47 promedio).
    #
    # AUDUSD: promedio=0.74, P90=0.893, max=0.947 → threshold 0.88 ✅ correcto
    "AUDUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_AUDUSD_v4.pth",
        "scaler_file":   "scaler_AUDUSD_v4.pkl",
        "cols_file":     "input_columns_AUDUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_AUDUSD.xlsx",
        "min_threshold": 0.88,
    },
    # GBPAUD: promedio=0.463, P90=0.499, max=0.904
    #         Threshold 0.55 estaba por encima del P90 → casi nunca señalaba.
    #         Las 2 señales reales que salieron lo hicieron con conf=0.88-0.90.
    #         Nuevo threshold: 0.67 — captura solo picos reales (>P90+margen amplio).
    #         El cooldown evitará rachas si aparecen.
    "GBPAUD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GBPAUD_v4.pth",
        "scaler_file":   "scaler_GBPAUD_v4.pkl",
        "cols_file":     "input_columns_GBPAUD_v4.pkl",
        "data_file":     "Data_Entrenamiento_GBPAUD.xlsx",
        "min_threshold": 0.67,   # ↑ subido desde 0.55 — P90=0.50, señales reales en 0.88-0.90
    },
    # EURUSD: promedio=0.470, P90=0.492, max=0.899
    #         Las señales reales salen en picos de 0.87-0.90.
    #         Threshold 0.70 correcto — no tocar.
    "EURUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_EURUSD_v4.pth",
        "scaler_file":   "scaler_EURUSD_v4.pkl",
        "cols_file":     "input_columns_EURUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_EURUSD.xlsx",
        "min_threshold": 0.70,   # ✅ correcto — señales reales en 0.87-0.90
    },
    # GBPUSD: promedio=0.488, P90=0.541, max=0.909
    #         Threshold 0.80 era inalcanzable (solo 2 velas en 400 lo superaron).
    #         Las señales reales salen en 0.88-0.91.
    #         Nuevo threshold: 0.70 — por encima del P90 pero alcanzable en picos reales.
    "GBPUSD": {
        "model_dir":     os.path.join(BASE_DIR, "Trading_Modelv4"),
        "model_file":    "best_trading_model_GBPUSD_v4.pth",
        "scaler_file":   "scaler_GBPUSD_v4.pkl",
        "cols_file":     "input_columns_GBPUSD_v4.pkl",
        "data_file":     "Data_Entrenamiento_GBPUSD.xlsx",
        "min_threshold": 0.70,   # ↓ bajado desde 0.80 — P90=0.54, señales reales en 0.88-0.91
    },
}

# ── Cache de resultados por vela ──────────────────────────────
# Clave: "SYM_YYYYMMDDHHSS" — única por símbolo+vela
# Capacidad máxima: 200 entradas (LRU)
# Antes se usaba solo sym como clave → peticiones de velas distintas
# devolvían el resultado de la última vela procesada.
CACHE_VELAS: OrderedDict = OrderedDict()
CACHE_MAX = 500  # más entradas para no perder velas históricas

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

# ── Cooldown por símbolo ──────────────────────────────────────
# Evita señales consecutivas vela tras vela.
# Después de una señal válida (BUY/SELL), el símbolo entra en cooldown
# y las siguientes N velas devuelven IGNORE aunque superen el threshold.
#
# Forex:   COOLDOWN_VELAS_FOREX   = 3  (pares tienden a generar más seguidas)
# Índices: COOLDOWN_VELAS_INDICES = 2
#
# El cooldown NO altera thresholds ni confianzas — solo decide si publicar
# la señal al EA. El log forex sigue registrando todo normalmente.
#
# Estructura: { "EURUSD": {"last_signal_vela": "EURUSD_202601191315", "restantes": 2} }
COOLDOWN_VELAS_FOREX   = 3
COOLDOWN_VELAS_INDICES = 2
FOREX_SYMS_COOLDOWN    = {"AUDUSD", "GBPAUD", "EURUSD", "GBPUSD", "BTCUSD"}
_cooldown_state: dict  = {}
_cooldown_lock         = threading.Lock()

def cooldown_check_and_update(sym: str, clave_vela: str, tiene_señal: bool) -> bool:
    """
    Devuelve True si la señal debe publicarse, False si está en cooldown.
    Registra la nueva vela activa y decrementa el contador.
    """
    velas_cd = COOLDOWN_VELAS_FOREX if sym in FOREX_SYMS_COOLDOWN else COOLDOWN_VELAS_INDICES
    with _cooldown_lock:
        estado = _cooldown_state.get(sym)

        if estado and estado["last_vela"] != clave_vela:
            # Nueva vela → decrementar cooldown restante
            if estado["restantes"] > 0:
                estado["restantes"] -= 1
                _cooldown_state[sym] = estado

        if estado and estado["restantes"] > 0 and estado["last_vela"] != clave_vela:
            # Todavía en cooldown: bloquear señal
            return False

        if tiene_señal:
            # Señal válida → activar cooldown para las próximas N velas
            _cooldown_state[sym] = {"last_vela": clave_vela, "restantes": velas_cd}

    return True

# ── Sistema de log para diagnóstico forex ─────────────────────
# Registra cada predicción forex en AnalisisForex.jsonl
# Una línea JSON por predicción → fácil de leer y analizar
FOREX_SYMS   = {"AUDUSD", "GBPAUD", "EURUSD", "GBPUSD"}
LOG_PATH     = os.path.join(BASE_DIR, "AnalisisForex.jsonl")
_log_lock    = threading.Lock()

INDEX_SYMS    = {"NAS100", "GER40", "US30", "BTCUSD"}
LOG_INDEX_PATH = os.path.join(BASE_DIR, "AnalisisIndices.jsonl")
_log_index_lock = threading.Lock()

def log_forex(sym, fecha_str, o5, h5, l5, c5, v5,
              r5, m5, s5,
              ema550, ema5200, adx5, diPlus5, diMinus5,
              macdLine5, signalLine5,
              confidence, threshold, signal, resultado_final,
              profit_est, cooldown_bloqueado=False):
    """
    Escribe una línea JSON en AnalisisForex.jsonl por cada predicción forex.
    Campos añadidos v6.6:
      - cooldown_bloqueado: True si el cooldown suprimió esta señal
      - vela_body_pips:     tamaño del cuerpo de la vela en pips (c5-o5 normalizado)
      - tendencia_ema:      dirección EMA550 vs EMA5200 ("alcista"/"bajista"/"lateral")
      - macd_cruce:         True si hubo cruce MACD en esta vela
      - adx_fuerza:         "fuerte"(>25) / "moderado"(20-25) / "debil"(<20)
      - hora_sesion:        sesión de mercado estimada (Londres/NY/Asia/Overlap)
    """
    # ── Contexto técnico adicional ────────────────────────────────
    body_pips = round(abs(c5 - o5) * (10000 if c5 < 10 else 100), 2)  # pips aprox

    if ema550 is not None and ema5200 is not None:
        diff_ema = ema550 - ema5200
        if   diff_ema >  0.0001: tendencia_ema = "alcista"
        elif diff_ema < -0.0001: tendencia_ema = "bajista"
        else:                    tendencia_ema = "lateral"
    else:
        tendencia_ema = None

    if macdLine5 is not None and signalLine5 is not None:
        macd_cruce = (macdLine5 > signalLine5) != (macdLine5 < signalLine5)  # siempre True/False
        macd_dir   = "sobre_signal" if macdLine5 >= signalLine5 else "bajo_signal"
    else:
        macd_cruce = None
        macd_dir   = None

    if adx5 is not None:
        if   adx5 >= 25: adx_fuerza = "fuerte"
        elif adx5 >= 20: adx_fuerza = "moderado"
        else:            adx_fuerza = "debil"
    else:
        adx_fuerza = None

    try:
        hora_dt = datetime.fromisoformat(fecha_str)
        hora_utc = hora_dt.hour
        if   8 <= hora_utc < 12:  hora_sesion = "Londres"
        elif 12 <= hora_utc < 16: hora_sesion = "Overlap_LDN_NY"
        elif 16 <= hora_utc < 21: hora_sesion = "Nueva_York"
        else:                     hora_sesion = "Asia_Pacifico"
        dia_semana_n = hora_dt.weekday()   # 0=Lun, 4=Vie
    except Exception:
        hora_sesion  = None
        dia_semana_n = None

    entry = {
        "ts":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sym":        sym,
        "fecha":      fecha_str,
        # ── Contexto de sesión ────────────────────────────────────
        "hora_sesion":       hora_sesion,
        "dia_semana":        dia_semana_n,   # 0=Lun … 4=Vie
        # ── precios raw ───────────────────────────────────────────
        "o5": round(o5, 6), "h5": round(h5, 6),
        "l5": round(l5, 6), "c5": round(c5, 6), "v5": round(v5, 2),
        "body_pips": body_pips,
        # ── indicadores raw ───────────────────────────────────────
        "rsi5": round(r5, 2), "stoch_main": round(m5, 2), "stoch_sign": round(s5, 2),
        "ema550":  round(ema550,  6) if ema550  is not None else None,
        "ema5200": round(ema5200, 6) if ema5200 is not None else None,
        "tendencia_ema": tendencia_ema,
        "adx5":      round(adx5,   2) if adx5    is not None else None,
        "adx_fuerza":adx_fuerza,
        "diPlus5":   round(diPlus5,2) if diPlus5 is not None else None,
        "diMinus5":  round(diMinus5,2) if diMinus5 is not None else None,
        "macdLine5":   round(macdLine5,  6) if macdLine5   is not None else None,
        "signalLine5": round(signalLine5,6) if signalLine5 is not None else None,
        "macd_dir":    macd_dir,
        # ── resultado del modelo ──────────────────────────────────
        "confidence":        round(confidence, 4),
        "threshold":         round(threshold, 4),
        "gap_vs_threshold":  round(confidence - threshold, 4),
        "supera_umbral":     confidence >= threshold,
        "signal_modelo":     signal,
        "resultado_final":   resultado_final,
        "cooldown_bloqueado":cooldown_bloqueado,   # ← NUEVO v6.6
        "profit_est":        round(profit_est, 6),
    }
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def log_index(sym, fecha_str, o5, h5, l5, c5, v5,
              r5, m5, s5,
              ema550, ema5200, adx5, diPlus5, diMinus5,
              macdLine5, signalLine5,
              confidence, threshold, signal, resultado_final,
              profit_est, cooldown_bloqueado=False):
    """
    Registra cada predicción de índice en AnalisisIndices.jsonl.
    Mismo esquema que log_forex pero adaptado a índices:
      - body_pts: tamaño del cuerpo en puntos (sin pips_factor, ya que PIPS=1)
      - sesión basada en UTC igual que forex
      - adx_fuerza, tendencia_ema, macd_dir: idénticos a forex
    """
    # ── Cuerpo de la vela en puntos (índices no usan pips) ───────
    body_pts = round(abs(c5 - o5), 2)
 
    # ── Tendencia EMA ─────────────────────────────────────────────
    if ema550 is not None and ema5200 is not None:
        diff_ema = ema550 - ema5200
        if   diff_ema >  1.0: tendencia_ema = "alcista"   # umbral mayor para índices
        elif diff_ema < -1.0: tendencia_ema = "bajista"
        else:                 tendencia_ema = "lateral"
    else:
        tendencia_ema = None
 
    # ── MACD ──────────────────────────────────────────────────────
    if macdLine5 is not None and signalLine5 is not None:
        macd_dir = "sobre_signal" if macdLine5 >= signalLine5 else "bajo_signal"
    else:
        macd_dir = None
 
    # ── ADX fuerza ────────────────────────────────────────────────
    if adx5 is not None:
        if   adx5 >= 25: adx_fuerza = "fuerte"
        elif adx5 >= 20: adx_fuerza = "moderado"
        else:            adx_fuerza = "debil"
    else:
        adx_fuerza = None
 
    # ── Sesión de mercado ─────────────────────────────────────────
    # NAS100/US30: sesión principal NY (13:30-20:00 UTC)
    # GER40:       sesión principal Xetra (08:00-16:30 UTC)
    # BTCUSD:      24h — igual que forex
    try:
        hora_dt  = datetime.fromisoformat(fecha_str)
        hora_utc = hora_dt.hour
        dia_semana_n = hora_dt.weekday()  # 0=Lun, 4=Vie
 
        if sym in ("NAS100", "US30"):
            # Sesiones para índices americanos
            if   7 <= hora_utc < 13:  hora_sesion = "Pre_Market"
            elif 13 <= hora_utc < 17: hora_sesion = "NY_Open"
            elif 17 <= hora_utc < 20: hora_sesion = "NY_Core"
            elif 20 <= hora_utc < 21: hora_sesion = "NY_Close"
            else:                     hora_sesion = "Fuera_Sesion"
        elif sym == "GER40":
            # Sesiones para DAX
            if   6 <= hora_utc < 8:   hora_sesion = "Pre_Market"
            elif 8 <= hora_utc < 12:  hora_sesion = "Xetra_Open"
            elif 12 <= hora_utc < 15: hora_sesion = "Xetra_Core"
            elif 15 <= hora_utc < 17: hora_sesion = "Overlap_NY"
            else:                     hora_sesion = "Fuera_Sesion"
        else:  # BTCUSD — 24h, misma lógica que forex
            if   8 <= hora_utc < 12:  hora_sesion = "Londres"
            elif 12 <= hora_utc < 16: hora_sesion = "Overlap_LDN_NY"
            elif 16 <= hora_utc < 21: hora_sesion = "Nueva_York"
            else:                     hora_sesion = "Asia_Pacifico"
    except Exception:
        hora_sesion  = None
        dia_semana_n = None
 
    entry = {
        "ts":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sym":        sym,
        "fecha":      fecha_str,
        # ── Contexto de sesión ────────────────────────────────────
        "hora_sesion":       hora_sesion,
        "dia_semana":        dia_semana_n,   # 0=Lun … 4=Vie
        # ── precios raw ───────────────────────────────────────────
        "o5": round(o5, 2), "h5": round(h5, 2),
        "l5": round(l5, 2), "c5": round(c5, 2), "v5": round(v5, 2),
        "body_pts": body_pts,          # puntos, no pips
        # ── indicadores raw ───────────────────────────────────────
        "rsi5": round(r5, 2), "stoch_main": round(m5, 2), "stoch_sign": round(s5, 2),
        "ema550":  round(ema550,  2) if ema550  is not None else None,
        "ema5200": round(ema5200, 2) if ema5200 is not None else None,
        "tendencia_ema": tendencia_ema,
        "adx5":      round(adx5,   2) if adx5    is not None else None,
        "adx_fuerza":adx_fuerza,
        "diPlus5":   round(diPlus5,2) if diPlus5 is not None else None,
        "diMinus5":  round(diMinus5,2) if diMinus5 is not None else None,
        "macdLine5":   round(macdLine5,  2) if macdLine5   is not None else None,
        "signalLine5": round(signalLine5,2) if signalLine5 is not None else None,
        "macd_dir":    macd_dir,
        # ── resultado del modelo ──────────────────────────────────
        "confidence":        round(confidence, 4),
        "threshold":         round(threshold, 4),
        "gap_vs_threshold":  round(confidence - threshold, 4),
        "supera_umbral":     confidence >= threshold,
        "signal_modelo":     signal,
        "resultado_final":   resultado_final,
        "cooldown_bloqueado":cooldown_bloqueado,
        "profit_est_pts":    round(profit_est, 2),   # en puntos para índices
    }
    with _log_index_lock:
        with open(LOG_INDEX_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

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
    # Para forex (pips > 1), el profit en el dataset es pr2-pr1.
    # En SELL: pr2 < pr1 cuando gana → profit_raw negativo aunque es ganadora.
    # Los modelos de entrenamiento invierten el signo para SELL antes de normalizar.
    # Aquí hacemos lo mismo para que el umbral se calcule sobre el mismo espacio.
    profit_raw = df['profit'].fillna(0) * pips
    if pips > 1 and 'tipo' in df.columns:
        profit_raw = profit_raw.copy()
        profit_raw[df['tipo'] == 'SELL'] *= -1
    df['profit_original'] = profit_raw
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

# Incrementar esta versión cada vez que cambie la lógica de calcular_umbral_optimo
# o normalizar_dataset, para invalidar caches anteriores automáticamente.
THRESHOLD_CACHE_VERSION = 2  # v2: inversión de signo SELL en normalizar_dataset

def _load_threshold_cache(symbol, model_path):
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("cache_version", 1) != THRESHOLD_CACHE_VERSION:
            print(f"  [{symbol}] Cache obsoleto (v{data.get('cache_version',1)} → v{THRESHOLD_CACHE_VERSION}) → recalculando")
            return None
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
                "cache_version": THRESHOLD_CACHE_VERSION,
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

    # ── Cache por clave sym + vela de 5m (sin precios) ───────────
    # La clave es solo sym + fecha redondeada a la vela de 5m.
    # NO incluimos precios porque el EA envía precios levemente distintos
    # en cada tick de la misma vela (el c5 se actualiza cada segundo), lo que
    # hacía que cada petición calculara de nuevo aunque fuera la misma vela.
    # Con esta clave: la primera petición calcula y guarda, todas las siguientes
    # de esa misma vela devuelven el mismo resultado en cache.
    minuto_vela = dt.minute - (dt.minute % 5)
    clave_vela  = f"{sym}_{dt.strftime('%Y%m%d%H')}{minuto_vela:02d}"

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

    # ── Cooldown: evitar señales en velas consecutivas ────────────
    # Si el modelo supera el threshold pero el símbolo está en cooldown,
    # la señal se convierte en IGNORE para el EA. Los thresholds no cambian.
    tiene_señal = valid
    puede_publicar = cooldown_check_and_update(sym, clave_vela, tiene_señal)
    resultado_final = (signal if (valid and puede_publicar) else "IGNORE")

    resultado = {
        "valor_profit":  round(profit_est, 6),
        "RESULTADO":     resultado_final,
        "percentil_inf": m["p_inf"],
        "percentil_sup": m["p_sup"],
        # Nota: confidence, threshold y cooldown_activo se registran en el log
        # pero NO se devuelven en la respuesta al EA para mantener compatibilidad
        # con el parser del EA (SafeStringSplit3 basado en orden de campos).
    }

    # ── Log de diagnóstico para forex ────────────────────────────
    # Solo registra AUDUSD, GBPAUD, EURUSD, GBPUSD
    # El archivo AnalisisForex.jsonl se crea junto a la API
    if sym in FOREX_SYMS:
        log_forex(
            sym=sym, fecha_str=fecha,
            o5=o5, h5=h5, l5=l5, c5=c5, v5=v5,
            r5=r5, m5=m5, s5=s5,
            ema550=ema550, ema5200=ema5200,
            adx5=adx5, diPlus5=diPlus5, diMinus5=diMinus5,
            macdLine5=macdLine5, signalLine5=signalLine5,
            confidence=confidence, threshold=THRESHOLD,
            signal=signal, resultado_final=resultado_final,
            profit_est=profit_est,
            cooldown_bloqueado=(valid and not puede_publicar),
        )

    if sym in INDEX_SYMS:
        log_index(
            sym=sym, fecha_str=fecha,
            o5=o5, h5=h5, l5=l5, c5=c5, v5=v5,
            r5=r5, m5=m5, s5=s5,
            ema550=ema550, ema5200=ema5200,
            adx5=adx5, diPlus5=diPlus5, diMinus5=diMinus5,
            macdLine5=macdLine5, signalLine5=signalLine5,
            confidence=confidence, threshold=THRESHOLD,
            signal=signal, resultado_final=resultado_final,
            profit_est=profit_est,
            cooldown_bloqueado=(valid and not puede_publicar),
        )

    cache_set(clave_vela, resultado)
    return JSONResponse(resultado)

@app.get("/analisis_forex")
def analisis_forex(symbol: str = Query(None), ultimas: int = Query(200)):
    """
    Resumen de diagnóstico del log forex.
    Parámetros:
      symbol  — filtrar por par (AUDUSD, GBPAUD, EURUSD, GBPUSD). Opcional.
      ultimas — cuántas entradas recientes analizar (default 200).

    Secciones de respuesta (v6.6):
      1. resumen_confianza      — estadísticas de distribución de confianza
      2. señales                — cuántas llegaron al EA, bloqueadas por cooldown, etc.
      3. cooldown               — efectividad del cooldown: cuántas señales filtró
      4. rachas_consecutivas    — distribución de rachas de señales seguidas (antes del cooldown)
      5. sesiones               — señales y confianza media por sesión de mercado
      6. dias_semana            — señales por día (0=Lun … 4=Vie)
      7. sesgo_direccional      — BUY vs SELL por sesión y tendencia EMA
      8. indicadores_en_señales — RSI / ADX / MACD promedio cuando hay señal real
      9. diagnosis              — texto accionable
      10. ultimas_15_entradas   — tabla rápida de las últimas entradas
    """
    if not os.path.exists(LOG_PATH):
        return {"error": "Aún no hay datos. El archivo AnalisisForex.jsonl se crea con la primera petición forex."}

    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    if symbol:
        entries = [e for e in entries if e.get("sym","").upper() == symbol.upper()]

    entries = entries[-ultimas:]

    if not entries:
        return {"error": f"Sin datos para {'symbol='+symbol if symbol else 'todos los pares forex'}"}

    n = len(entries)

    # ── Helpers ───────────────────────────────────────────────────
    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    def pct(a, total):
        return round(a / total * 100, 1) if total else 0.0

    # ── Clasificaciones base ──────────────────────────────────────
    confs      = [e["confidence"]       for e in entries]
    gaps       = [e["gap_vs_threshold"] for e in entries]
    superan    = [e for e in entries if e.get("supera_umbral")]
    publicadas = [e for e in entries if e.get("resultado_final") in ("BUY","SELL")]
    bloqueadas_cd = [e for e in entries if e.get("cooldown_bloqueado", False)]
    buys_pub   = [e for e in publicadas if e.get("resultado_final") == "BUY"]
    sells_pub  = [e for e in publicadas if e.get("resultado_final") == "SELL"]
    threshold_usado = entries[-1].get("threshold") if entries else None

    # ── 1. Distribución de confianza ─────────────────────────────
    confs_s = sorted(confs)
    rangos = {"<0.40": 0, "0.40-0.50": 0, "0.50-0.60": 0,
              "0.60-0.70": 0, "0.70-0.80": 0, ">0.80": 0}
    for c in confs:
        if   c < 0.40: rangos["<0.40"] += 1
        elif c < 0.50: rangos["0.40-0.50"] += 1
        elif c < 0.60: rangos["0.50-0.60"] += 1
        elif c < 0.70: rangos["0.60-0.70"] += 1
        elif c < 0.80: rangos["0.70-0.80"] += 1
        else:          rangos[">0.80"] += 1

    resumen_confianza = {
        "minima":  round(min(confs), 4),
        "maxima":  round(max(confs), 4),
        "promedio":round(sum(confs)/n, 4),
        "P25":     round(confs_s[int(n*0.25)], 4),
        "P50":     round(confs_s[int(n*0.50)], 4),
        "P75":     round(confs_s[int(n*0.75)], 4),
        "P90":     round(confs_s[int(n*0.90)], 4),
        "distribucion": rangos,
    }

    # ── 2. Señales ────────────────────────────────────────────────
    señales = {
        "total_velas_analizadas":   n,
        "superan_umbral":           len(superan),
        "publicadas_al_EA":         len(publicadas),
        "bloqueadas_por_cooldown":  len(bloqueadas_cd),
        "ignoradas_por_threshold":  n - len(superan),
        "pct_publicadas":           pct(len(publicadas), n),
        "pct_superan_umbral":       pct(len(superan), n),
        "BUY_publicadas":           len(buys_pub),
        "SELL_publicadas":          len(sells_pub),
        "ratio_buy_sell":           round(len(buys_pub)/len(sells_pub), 2) if sells_pub else "∞",
    }

    # ── 3. Cooldown — efectividad ─────────────────────────────────
    cooldown_stats = {
        "señales_bloqueadas":          len(bloqueadas_cd),
        "señales_que_habrian_salido":  len(superan),
        "reduccion_pct":               pct(len(bloqueadas_cd), len(superan)) if superan else 0.0,
        "BUY_bloqueados":              sum(1 for e in bloqueadas_cd if e.get("signal_modelo")=="BUY"),
        "SELL_bloqueados":             sum(1 for e in bloqueadas_cd if e.get("signal_modelo")=="SELL"),
        "confianza_promedio_bloqueadas": safe_mean([e["confidence"] for e in bloqueadas_cd]),
        "confianza_promedio_publicadas": safe_mean([e["confidence"] for e in publicadas]),
        "nota": (
            "✅ Cooldown funcionando — filtra señales seguidas sin alterar thresholds."
            if bloqueadas_cd else
            "ℹ️ Sin señales bloqueadas por cooldown aún en este período."
        ),
    }

    # ── 4. Rachas consecutivas (pre-cooldown, sobre superan_umbral) ──
    # Detecta cuántas veces el modelo dio señales N veces seguidas
    rachas = []
    racha_actual = 0
    prev_supera  = False
    for e in entries:
        if e.get("supera_umbral"):
            racha_actual += 1
        else:
            if racha_actual > 0:
                rachas.append(racha_actual)
            racha_actual = 0
        prev_supera = e.get("supera_umbral", False)
    if racha_actual > 0:
        rachas.append(racha_actual)

    dist_rachas = {}
    for r in rachas:
        k = f"{r}_seguidas"
        dist_rachas[k] = dist_rachas.get(k, 0) + 1

    rachas_stats = {
        "total_rachas_detectadas": len(rachas),
        "racha_maxima":            max(rachas) if rachas else 0,
        "racha_promedio":          round(sum(rachas)/len(rachas), 2) if rachas else 0,
        "distribucion":            dict(sorted(dist_rachas.items())),
        "nota": (
            f"⚠️ Hay rachas de hasta {max(rachas)} señales seguidas — el cooldown las filtra."
            if rachas and max(rachas) >= 3 else
            "✅ Las rachas son cortas (≤2 seguidas)."
        ) if rachas else "ℹ️ Sin rachas detectadas en este período.",
    }

    # ── 5. Por sesión de mercado ──────────────────────────────────
    sesiones_orden = ["Londres", "Overlap_LDN_NY", "Nueva_York", "Asia_Pacifico"]
    sesiones = {}
    for s in sesiones_orden:
        grp = [e for e in entries if e.get("hora_sesion") == s]
        pub = [e for e in grp if e.get("resultado_final") in ("BUY","SELL")]
        sesiones[s] = {
            "velas":       len(grp),
            "publicadas":  len(pub),
            "pct_señal":   pct(len(pub), len(grp)),
            "conf_media":  safe_mean([e["confidence"] for e in grp]),
        }

    # ── 6. Por día de semana ──────────────────────────────────────
    dias_nombres = {0:"Lunes", 1:"Martes", 2:"Miercoles", 3:"Jueves", 4:"Viernes"}
    dias = {}
    for d in range(5):
        grp = [e for e in entries if e.get("dia_semana") == d]
        pub = [e for e in grp if e.get("resultado_final") in ("BUY","SELL")]
        if grp:
            dias[dias_nombres[d]] = {
                "velas":      len(grp),
                "publicadas": len(pub),
                "pct_señal":  pct(len(pub), len(grp)),
                "conf_media": safe_mean([e["confidence"] for e in grp]),
            }

    # ── 7. Sesgo direccional ──────────────────────────────────────
    sesgo = {}
    for s in sesiones_orden:
        grp_pub = [e for e in publicadas if e.get("hora_sesion") == s]
        if grp_pub:
            buys_s  = sum(1 for e in grp_pub if e.get("resultado_final")=="BUY")
            sells_s = sum(1 for e in grp_pub if e.get("resultado_final")=="SELL")
            sesgo[s] = {"BUY": buys_s, "SELL": sells_s,
                        "sesgo": "BUY" if buys_s > sells_s else ("SELL" if sells_s > buys_s else "NEUTRO")}

    # Sesgo por tendencia EMA
    for tend in ("alcista", "bajista", "lateral"):
        grp_t = [e for e in publicadas if e.get("tendencia_ema") == tend]
        if grp_t:
            b = sum(1 for e in grp_t if e.get("resultado_final")=="BUY")
            s_ = sum(1 for e in grp_t if e.get("resultado_final")=="SELL")
            sesgo[f"EMA_{tend}"] = {"BUY": b, "SELL": s_,
                                    "sesgo": "BUY" if b>s_ else ("SELL" if s_>b else "NEUTRO")}

    # ── 8. Indicadores en señales publicadas ──────────────────────
    rsis  = [e["rsi5"]  for e in publicadas if e.get("rsi5")  is not None]
    adxs  = [e["adx5"]  for e in publicadas if e.get("adx5")  is not None]
    bodys = [e.get("body_pips", 0) for e in publicadas]
    indicadores_señales = {
        "rsi5_promedio":       safe_mean(rsis),
        "adx5_promedio":       safe_mean(adxs),
        "body_pips_promedio":  safe_mean(bodys),
        "adx_fuerza_dist": {
            "fuerte":   sum(1 for e in publicadas if e.get("adx_fuerza")=="fuerte"),
            "moderado": sum(1 for e in publicadas if e.get("adx_fuerza")=="moderado"),
            "debil":    sum(1 for e in publicadas if e.get("adx_fuerza")=="debil"),
        },
        "macd_dir_dist": {
            "sobre_signal": sum(1 for e in publicadas if e.get("macd_dir")=="sobre_signal"),
            "bajo_signal":  sum(1 for e in publicadas if e.get("macd_dir")=="bajo_signal"),
        },
    }

    # ── 9. Diagnosis accionable ───────────────────────────────────
    problemas, sugerencias = [], []
    pct_pub = pct(len(publicadas), n)

    if pct_pub == 0:
        problemas.append(f"❌ 0% de velas generan señal. Threshold={threshold_usado} posiblemente demasiado alto.")
        sugerencias.append("Reducir min_threshold en MODELOS_CONFIG para este par.")
    elif pct_pub > 40:
        problemas.append(f"⚠️ {pct_pub}% de velas generan señal — demasiado frecuente.")
        sugerencias.append("Aumentar COOLDOWN_VELAS_FOREX o subir min_threshold.")
    elif pct_pub < 5:
        problemas.append(f"⚠️ Solo {pct_pub}% de velas generan señal — puede ser insuficiente.")
        sugerencias.append("Bajar ligeramente min_threshold o COOLDOWN_VELAS_FOREX.")

    if señales["ratio_buy_sell"] != "∞":
        rb = señales["ratio_buy_sell"]
        if rb > 3:
            problemas.append(f"⚠️ Fuerte sesgo BUY (ratio {rb}:1). El modelo puede necesitar reentrenamiento con dataset balanceado.")
        elif rb < 0.33:
            problemas.append(f"⚠️ Fuerte sesgo SELL (ratio {rb}:1). Mismo diagnóstico.")

    if rachas and rachas_stats["racha_maxima"] >= 5:
        problemas.append(f"⚠️ Rachas de hasta {rachas_stats['racha_maxima']} señales seguidas detectadas antes del cooldown.")
        sugerencias.append(f"Considerar aumentar COOLDOWN_VELAS_FOREX a {rachas_stats['racha_maxima']-1} o más.")

    if cooldown_stats["confianza_promedio_bloqueadas"] and cooldown_stats["confianza_promedio_publicadas"]:
        if cooldown_stats["confianza_promedio_bloqueadas"] > cooldown_stats["confianza_promedio_publicadas"]:
            problemas.append("⚠️ Las señales bloqueadas por cooldown tienen confianza MAYOR que las publicadas — el cooldown puede estar filtrando las mejores señales.")
            sugerencias.append("Reducir COOLDOWN_VELAS_FOREX o revisar la lógica de cooldown.")

    diagnosis = {
        "problemas":    problemas if problemas else ["✅ Sin problemas detectados en este período."],
        "sugerencias":  sugerencias if sugerencias else ["✅ Configuración estable."],
    }

    # ── 10. Últimas 15 entradas ───────────────────────────────────
    campos_tabla = ["ts","sym","fecha","confidence","threshold","gap_vs_threshold",
                    "supera_umbral","cooldown_bloqueado","resultado_final",
                    "hora_sesion","adx_fuerza","tendencia_ema","body_pips"]
    ultimas_entradas = [
        {k: e.get(k) for k in campos_tabla}
        for e in entries[-15:]
    ]

    return {
        "symbol_filtrado":          symbol or "TODOS_FOREX",
        "periodo_analizado":        f"{entries[0].get('ts','?')} → {entries[-1].get('ts','?')}",
        "threshold_actual":         threshold_usado,
        "resumen_confianza":        resumen_confianza,
        "señales":                  señales,
        "cooldown":                 cooldown_stats,
        "rachas_consecutivas":      rachas_stats,
        "por_sesion":               sesiones,
        "por_dia_semana":           dias,
        "sesgo_direccional":        sesgo,
        "indicadores_en_señales":   indicadores_señales,
        "gap_promedio_vs_threshold":round(sum(gaps)/n, 4),
        "diagnosis":                diagnosis,
        "ultimas_15_entradas":      ultimas_entradas,
    }

@app.get("/analisis_indices")
def analisis_indices(symbol: str = Query(None), ultimas: int = Query(200)):
    """
    Resumen de diagnóstico del log de índices (NAS100, GER40, US30, BTCUSD).
    Parámetros:
      symbol  — filtrar por índice. Opcional. Ej: symbol=NAS100
      ultimas — cuántas entradas recientes analizar (default 200).
    """
    if not os.path.exists(LOG_INDEX_PATH):
        return {"error": "Aún no hay datos. El archivo AnalisisIndices.jsonl se crea con la primera petición de índice."}
 
    entries = []
    with open(LOG_INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
 
    if symbol:
        entries = [e for e in entries if e.get("sym", "").upper() == symbol.upper()]
 
    entries = entries[-ultimas:]
 
    if not entries:
        return {"error": f"Sin datos para {'symbol='+symbol if symbol else 'todos los índices'}"}
 
    n = len(entries)
 
    # ── Helpers ───────────────────────────────────────────────────
    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None
 
    def pct(a, total):
        return round(a / total * 100, 1) if total else 0.0
 
    # ── Clasificaciones base ──────────────────────────────────────
    confs      = [e["confidence"]       for e in entries]
    gaps       = [e["gap_vs_threshold"] for e in entries]
    superan    = [e for e in entries if e.get("supera_umbral")]
    publicadas = [e for e in entries if e.get("resultado_final") in ("BUY", "SELL")]
    bloqueadas_cd = [e for e in entries if e.get("cooldown_bloqueado", False)]
    buys_pub   = [e for e in publicadas if e.get("resultado_final") == "BUY"]
    sells_pub  = [e for e in publicadas if e.get("resultado_final") == "SELL"]
    threshold_usado = entries[-1].get("threshold") if entries else None
 
    # ── 1. Distribución de confianza ─────────────────────────────
    confs_s = sorted(confs)
    rangos = {"<0.40": 0, "0.40-0.50": 0, "0.50-0.60": 0,
              "0.60-0.70": 0, "0.70-0.80": 0, "0.80-0.90": 0, ">0.90": 0}
    for c in confs:
        if   c < 0.40: rangos["<0.40"] += 1
        elif c < 0.50: rangos["0.40-0.50"] += 1
        elif c < 0.60: rangos["0.50-0.60"] += 1
        elif c < 0.70: rangos["0.60-0.70"] += 1
        elif c < 0.80: rangos["0.70-0.80"] += 1
        elif c < 0.90: rangos["0.80-0.90"] += 1
        else:          rangos[">0.90"] += 1
 
    resumen_confianza = {
        "minima":   round(min(confs), 4),
        "maxima":   round(max(confs), 4),
        "promedio": round(sum(confs) / n, 4),
        "P25":      round(confs_s[int(n * 0.25)], 4),
        "P50":      round(confs_s[int(n * 0.50)], 4),
        "P75":      round(confs_s[int(n * 0.75)], 4),
        "P90":      round(confs_s[int(n * 0.90)], 4),
        "distribucion": rangos,
    }
 
    # ── 2. Señales ────────────────────────────────────────────────
    señales = {
        "total_velas_analizadas":  n,
        "superan_umbral":          len(superan),
        "publicadas_al_EA":        len(publicadas),
        "bloqueadas_por_cooldown": len(bloqueadas_cd),
        "ignoradas_por_threshold": n - len(superan),
        "pct_publicadas":          pct(len(publicadas), n),
        "pct_superan_umbral":      pct(len(superan), n),
        "BUY_publicadas":          len(buys_pub),
        "SELL_publicadas":         len(sells_pub),
        "ratio_buy_sell":          round(len(buys_pub) / len(sells_pub), 2) if sells_pub else "∞",
    }
 
    # ── 3. Cooldown — efectividad ─────────────────────────────────
    cooldown_stats = {
        "señales_bloqueadas":            len(bloqueadas_cd),
        "señales_que_habrian_salido":    len(superan),
        "reduccion_pct":                 pct(len(bloqueadas_cd), len(superan)) if superan else 0.0,
        "BUY_bloqueados":                sum(1 for e in bloqueadas_cd if e.get("signal_modelo") == "BUY"),
        "SELL_bloqueados":               sum(1 for e in bloqueadas_cd if e.get("signal_modelo") == "SELL"),
        "confianza_promedio_bloqueadas": safe_mean([e["confidence"] for e in bloqueadas_cd]),
        "confianza_promedio_publicadas": safe_mean([e["confidence"] for e in publicadas]),
        "nota": (
            "✅ Cooldown funcionando — filtra señales seguidas sin alterar thresholds."
            if bloqueadas_cd else
            "ℹ️ Sin señales bloqueadas por cooldown aún en este período."
        ),
    }
 
    # ── 4. Rachas consecutivas ────────────────────────────────────
    rachas = []
    racha_actual = 0
    for e in entries:
        if e.get("supera_umbral"):
            racha_actual += 1
        else:
            if racha_actual > 0:
                rachas.append(racha_actual)
            racha_actual = 0
    if racha_actual > 0:
        rachas.append(racha_actual)
 
    dist_rachas = {}
    for r in rachas:
        k = f"{r}_seguidas"
        dist_rachas[k] = dist_rachas.get(k, 0) + 1
 
    rachas_stats = {
        "total_rachas_detectadas": len(rachas),
        "racha_maxima":            max(rachas) if rachas else 0,
        "racha_promedio":          round(sum(rachas) / len(rachas), 2) if rachas else 0,
        "distribucion":            dict(sorted(dist_rachas.items())),
        "nota": (
            f"⚠️ Rachas de hasta {max(rachas)} señales seguidas — el cooldown las filtra."
            if rachas and max(rachas) >= 3 else
            "✅ Las rachas son cortas (≤2 seguidas)."
        ) if rachas else "ℹ️ Sin rachas detectadas en este período.",
    }
 
    # ── 5. Por sesión de mercado ──────────────────────────────────
    # Detectar qué símbolo/grupo de sesiones usar
    sym_actual = symbol.upper() if symbol else None
 
    if sym_actual in ("NAS100", "US30"):
        sesiones_orden = ["Pre_Market", "NY_Open", "NY_Core", "NY_Close", "Fuera_Sesion"]
    elif sym_actual == "GER40":
        sesiones_orden = ["Pre_Market", "Xetra_Open", "Xetra_Core", "Overlap_NY", "Fuera_Sesion"]
    elif sym_actual == "BTCUSD":
        sesiones_orden = ["Londres", "Overlap_LDN_NY", "Nueva_York", "Asia_Pacifico"]
    else:
        # Sin filtro de símbolo: incluir todas las sesiones posibles
        sesiones_orden = ["Pre_Market", "NY_Open", "NY_Core", "NY_Close",
                          "Xetra_Open", "Xetra_Core", "Overlap_NY",
                          "Londres", "Overlap_LDN_NY", "Nueva_York",
                          "Asia_Pacifico", "Fuera_Sesion"]
 
    sesiones = {}
    for s in sesiones_orden:
        grp = [e for e in entries if e.get("hora_sesion") == s]
        if not grp:
            continue
        pub = [e for e in grp if e.get("resultado_final") in ("BUY", "SELL")]
        sesiones[s] = {
            "velas":      len(grp),
            "publicadas": len(pub),
            "pct_señal":  pct(len(pub), len(grp)),
            "conf_media": safe_mean([e["confidence"] for e in grp]),
        }
 
    # ── 6. Por día de semana ──────────────────────────────────────
    dias_nombres = {0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves", 4: "Viernes"}
    dias = {}
    for d in range(5):
        grp = [e for e in entries if e.get("dia_semana") == d]
        pub = [e for e in grp if e.get("resultado_final") in ("BUY", "SELL")]
        if grp:
            dias[dias_nombres[d]] = {
                "velas":      len(grp),
                "publicadas": len(pub),
                "pct_señal":  pct(len(pub), len(grp)),
                "conf_media": safe_mean([e["confidence"] for e in grp]),
            }
 
    # ── 7. Sesgo direccional ──────────────────────────────────────
    sesgo = {}
    for s in sesiones_orden:
        grp_pub = [e for e in publicadas if e.get("hora_sesion") == s]
        if grp_pub:
            buys_s  = sum(1 for e in grp_pub if e.get("resultado_final") == "BUY")
            sells_s = sum(1 for e in grp_pub if e.get("resultado_final") == "SELL")
            sesgo[s] = {
                "BUY": buys_s, "SELL": sells_s,
                "sesgo": "BUY" if buys_s > sells_s else ("SELL" if sells_s > buys_s else "NEUTRO"),
            }
 
    for tend in ("alcista", "bajista", "lateral"):
        grp_t = [e for e in publicadas if e.get("tendencia_ema") == tend]
        if grp_t:
            b  = sum(1 for e in grp_t if e.get("resultado_final") == "BUY")
            s_ = sum(1 for e in grp_t if e.get("resultado_final") == "SELL")
            sesgo[f"EMA_{tend}"] = {
                "BUY": b, "SELL": s_,
                "sesgo": "BUY" if b > s_ else ("SELL" if s_ > b else "NEUTRO"),
            }
 
    # ── 8. Indicadores en señales publicadas ─────────────────────
    rsis  = [e["rsi5"]    for e in publicadas if e.get("rsi5")    is not None]
    adxs  = [e["adx5"]    for e in publicadas if e.get("adx5")    is not None]
    bodys = [e.get("body_pts", 0) for e in publicadas]
    indicadores_señales = {
        "rsi5_promedio":        safe_mean(rsis),
        "adx5_promedio":        safe_mean(adxs),
        "body_pts_promedio":    safe_mean(bodys),
        "adx_fuerza_dist": {
            "fuerte":   sum(1 for e in publicadas if e.get("adx_fuerza") == "fuerte"),
            "moderado": sum(1 for e in publicadas if e.get("adx_fuerza") == "moderado"),
            "debil":    sum(1 for e in publicadas if e.get("adx_fuerza") == "debil"),
        },
        "macd_dir_dist": {
            "sobre_signal": sum(1 for e in publicadas if e.get("macd_dir") == "sobre_signal"),
            "bajo_signal":  sum(1 for e in publicadas if e.get("macd_dir") == "bajo_signal"),
        },
    }
 
    # ── 9. Ventana óptima ─────────────────────────────────────────
    # Detecta la sesión + día con mayor % de señal y mayor confianza media
    mejor_sesion    = None
    mejor_sesion_pct = -1
    mejor_dia       = None
    mejor_dia_pct   = -1
 
    for s, data in sesiones.items():
        if data["velas"] >= 10 and data["pct_señal"] > mejor_sesion_pct:
            mejor_sesion_pct = data["pct_señal"]
            mejor_sesion     = s
 
    for d, data in dias.items():
        if data["velas"] >= 5 and data["pct_señal"] > mejor_dia_pct:
            mejor_dia_pct = data["pct_señal"]
            mejor_dia     = d
 
    # Mapeo de sesión a horario UTC legible
    sesion_horario = {
        "Pre_Market":     "07:00–13:00 UTC",
        "NY_Open":        "13:30–17:00 UTC",
        "NY_Core":        "17:00–20:00 UTC",
        "NY_Close":       "20:00–21:00 UTC",
        "Fuera_Sesion":   "21:00–07:00 UTC",
        "Xetra_Open":     "08:00–12:00 UTC",
        "Xetra_Core":     "12:00–15:00 UTC",
        "Overlap_NY":     "15:00–17:00 UTC",
        "Londres":        "08:00–12:00 UTC",
        "Overlap_LDN_NY": "12:00–16:00 UTC",
        "Nueva_York":     "16:00–21:00 UTC",
        "Asia_Pacifico":  "00:00–08:00 UTC",
    }
    horario_str = sesion_horario.get(mejor_sesion, "")
 
    if mejor_sesion and mejor_dia:
        ventana_optima = {
            "sesion":   mejor_sesion,
            "horario":  horario_str,
            "dia":      mejor_dia,
            "resumen":  (
                f"Ventana óptima: {mejor_sesion} ({horario_str}), {mejor_dia}. "
                f"Considera ajustar el EA para que esté más activo en esa ventana."
            ),
        }
    else:
        ventana_optima = {"resumen": "Insuficientes datos para determinar ventana óptima (mín. 10 velas por sesión)."}
 
    # ── 10. Diagnosis accionable ──────────────────────────────────
    problemas, sugerencias = [], []
    pct_pub = pct(len(publicadas), n)
 
    if pct_pub == 0:
        problemas.append(f"❌ 0% de velas generan señal. Threshold={threshold_usado} posiblemente demasiado alto.")
        sugerencias.append("Reducir min_threshold en MODELOS_CONFIG para este índice.")
    elif pct_pub > 30:
        problemas.append(f"⚠️ {pct_pub}% de velas generan señal — alta frecuencia para un índice.")
        sugerencias.append("Aumentar COOLDOWN_VELAS_INDICES o subir min_threshold.")
    elif pct_pub < 2:
        problemas.append(f"⚠️ Solo {pct_pub}% de velas generan señal — muy poco para operar.")
        sugerencias.append("Bajar ligeramente min_threshold (con cuidado — índices requieren alta confianza).")
 
    if señales["ratio_buy_sell"] != "∞":
        rb = señales["ratio_buy_sell"]
        if rb > 3:
            problemas.append(f"⚠️ Fuerte sesgo BUY (ratio {rb}:1). Verificar si el período analizado fue tendencia alcista o hay sesgo del modelo.")
        elif rb < 0.33:
            problemas.append(f"⚠️ Fuerte sesgo SELL (ratio {rb}:1). Mismo diagnóstico.")
 
    if rachas and rachas_stats["racha_maxima"] >= 4:
        problemas.append(f"⚠️ Rachas de hasta {rachas_stats['racha_maxima']} señales seguidas detectadas.")
        sugerencias.append(f"Considerar aumentar COOLDOWN_VELAS_INDICES a {rachas_stats['racha_maxima'] - 1} o más.")
 
    if cooldown_stats["confianza_promedio_bloqueadas"] and cooldown_stats["confianza_promedio_publicadas"]:
        if cooldown_stats["confianza_promedio_bloqueadas"] > cooldown_stats["confianza_promedio_publicadas"]:
            problemas.append("⚠️ Las señales bloqueadas por cooldown tienen confianza MAYOR que las publicadas.")
            sugerencias.append("Reducir COOLDOWN_VELAS_INDICES o revisar lógica de cooldown para este índice.")
 
    # Gap vs threshold
    gap_prom = round(sum(gaps) / n, 4)
    if gap_prom < -0.10:
        problemas.append(f"⚠️ Gap promedio vs threshold: {gap_prom} — la mayoría de velas quedan lejos del umbral.")
        sugerencias.append("Bajar min_threshold gradualmente (0.02 a la vez) y monitorear PF.")
 
    diagnosis = {
        "problemas":   problemas if problemas else ["✅ Sin problemas detectados en este período."],
        "sugerencias": sugerencias if sugerencias else ["✅ Configuración estable."],
    }
 
    # ── 11. Últimas 15 entradas ───────────────────────────────────
    campos_tabla = ["ts", "sym", "fecha", "confidence", "threshold", "gap_vs_threshold",
                    "supera_umbral", "cooldown_bloqueado", "resultado_final",
                    "hora_sesion", "adx_fuerza", "tendencia_ema", "body_pts"]
    ultimas_entradas = [
        {k: e.get(k) for k in campos_tabla}
        for e in entries[-15:]
    ]
 
    return {
        "symbol_filtrado":          symbol or "TODOS_INDICES",
        "periodo_analizado":        f"{entries[0].get('ts', '?')} → {entries[-1].get('ts', '?')}",
        "threshold_actual":         threshold_usado,
        "resumen_confianza":        resumen_confianza,
        "señales":                  señales,
        "cooldown":                 cooldown_stats,
        "rachas_consecutivas":      rachas_stats,
        "por_sesion":               sesiones,
        "por_dia_semana":           dias,
        "sesgo_direccional":        sesgo,
        "indicadores_en_señales":   indicadores_señales,
        "gap_promedio_vs_threshold": gap_prom,
        "ventana_optima":           ventana_optima,
        "diagnosis":                diagnosis,
        "ultimas_15_entradas":      ultimas_entradas,
    }

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "api_version":  API_VERSION,
        "startup_time": STARTUP_TIME,
        "cache_size":   len(CACHE_VELAS),
        "modelos": {
            sym: {
                "threshold":   float(m["threshold"]),
                "pf":          round(float(m["pf"]), 2),
                "ops":         f"{m['n_ops']}/{m['total']}",
                "p_inf":       float(m["p_inf"]),
                "p_sup":       float(m["p_sup"]),
                "model_type":  m["model_type"],
                "pips_factor": float(m["pips_factor"]),
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