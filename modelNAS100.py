"""
Modelo v4 NAS100 — Solo datos 5m
Cambios v4.1:
  - EarlyStopping guarda por profit factor filtrado (threshold 0.85) en lugar de val_loss
  - Peso clasificacion subido de 0.3 a 0.5 (direccion es critica en NAS100)
  - evaluate_model reporta metricas a multiples thresholds (0.70, 0.80, 0.92)
  - Confianza normalizada por percentil para estabilidad entre corridas
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "DataFiles")
MODEL_DIR = os.path.join(BASE_DIR, "Trading_Modelv4")
BATCH_SIZE  = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold de referencia para early stopping — mismo nivel que usa la API
ES_THRESHOLD = 0.85

def normalize_robust(column, p1, p99):
    if p99 - p1 == 0:
        return column * 0
    return (2 * (column - p1) / (p99 - p1) - 1).clip(-1.5, 1.5)

def denormalize(column, min_val, max_val):
    return (column + 1) * (max_val - min_val) / 2 + min_val

def normalize_confidence(conf):
    """
    Normaliza la distribucion de confianza al rango [0, 1] usando percentiles
    para que sea estable entre corridas y comparable entre modelos.
    """
    p_low  = np.percentile(conf, 5)
    p_high = np.percentile(conf, 95)
    if p_high - p_low < 1e-6:
        return conf
    norm = (conf - p_low) / (p_high - p_low)
    return np.clip(norm, 0.0, 1.0)

class TradingDataset(Dataset):
    def __init__(self, X, y_profit, y_tipo):
        self.X        = torch.tensor(X,        dtype=torch.float32)
        self.y_profit = torch.tensor(y_profit, dtype=torch.float32)
        self.y_tipo   = torch.tensor(y_tipo,   dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_profit[idx], self.y_tipo[idx]

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

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
        self.eval()
        enable_dropout(self)
        profit_preds, tipo_preds = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                p, t = self.forward(x)
                profit_preds.append(p.unsqueeze(0))
                tipo_preds.append(torch.softmax(t, dim=1).unsqueeze(0))
        profit_stack = torch.cat(profit_preds, dim=0)
        tipo_stack   = torch.cat(tipo_preds,   dim=0)
        profit_mean  = profit_stack.mean(dim=0)
        tipo_mean    = tipo_stack.mean(dim=0)
        # confianza raw: mean - std (antes de normalizar)
        conf = tipo_mean.max(dim=1).values - tipo_stack.std(dim=0).max(dim=1).values
        return profit_mean, tipo_mean, conf


class EarlyStoppingPF:
    """
    Guarda el modelo cuando mejora el Profit Factor a threshold ES_THRESHOLD,
    no cuando baja el val_loss. Esto alinea el criterio de guardado con lo que
    realmente importa en produccion.
    Si no hay suficientes operaciones filtradas cae back a val_loss.
    """
    def __init__(self, patience=30, min_delta=0.001, min_ops=10):
        self.patience   = patience
        self.min_delta  = min_delta
        self.min_ops    = min_ops          # minimo de ops filtradas para usar PF
        self.counter    = 0
        self.best_score = None             # PF o -val_loss segun disponibilidad
        self.early_stop = False
        self.using_pf   = False            # para mostrar en log

    def compute_pf_at_threshold(self, model, dataloader, min_profit, max_profit, threshold):
        """Calcula PF rapido sin Monte Carlo completo (1 sample) para el ES."""
        model.eval()
        reals = []
        with torch.no_grad():
            for X_b, yp_b, _ in dataloader:
                pred_p, tipo_probs, conf_raw = model.predict_with_uncertainty(
                    X_b.to(DEVICE), n_samples=10   # menos samples para velocidad
                )
                conf_norm = normalize_confidence(conf_raw.cpu().numpy())
                mask      = conf_norm > threshold
                if mask.sum() > 0:
                    real_p = denormalize(yp_b.numpy()[mask], min_profit, max_profit)
                    reals.extend(real_p.tolist())
        if len(reals) < self.min_ops:
            return None
        arr     = np.array(reals)
        winners = arr[arr > 0]
        losers  = arr[arr < 0]
        if len(losers) == 0 or len(winners) == 0:
            return None
        return abs(winners.sum() / losers.sum())

    def __call__(self, val_loss, model, path, dataloader=None,
                 min_profit=None, max_profit=None):
        pf = None
        if dataloader is not None:
            pf = self.compute_pf_at_threshold(
                model, dataloader, min_profit, max_profit, ES_THRESHOLD
            )

        if pf is not None:
            score       = pf
            self.using_pf = True
        else:
            score         = -val_loss     # negativo: mayor es mejor
            self.using_pf = False

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            torch.save(model.state_dict(), path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def evaluate_model(model, dataloader, min_profit, max_profit, label,
                   thresholds=(0.70, 0.80, 0.92)):
    """
    Evalua el modelo y reporta metricas a multiples thresholds para ver
    como se comporta en cada nivel de confianza (especialmente el 0.92 de produccion).
    """
    all_pred_profit, all_real_profit = [], []
    all_pred_tipo,   all_real_tipo   = [], []
    all_confidence_raw = []

    for X_batch, yp_batch, yt_batch in dataloader:
        pred_profit, tipo_probs, confidence = model.predict_with_uncertainty(
            X_batch.to(DEVICE)
        )
        pred_profit = pred_profit.cpu()
        tipo_probs  = tipo_probs.cpu()
        confidence  = confidence.cpu()
        all_pred_profit.extend(pred_profit.numpy().flatten())
        all_real_profit.extend(yp_batch.numpy().flatten())
        all_pred_tipo.extend(tipo_probs.argmax(dim=1).numpy())
        all_real_tipo.extend(yt_batch.numpy())
        all_confidence_raw.extend(confidence.numpy().flatten())

    pred_p   = denormalize(np.array(all_pred_profit), min_profit, max_profit)
    real_p   = denormalize(np.array(all_real_profit), min_profit, max_profit)
    pred_t   = np.array(all_pred_tipo)
    real_t   = np.array(all_real_tipo)
    conf_raw = np.array(all_confidence_raw)

    # Normalizar confianza con percentiles para comparabilidad
    conf = normalize_confidence(conf_raw)

    mae      = np.mean(np.abs(real_p - pred_p))
    accuracy = np.mean(pred_t == real_t)

    print(f"\n── {label.upper()} ─────────────────────────────────")
    print(f"  Clasificacion   → Accuracy: {accuracy*100:.1f}%  |  MAE profit: ${mae:.0f}")
    print(f"  Confianza norm  → min: {conf.min():.3f}  max: {conf.max():.3f}  "
          f"media: {conf.mean():.3f}  p50: {np.percentile(conf,50):.3f}")

    best_result = None

    for thr in thresholds:
        mask      = conf > thr
        n_ops     = mask.sum()
        pct_ops   = n_ops / len(conf) * 100

        if n_ops == 0:
            print(f"\n  [thr={thr:.2f}]  Sin operaciones filtradas")
            continue

        filt_real = real_p[mask]
        filt_acc  = np.mean(pred_t[mask] == real_t[mask])
        total_pf  = filt_real.sum()
        win_rate  = np.sum(filt_real > 0) / n_ops * 100
        winners   = filt_real[filt_real > 0]
        losers    = filt_real[filt_real < 0]
        pf        = abs(winners.sum() / losers.sum()) if len(losers) > 0 and len(winners) > 0 else 0

        buys  = np.sum(pred_t[mask] == 1)
        sells = np.sum(pred_t[mask] == 0)
        b_pct = buys  / n_ops * 100
        s_pct = sells / n_ops * 100

        # Marcar el threshold de produccion
        prod_marker = "  ← PRODUCCION" if thr == 0.92 else ""

        print(f"\n  [thr={thr:.2f}]{prod_marker}")
        print(f"    Ops          → {n_ops}/{len(conf)} ({pct_ops:.0f}%)")
        print(f"    Senales      → BUY: {buys} ({b_pct:.1f}%)  SELL: {sells} ({s_pct:.1f}%)")
        print(f"    Acc filtrado → {filt_acc*100:.1f}%")
        print(f"    Trading      → Profit: ${total_pf:,.0f}  |  Win rate: {win_rate:.1f}%  |  PF: {pf:.2f}")

        if thr == 0.92:
            best_result = {
                'mae': mae, 'accuracy': accuracy,
                'confidence': conf, 'conf_raw': conf_raw,
                'pred_tipo': pred_t, 'real_tipo': real_t,
                'pred_profit': pred_p, 'real_profit': real_p,
                'pf_092': pf, 'ops_092': n_ops,
            }

    # Si no habia resultado a 0.92 devolver igual con datos completos
    if best_result is None:
        best_result = {
            'mae': mae, 'accuracy': accuracy,
            'confidence': conf, 'conf_raw': conf_raw,
            'pred_tipo': pred_t, 'real_tipo': real_t,
            'pred_profit': pred_p, 'real_profit': real_p,
            'pf_092': 0, 'ops_092': 0,
        }

    return best_result


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    print(f"\nDispositivo: {DEVICE}{gpu_info}")

    # ── Carga y filtrado ─────────────────────────────────────
    data = pd.read_excel(os.path.join(DATA_DIR, "Data_Entrenamiento_NAS100.xlsx"))
    data.columns = data.columns.str.strip()
    data["profit"] = pd.to_numeric(data["profit"], errors="coerce")
    data = data[data['tipo'].isin(['BUY', 'SELL']) & data["profit"].notna()].copy()

    data['fecha'] = pd.to_datetime(
        data['fecha'].astype(str).str.replace(',', '-', regex=False),
        format="%Y-%m-%d %H:%M", errors='coerce'
    )
    data = data[data['fecha'].notna()].sort_values('fecha').reset_index(drop=True)
    print(f"Operaciones: {len(data):,}  |  {data['fecha'].min().date()} → {data['fecha'].max().date()}")

    # ── Division 70/15/15 temporal ───────────────────────────
    n = len(data)
    t1, t2 = int(n * 0.70), int(n * 0.85)
    train_data = data.iloc[:t1].copy()
    val_data   = data.iloc[t1:t2].copy()
    test_data  = data.iloc[t2:].copy()
    print(f"Division 70/15/15  →  Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")
    print(f"Test periodo: {test_data['fecha'].min().date()} → {test_data['fecha'].max().date()}")

    # ── Features temporales ──────────────────────────────────
    for ds in [train_data, val_data, test_data]:
        ds['dia_semana'] = ds['fecha'].dt.weekday / 6.0
        ds['hora']       = ds['fecha'].dt.hour    / 23.0
        ds['minuto']     = ds['fecha'].dt.minute  / 55.0

    # ── Normalizacion ────────────────────────────────────────
    for ds in [train_data, val_data, test_data]:
        ds['profit_original'] = ds['profit'].fillna(0)

    min_profit_real = train_data['profit_original'].min()
    max_profit_real = train_data['profit_original'].max()
    p1_profit       = train_data['profit_original'].quantile(0.01)
    p99_profit      = train_data['profit_original'].quantile(0.99)

    all_p5        = pd.concat([train_data[c] for c in ['precioopen5','precioclose5','preciohigh5','preciolow5']])
    p1_p5, p99_p5 = all_p5.quantile(0.01), all_p5.quantile(0.99)
    p1_v5, p99_v5 = train_data['volume5'].quantile(0.01), train_data['volume5'].quantile(0.99)

    for ds in [train_data, val_data, test_data]:
        ds['profit'] = normalize_robust(ds['profit_original'], p1_profit, p99_profit)
        for c in ['precioopen5','precioclose5','preciohigh5','preciolow5']:
            ds[c] = normalize_robust(ds[c], p1_p5, p99_p5)
        ds['volume5'] = normalize_robust(ds['volume5'], p1_v5, p99_v5)
        for c in ['rsi5','iStochaMain5','iStochaSign5']:
            ds[c] = ds[c] / 100.0

    extra_cols_5m = [
        'ema550', 'ema5200', 'ema50_prev', 'ema5200_prev',
        'macdLine5', 'signalLine5', 'macdLine_prev5', 'signalLine_prev5',
        'adx5', 'diPlus5', 'diMinus5',
    ]
    scaler_extra = {}
    for c in extra_cols_5m:
        p1, p99 = train_data[c].quantile(0.01), train_data[c].quantile(0.99)
        scaler_extra[f"p1_{c}"] = p1
        scaler_extra[f"p99_{c}"] = p99
        for ds in [train_data, val_data, test_data]:
            ds[c] = normalize_robust(ds[c], p1, p99)

    for ds in [train_data, val_data, test_data]:
        ds.replace([np.inf, -np.inf], np.nan, inplace=True)
        ds.fillna(0, inplace=True)

    # ── Columnas de entrada (solo 5m) ────────────────────────
    input_columns = [
        'dia_semana', 'hora', 'minuto',
        'precioopen5', 'precioclose5', 'preciohigh5', 'preciolow5', 'volume5',
        'rsi5', 'iStochaMain5', 'iStochaSign5',
    ] + extra_cols_5m
    print(f"Features de entrada: {len(input_columns)} (solo 5m)")

    print("\nVerificando rangos en Test:")
    for col in ['precioopen5', 'volume5', 'rsi5', 'adx5']:
        if col in test_data.columns:
            out = ((test_data[col] < -1.1) | (test_data[col] > 1.1)).sum()
            print(f"  {col}: fuera de [-1,1]: {out} ({out/len(test_data)*100:.1f}%)")

    encode = lambda x: 1 if x == "BUY" else 0
    for ds in [train_data, val_data, test_data]:
        ds["tipo_encoded"] = ds["tipo"].apply(encode)

    def make_loader(ds, shuffle):
        X  = ds[input_columns].values
        yp = ds["profit"].values.reshape(-1, 1)
        yt = ds["tipo_encoded"].values
        return DataLoader(TradingDataset(X, yp, yt), batch_size=BATCH_SIZE,
                          shuffle=shuffle, num_workers=NUM_WORKERS), X, yt

    train_loader, X_train, y_tipo_train = make_loader(train_data, True)
    val_loader,   _,       _            = make_loader(val_data,   False)
    test_loader,  _,       _            = make_loader(test_data,  False)

    print(f"\nBalance train — BUY: {np.sum(y_tipo_train==1):,}  SELL: {np.sum(y_tipo_train==0):,}")

    del train_data, val_data, test_data
    import gc; gc.collect()

    model = TradingModelV4(len(input_columns), dropout=0.4).to(DEVICE)
    print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(y_tipo_train), y=y_tipo_train),
        dtype=torch.float32
    ).to(DEVICE)

    criterion_profit = nn.SmoothL1Loss()
    # Peso subido a 0.5: en NAS100 la direccion es tan critica como el profit
    criterion_tipo   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer        = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler        = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping   = EarlyStoppingPF(patience=30)
    best_path        = os.path.join(MODEL_DIR, "best_trading_model_NAS100_v4.pth")

    print(f"\nEntrenando...  (ES guarda por PF a thr={ES_THRESHOLD}, fallback a val_loss)")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'ES':>6} {'Criterio':>10}")
    print("─" * 60)

    for epoch in range(200):
        model.train()
        train_loss = 0
        for X_b, yp_b, yt_b in train_loader:
            X_b, yp_b, yt_b = X_b.to(DEVICE), yp_b.to(DEVICE), yt_b.to(DEVICE)
            optimizer.zero_grad()
            pp, pt = model(X_b)
            # Peso clasificacion 0.5 — direccion critica en NAS100
            loss = criterion_profit(pp, yp_b) + criterion_tipo(pt, yt_b) * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, yp_b, yt_b in val_loader:
                X_b, yp_b, yt_b = X_b.to(DEVICE), yp_b.to(DEVICE), yt_b.to(DEVICE)
                pp, pt = model(X_b)
                val_loss += (criterion_profit(pp, yp_b) + criterion_tipo(pt, yt_b) * 0.5).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # EarlyStopping por PF — pasa el val_loader para calcular PF filtrado
        early_stopping(
            val_loss, model, best_path,
            dataloader=val_loader,
            min_profit=min_profit_real,
            max_profit=max_profit_real,
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr       = optimizer.param_groups[0]['lr']
            criterio = "PF" if early_stopping.using_pf else "loss"
            score_str = f"{early_stopping.best_score:.3f}" if early_stopping.best_score else "—"
            print(f"{epoch+1:>6} {train_loss:>10.5f} {val_loss:>10.5f} {lr:>10.6f} "
                  f"{early_stopping.counter:>4}/{early_stopping.patience}  "
                  f"{criterio}={score_str}")

        if early_stopping.early_stop:
            print(f"\n  → Early stopping en epoca {epoch+1}")
            break

    model.load_state_dict(torch.load(best_path, weights_only=True))

    print("\nEvaluando con Monte Carlo Dropout (thresholds: 0.70 / 0.80 / 0.92)...")
    val_r  = evaluate_model(model, val_loader,  min_profit_real, max_profit_real, "Validacion")
    test_r = evaluate_model(model, test_loader, min_profit_real, max_profit_real, "Test")

    acc_gap = abs(val_r['accuracy'] - test_r['accuracy']) * 100
    mae_gap = abs(val_r['mae'] - test_r['mae'])
    status  = "Generaliza bien" if acc_gap < 10 else ("Brecha moderada" if acc_gap < 20 else "Posible overfitting")
    pf_val  = val_r.get('pf_092', 0)
    pf_test = test_r.get('pf_092', 0)
    print(f"\n── BRECHA VAL vs TEST ──────────────────────────────")
    print(f"  Accuracy gap : {acc_gap:.1f}%  |  MAE gap: ${mae_gap:.0f}  →  {status}")
    print(f"  PF@0.92      : Val={pf_val:.2f}  Test={pf_test:.2f}")
    if pf_test < 1.0:
        print(f"  ATENCION: PF en test < 1.0 — considerar ajustar datos o threshold")
    elif pf_test < 1.5:
        print(f"  AVISO: PF en test < 1.5 — modelo marginal, monitorear en produccion")
    else:
        print(f"  OK: PF en test aceptable para produccion")

    # ── Guardar ──────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "trading_model_NAS100_v4.pth"))
    scaler = {
        "p1_profit":   p1_profit,       "p99_profit":  p99_profit,
        "min_profit":  min_profit_real,  "max_profit":  max_profit_real,
        "p1_precio5":  p1_p5,           "p99_precio5": p99_p5,
        "p1_vol5":     p1_v5,           "p99_vol5":    p99_v5,
    }
    scaler.update(scaler_extra)
    with open(os.path.join(MODEL_DIR, "scaler_NAS100_v4.pkl"),       "wb") as f: pickle.dump(scaler,        f)
    with open(os.path.join(MODEL_DIR, "input_columns_NAS100_v4.pkl"), "wb") as f: pickle.dump(input_columns, f)
    print(f"\nModelo v4.1 NAS100 guardado en {MODEL_DIR}")

if __name__ == '__main__':
    main()