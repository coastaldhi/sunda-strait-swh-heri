r"""
08_multioutput_gru.py
=====================
GRU multi-output: memprediksi SWH, WS10, MWP, MWD (sin/cos) sekaligus untuk t+6/12/24 h,
lalu menghitung predictive HERI skenario S3 (SEMUA input HERI diprediksi) untuk fold 2024.

Arsitektur, fitur, lookback, split, hyperparameter = GRU di 05_rolling_origin.py;
bedanya hanya lapisan output: 5 variabel × 3 horizon = 15 output.

CARA PAKAI (dari folder DL_SelatSunda):
    .\.venv\Scripts\python.exe 08_multioutput_gru.py
    (±10 menit per seed di CPU; 3 seed ≈ 30 menit)
Output di data/multioutput_gru/  -> upload 2 file ini ke Claude:
    mo_variable_skill.csv   (RMSE tiap variabel vs persistence; MWD = galat sudut, derajat)
    mo_heri_summary.csv     (metrik HERI skenario S3, mean ± std antar seed)

CARA MENGUBAH:
    - Seed: SEEDS (default 3 seed supaya cepat; tambah kalau waktu cukup)
    - Fold: TEST_YEAR (default 2024 = fold paper)
    - Parameter HERI identik dengan 07_predictive_heri.py

ASUMSI:
    - Target diskalakan dengan scaler fitur (fit di data train), jadi kelima variabel
      berbobot setara di loss MSE.
    - MWD direkonstruksi dari prediksi sin/cos: MWD = atan2(sin, cos).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ═════════ KONFIGURASI ═════════
CSV_PATH = Path("data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv")
OUT_DIR = Path("data/multioutput_gru")
TEST_YEAR = 2024
SEEDS = [42, 123, 2024]

FEATURE_COLS = ['SWH', 'WS10', 'MWP', 'MWD_sin', 'MWD_cos', 'MSLP', 'U10', 'V10']
TARGETS = ['SWH', 'WS10', 'MWP', 'MWD_sin', 'MWD_cos']
LOOKBACK = 72
HORIZONS = [6, 12, 24]
EPOCHS, BATCH_SIZE, PATIENCE, LR = 100, 64, 10, 0.001

# HERI (identik dengan heri_v2_FINAL.py + override)
SWH_MAX, T_N, BETA, FLOOR, W_H, W_D, THETA = 2.5, 5.5, 0.30, 0.30, 0.70, 0.30, 243.8
CAT_BOUNDS = [0.382, 0.50, 0.727]
CAT_LABELS = ["SAFE", "CAUTION", "DANGEROUS", "VERY DANGEROUS"]
SWH_LIMIT, WS10_LIMIT = 2.5, 21.0 * 0.514444
# ═══════════════════════════════

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def heri(swh, mwp, mwd):
    swh_norm = np.clip(swh / SWH_MAX, 0, 1)
    R = FLOOR + (1 - FLOOR) * np.exp(-(((mwp - T_N) / (BETA * T_N)) ** 2))
    f_dir = np.maximum(0, np.cos(np.radians(mwd - THETA)))
    return np.clip(W_H * swh_norm * R + W_D * f_dir, 0, 1)


def category(swh, mwp, mwd, ws10):
    lvl = np.digitize(heri(swh, mwp, mwd), CAT_BOUNDS)
    return np.where((swh > SWH_LIMIT) | (ws10 > WS10_LIMIT), 3, lvl)


def create_sequences(data, tidx):
    """Input identik 02_preprocessing.py; target = TARGETS pada tiap horizon (h-major)."""
    X, y = [], []
    for i in range(LOOKBACK, len(data) - max(HORIZONS) + 1):
        X.append(data[i - LOOKBACK:i, :])
        y.append([data[i + h - 1, j] for h in HORIZONS for j in tidx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_gru(n_out):
    return Sequential([
        GRU(64, input_shape=(LOOKBACK, len(FEATURE_COLS)), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_out),
    ], name='GRU_multi')


def unscale(arr, sc, tidx):
    """arr (n, H*V) skala [0,1] -> dict var -> (n, H) satuan asli."""
    out = {}
    for v, j in enumerate(tidx):
        cols = arr[:, [hh * len(tidx) + v for hh in range(len(HORIZONS))]]
        out[TARGETS[v]] = cols * sc.data_range_[j] + sc.data_min_[j]
    out['MWD'] = np.degrees(np.arctan2(out['MWD_sin'], out['MWD_cos'])) % 360
    return out


def ang_err(a, b):
    return (a - b + 180) % 360 - 180


def cls_metrics(ref, pred):
    rb, pb = ref >= 2, pred >= 2
    tp = np.sum(rb & pb)
    m = {'accuracy': np.mean(ref == pred),
         'bin_precision': tp / max(pb.sum(), 1), 'bin_recall': tp / max(rb.sum(), 1),
         'skip2': int(np.sum(ref - pred >= 2)), 'over2': int(np.sum(pred - ref >= 2))}
    f1s = []
    for k, lab in enumerate(CAT_LABELS):
        t = np.sum((pred == k) & (ref == k))
        p, r = t / max(np.sum(pred == k), 1), t / max(np.sum(ref == k), 1)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        m[f'rec_{lab}'] = r
    m['macro_F1'] = np.mean(f1s)
    return m


def main():
    print(f"TensorFlow {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    df = pd.read_csv(CSV_PATH, index_col='time', parse_dates=True)
    df = df.loc["2010-01-01 00:00":f"{TEST_YEAR}-12-31 23:00"]
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(limit=6).ffill().bfill()
    raw_mwd = df['MWD']

    train = df.loc[:f"{TEST_YEAR - 2}-12-31 23:00", FEATURE_COLS]
    val = df.loc[f"{TEST_YEAR - 1}-01-01":f"{TEST_YEAR - 1}-12-31 23:00", FEATURE_COLS]
    test = df.loc[f"{TEST_YEAR}-01-01":f"{TEST_YEAR}-12-31 23:00", FEATURE_COLS]
    sc = MinMaxScaler().fit(train)
    tidx = [FEATURE_COLS.index(t) for t in TARGETS]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (create_sequences(sc.transform(p), tidx)
                                          for p in (train, val, test))
    origins = test.index[LOOKBACK - 1: LOOKBACK - 1 + len(Xte)]
    print(f"Sampel train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}")

    # kondisi aktual di jam origin (persistence) dan jam target
    o = df.loc[origins]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    var_rows, heri_rows = [], []

    for seed in SEEDS:
        run = OUT_DIR / f"seed_{seed}"
        run.mkdir(exist_ok=True)
        if (run / 'y_pred.npy').exists():
            print(f"[SKIP] seed {seed}")
            y_pred = np.load(run / 'y_pred.npy')
        else:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_gru(ytr.shape[1])
            model.compile(optimizer=Adam(LR), loss='mse')
            t0 = time.time()
            hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                             batch_size=BATCH_SIZE, verbose=2,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                      restore_best_weights=True),
                                        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                          patience=5, min_lr=1e-6)])
            y_pred = model.predict(Xte, verbose=0)
            np.save(run / 'y_pred.npy', y_pred)
            (run / 'train_info.json').write_text(json.dumps(
                {'train_min': round((time.time() - t0) / 60, 2),
                 'best_epoch': int(np.argmin(hist.history['val_loss']) + 1)}))

        P = unscale(y_pred, sc, tidx)
        for i, h in enumerate(HORIZONS):
            tgt = df.loc[origins + pd.Timedelta(hours=h)]
            act = {'SWH': tgt.SWH.values, 'WS10': tgt.WS10.values,
                   'MWP': tgt.MWP.values, 'MWD': tgt.MWD.values}
            per = {'SWH': o.SWH.values, 'WS10': o.WS10.values,
                   'MWP': o.MWP.values, 'MWD': o.MWD.values}
            for v in ['SWH', 'WS10', 'MWP', 'MWD']:
                e_m = (ang_err(P[v][:, i], act[v]) if v == 'MWD' else P[v][:, i] - act[v])
                e_p = (ang_err(per[v], act[v]) if v == 'MWD' else per[v] - act[v])
                rm, rp = np.sqrt(np.mean(e_m ** 2)), np.sqrt(np.mean(e_p ** 2))
                var_rows.append({'seed': seed, 'horizon': f"t+{h}h", 'variable': v,
                                 'RMSE_model': rm, 'RMSE_persistence': rp,
                                 'skill_pct': 100 * (1 - rm / rp)})

            ref = category(act['SWH'], act['MWP'], act['MWD'], act['WS10'])
            pred = category(P['SWH'][:, i], P['MWP'][:, i], P['MWD'][:, i], P['WS10'][:, i])
            heri_rows.append({'scenario': 'S3_allpredicted', 'seed': seed,
                              'horizon': f"t+{h}h", **cls_metrics(ref, pred)})

    vs = (pd.DataFrame(var_rows).groupby(['variable', 'horizon'])
          [['RMSE_model', 'RMSE_persistence', 'skill_pct']].agg(['mean', 'std']).round(3))
    vs.to_csv(OUT_DIR / 'mo_variable_skill.csv')
    hs = (pd.DataFrame(heri_rows).groupby(['scenario', 'horizon'])
          [['accuracy', 'macro_F1', 'bin_precision', 'bin_recall', 'rec_DANGEROUS',
            'rec_VERY DANGEROUS', 'skip2', 'over2']].agg(['mean', 'std']).round(3))
    hs.to_csv(OUT_DIR / 'mo_heri_summary.csv')
    pd.set_option('display.width', 200)
    print("\n=== Skill per variabel vs persistence ===\n", vs.to_string())
    print("\n=== HERI S3 (semua input diprediksi) ===\n", hs.to_string())
    print(f"\nUpload: {OUT_DIR}\\mo_variable_skill.csv dan mo_heri_summary.csv")


if __name__ == "__main__":
    main()
