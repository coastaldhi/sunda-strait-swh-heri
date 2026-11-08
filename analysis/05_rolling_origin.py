"""
05_rolling_origin.py
====================
Rolling-origin evaluation: CNN-LSTM, BiLSTM, XGBoost, LSTM, GRU (+ persistence baseline).
Arsitektur, fitur, lookback, horizon, dan hyperparameter IDENTIK dengan
02_preprocessing.py + 03_train_modelsv2_GPU.py (konfigurasi v2).

FOLD (default):
    test 2021 : train 2010–2019 | val 2020 | test 2021
    test 2022 : train 2010–2020 | val 2021 | test 2022
    test 2023 : train 2010–2021 | val 2022 | test 2023
    test 2024 : train 2010–2022 | val 2023 | test 2024   (= konfigurasi paper)
Setiap fold: scaler di-fit ULANG hanya dari data train fold itu (tanpa leakage).

CARA PAKAI (PowerShell / PyCharm, dari folder DL_SelatSunda):
    python 05_rolling_origin.py
Output:
    data/rolling_origin/rolling_origin_results.csv   (fold × model × seed × horizon)
    data/rolling_origin/rolling_origin_summary.csv   (rata-rata ± std, semua fold × seed)
    -> dua file ini yang di-upload ke Claude.

CARA MENGUBAH:
    - Model yang dijalankan: MODELS = ['CNN_LSTM', 'BiLSTM', 'XGBoost', 'LSTM', 'GRU']
      (model yang sudah selesai di-skip; hapus dari list kalau tidak perlu)
    - Seed: SEEDS = [42, 123, 2024, 7, 99]
    - Run yang sudah selesai otomatis di-skip (SKIP_IF_EXISTS = True). Hasil CNN-LSTM seed 42
      dari 05_rolling_origin_cnnlstm.py ikut terpakai (folder fold_<tahun>/seed_42).
    - Ganti fold: TEST_YEARS.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Conv1D, Dense, Dropout,
                                     Flatten, MaxPooling1D, Reshape, TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor

# ═════════ KONFIGURASI ═════════
CSV_PATH = Path("data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv")
OUT_DIR = Path("data/rolling_origin")
DATA_START = "2010-01-01 00:00"
DATA_END = "2024-12-31 23:00"

TEST_YEARS = [2021, 2022, 2023, 2024]
MODELS = ['CNN_LSTM', 'BiLSTM', 'XGBoost', 'LSTM', 'GRU']
SEEDS = [42, 123, 2024, 7, 99]

FEATURE_COLS = ['SWH', 'WS10', 'MWP', 'MWD_sin', 'MWD_cos', 'MSLP', 'U10', 'V10']
TARGET_COL = 'SWH'
LOOKBACK = 72
HORIZONS = [6, 12, 24]

EPOCHS, BATCH_SIZE, PATIENCE, LR = 100, 64, 10, 0.001
SKIP_IF_EXISTS = True
# ═══════════════════════════════

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def create_sequences(data, lookback, horizons, target_idx):
    """Identik dengan 02_preprocessing.py."""
    max_h = max(horizons)
    X, y = [], []
    for i in range(lookback, len(data) - max_h + 1):
        X.append(data[i - lookback:i, :])
        y.append([data[i + h - 1, target_idx] for h in horizons])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_cnn_lstm(n_features, lookback, n_horizons):
    """Identik dengan 03_train_modelsv2_GPU.py."""
    n_subseq = 6
    n_steps = lookback // n_subseq
    return Sequential([
        Reshape((n_subseq, n_steps, n_features), input_shape=(lookback, n_features)),
        TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', padding='same')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_horizons),
    ], name='CNN_LSTM')


def build_bilstm(n_features, lookback, n_horizons):
    """Identik dengan 03_train_modelsv2_GPU.py."""
    return Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons),
    ], name='BiLSTM')


def build_lstm(n_features, lookback, n_horizons):
    """Identik dengan 03_train_modelsv2_GPU.py."""
    return Sequential([
        LSTM(64, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons),
    ], name='LSTM')


def build_gru(n_features, lookback, n_horizons):
    """Identik dengan 03_train_modelsv2_GPU.py."""
    return Sequential([
        GRU(64, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons),
    ], name='GRU')


BUILDERS = {'CNN_LSTM': build_cnn_lstm, 'BiLSTM': build_bilstm,
            'LSTM': build_lstm, 'GRU': build_gru}


def make_fold(df, test_year):
    train = df.loc[:f"{test_year - 2}-12-31 23:00"]
    val = df.loc[f"{test_year - 1}-01-01 00:00":f"{test_year - 1}-12-31 23:00"]
    test = df.loc[f"{test_year}-01-01 00:00":f"{test_year}-12-31 23:00"]

    sc_feat = MinMaxScaler().fit(train[FEATURE_COLS])
    sc_tgt = MinMaxScaler().fit(train[[TARGET_COL]])
    tidx = FEATURE_COLS.index(TARGET_COL)

    out = {}
    for name, part in [('train', train), ('val', val), ('test', test)]:
        out[name] = create_sequences(sc_feat.transform(part[FEATURE_COLS]),
                                     LOOKBACK, HORIZONS, tidx)
    origins = test.index[LOOKBACK - 1: LOOKBACK - 1 + len(out['test'][0])]
    return out, sc_tgt, origins, (len(train), len(val), len(test))


def metrics(y_true, y_pred):
    mask = y_true > 0.01
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'Bias': float(np.mean(y_pred - y_true)),
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE': float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
    }


def run_dl(name, seed, Xtr, ytr, Xva, yva, Xte, sc_tgt, run_dir):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = BUILDERS[name](len(FEATURE_COLS), LOOKBACK, len(HORIZONS))
    model.compile(optimizer=Adam(LR), loss='mse', metrics=['mae'])
    t0 = time.time()
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                 restore_best_weights=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)],
        verbose=2)
    info = {'train_min': round((time.time() - t0) / 60, 2),
            'best_epoch': int(np.argmin(hist.history['val_loss']) + 1),
            'n_epochs': len(hist.history['loss'])}
    model.save(run_dir / f'{name}.h5')
    return sc_tgt.inverse_transform(model.predict(Xte, verbose=0)), info


def run_xgb(seed, Xtr, ytr, Xva, yva, Xte, sc_tgt):
    """Identik dengan train_xgboost() di 03_train_modelsv2_GPU.py (satu model per horizon)."""
    t0 = time.time()
    Xtr_f, Xva_f, Xte_f = (a.reshape(len(a), -1) for a in (Xtr, Xva, Xte))
    pred, best = np.zeros((len(Xte), len(HORIZONS))), []
    for i in range(len(HORIZONS)):
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=20, eval_metric='rmse',
                         random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(Xtr_f, ytr[:, i], eval_set=[(Xva_f, yva[:, i])], verbose=False)
        pred[:, i] = m.predict(Xte_f)
        best.append(int(m.best_iteration))
    info = {'train_min': round((time.time() - t0) / 60, 2), 'best_epoch': str(best),
            'n_epochs': None}
    return sc_tgt.inverse_transform(pred), info


def run_dir_for(ty, name, seed):
    # CNN-LSTM pakai folder lama (fold_<th>/seed_<s>) supaya hasil run sebelumnya terpakai
    sub = f"seed_{seed}" if name == 'CNN_LSTM' else f"{name}_seed_{seed}"
    return OUT_DIR / f"fold_{ty}" / sub


def main():
    print(f"TensorFlow {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    df = pd.read_csv(CSV_PATH, index_col='time', parse_dates=True)
    df = df.loc[DATA_START:DATA_END, FEATURE_COLS]
    df = df.interpolate(limit=6).ffill().bfill()   # sama dengan handle_missing()
    print(f"Data: {len(df):,} jam ({df.index.min()} – {df.index.max()})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for ty in TEST_YEARS:
        data, sc_tgt, origins, n_hours = make_fold(df, ty)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = data['train'], data['val'], data['test']
        y_true = sc_tgt.inverse_transform(yte)
        print(f"\n=== Fold test {ty} | jam train/val/test = {n_hours} | "
              f"sampel = {len(Xtr)}/{len(Xva)}/{len(Xte)} ===")

        y_pers = np.repeat(sc_tgt.inverse_transform(Xte[:, -1, [0]]), len(HORIZONS), axis=1)
        for i, h in enumerate(HORIZONS):
            rows.append({'test_year': ty, 'model': 'Persistence', 'seed': None,
                         'horizon': f"t+{h}h", 'n_test': len(Xte),
                         **metrics(y_true[:, i], y_pers[:, i])})

        for name in MODELS:
            for seed in SEEDS:
                run_dir = run_dir_for(ty, name, seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                pred_path = run_dir / 'y_pred.npy'

                if SKIP_IF_EXISTS and pred_path.exists():
                    print(f"  [SKIP] {name} seed {seed}")
                    y_pred = np.load(pred_path)
                    info = json.loads((run_dir / 'train_info.json').read_text())
                else:
                    print(f"\n  >> {name} seed {seed}")
                    if name == 'XGBoost':
                        y_pred, info = run_xgb(seed, Xtr, ytr, Xva, yva, Xte, sc_tgt)
                    else:
                        y_pred, info = run_dl(name, seed, Xtr, ytr, Xva, yva, Xte,
                                              sc_tgt, run_dir)
                    np.save(pred_path, y_pred)
                    np.save(run_dir / 'y_true.npy', y_true)
                    pd.Series(origins, name='origin_time').to_csv(
                        run_dir / 'test_timestamps.csv', index=False)
                    (run_dir / 'train_info.json').write_text(json.dumps(info))

                for i, h in enumerate(HORIZONS):
                    m = metrics(y_true[:, i], y_pred[:, i])
                    rows.append({'test_year': ty, 'model': name, 'seed': seed,
                                 'horizon': f"t+{h}h", 'n_test': len(Xte), **m, **info})
                    print(f"  {name} seed {seed} t+{h:2d}h | RMSE={m['RMSE']:.4f} "
                          f"Bias={m['Bias']:+.4f} R2={m['R2']:.4f}")

                # simpan setiap selesai satu run (aman kalau proses terputus)
                pd.DataFrame(rows).to_csv(OUT_DIR / 'rolling_origin_results.csv', index=False)

    res = pd.DataFrame(rows)
    summary = (res.groupby(['model', 'horizon'])[['RMSE', 'MAE', 'Bias', 'R2']]
               .agg(['mean', 'std']).round(4))
    summary.to_csv(OUT_DIR / 'rolling_origin_summary.csv')
    print("\n=== RINGKASAN (mean ± std atas semua fold × seed) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
