"""
============================================================
train_universal.py
Training universal — jalankan dengan versi berbeda

Cara pakai:
  python train_universal.py --version v3
  python train_universal.py --version v4
  python train_universal.py --version v5
  python train_universal.py --all        ← jalankan semua

Output per versi:
  data/models_v3/    data/results_v3/
  data/models_v4/    data/results_v4/
  data/models_v5/    data/results_v5/
============================================================
"""

import numpy as np
import pandas as pd
import json
import pickle
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout,
    Conv1D, MaxPooling1D, Flatten, TimeDistributed, Reshape
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

EPOCHS         = 100
BATCH_SIZE     = 64
PATIENCE       = 10
LR             = 0.001
SEED           = 42
SKIP_IF_EXISTS = True

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

def load_preprocessed(version: str):
    prep_dir = Path(f"data/preprocessed_{version}")
    print(f"\n[1] Load data preprocessed_{version}...")

    X_train = np.load(prep_dir / 'X_train.npy')
    y_train = np.load(prep_dir / 'y_train.npy')
    X_val   = np.load(prep_dir / 'X_val.npy')
    y_val   = np.load(prep_dir / 'y_val.npy')
    X_test  = np.load(prep_dir / 'X_test.npy')
    y_test  = np.load(prep_dir / 'y_test.npy')

    with open(prep_dir / 'scaler_target.pkl', 'rb') as f:
        scaler_target = pickle.load(f)
    with open(prep_dir / 'preprocessing_config.json') as f:
        config = json.load(f)

    print(f"  Versi    : {config['desc']}")
    print(f"  X_train  : {X_train.shape}")
    print(f"  X_val    : {X_val.shape}")
    print(f"  X_test   : {X_test.shape}")

    test_timestamps = pd.to_datetime(
        pd.read_csv(prep_dir / 'test_timestamps.csv').iloc[:, 0]
    ).values

    return (X_train, y_train, X_val, y_val,
            X_test, y_test, scaler_target,
            config, test_timestamps)

# ============================================================
# DEFINISI MODEL
# ============================================================

def build_lstm(n_features, lookback, n_horizons):
    return Sequential([
        LSTM(64, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons)
    ], name='LSTM')


def build_gru(n_features, lookback, n_horizons):
    return Sequential([
        GRU(64, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons)
    ], name='GRU')


def build_bilstm(n_features, lookback, n_horizons):
    return Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(lookback, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons)
    ], name='BiLSTM')


def build_cnn_lstm(n_features, lookback, n_horizons):
    # Sesuaikan n_subseq agar habis dibagi lookback
    # lookback 72  → 6 subseq × 12 steps
    # lookback 120 → 6 subseq × 20 steps
    n_subseq = 6
    n_steps  = lookback // n_subseq
    return Sequential([
        Reshape((n_subseq, n_steps, n_features),
                input_shape=(lookback, n_features)),
        TimeDistributed(Conv1D(64, kernel_size=3,
                               activation='relu', padding='same')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_horizons)
    ], name='CNN_LSTM')

# ============================================================
# TRAINING DL
# ============================================================

def compile_and_train(model, X_train, y_train, X_val, y_val,
                      model_name, model_dir):
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(model_dir / f'{model_name}_best.h5'),
                        monitor='val_loss', save_best_only=True, verbose=0)
    ]

    print(f"\n  Training {model_name}...")
    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )
    t_elapsed  = time.time() - t_start
    best_epoch = np.argmin(history.history['val_loss']) + 1

    print(f"\n  {model_name} selesai!")
    print(f"  Waktu     : {t_elapsed/60:.1f} menit")
    print(f"  Best epoch: {best_epoch}/{len(history.history['loss'])}")
    print(f"  Val loss  : {min(history.history['val_loss']):.6f}")

    model.save(str(model_dir / f'{model_name}_final.h5'))
    return history, t_elapsed

# ============================================================
# XGBOOST
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val,
                  horizons, model_dir):
    print(f"\n  Training XGBoost...")
    t_start    = time.time()
    xgb_models = {}

    X_tr = X_train.reshape(X_train.shape[0], -1)
    X_vl = X_val.reshape(X_val.shape[0], -1)

    for i, h in enumerate(horizons):
        h_name     = f't+{h}h'
        model_path = model_dir / f'XGBoost_{h_name}.pkl'

        if SKIP_IF_EXISTS and model_path.exists():
            print(f"    [SKIP] XGBoost {h_name}")
            with open(model_path, 'rb') as f:
                xgb_models[h_name] = pickle.load(f)
            continue

        print(f"    {h_name}...", end=" ", flush=True)
        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, eval_metric='rmse',
            random_state=SEED, n_jobs=-1, verbosity=0
        )
        model.fit(X_tr, y_train[:, i],
                  eval_set=[(X_vl, y_val[:, i])],
                  verbose=False)
        xgb_models[h_name] = model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"OK (best iter={model.best_iteration})")

    t_elapsed = time.time() - t_start
    print(f"  XGBoost selesai! {t_elapsed/60:.1f} menit")
    return xgb_models, t_elapsed

# ============================================================
# EVALUASI
# ============================================================

def evaluate_model(model, X_test, y_test, scaler_target,
                   model_name, model_dir, horizons,
                   is_xgboost=False):
    if is_xgboost:
        X_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred_norm = np.column_stack([
            model[f't+{h}h'].predict(X_flat)
            for h in horizons
        ])
    else:
        y_pred_norm = model.predict(X_test, verbose=0)

    results = {}
    for i, h in enumerate(horizons):
        y_true = scaler_target.inverse_transform(
            y_test[:, i].reshape(-1, 1)).flatten()
        y_pred = scaler_target.inverse_transform(
            y_pred_norm[:, i].reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mask = y_true > 0.01
        mape = np.mean(np.abs(
            (y_true[mask] - y_pred[mask]) / y_true[mask]
        )) * 100

        results[f't+{h}h'] = {
            'RMSE': round(float(rmse), 4),
            'MAE' : round(float(mae),  4),
            'MAPE': round(float(mape), 2),
            'R2'  : round(float(r2),   4),
        }
        print(f"  t+{h:2d}h | RMSE={rmse:.4f}m | MAE={mae:.4f}m | "
              f"MAPE={mape:.2f}% | R²={r2:.4f}")

    np.save(model_dir / f'{model_name}_y_pred.npy', y_pred_norm)
    return results


def evaluate_persistence(X_test, y_test, scaler_target,
                         model_dir, horizons):
    swh_last = scaler_target.inverse_transform(
        X_test[:, -1, 0].reshape(-1, 1)).flatten()
    y_pers = np.column_stack([swh_last] * len(horizons))

    results = {}
    for i, h in enumerate(horizons):
        y_true = scaler_target.inverse_transform(
            y_test[:, i].reshape(-1, 1)).flatten()
        rmse = np.sqrt(mean_squared_error(y_true, y_pers[:, i]))
        mae  = mean_absolute_error(y_true, y_pers[:, i])
        r2   = r2_score(y_true, y_pers[:, i])
        mask = y_true > 0.01
        mape = np.mean(np.abs(
            (y_true[mask] - y_pers[:, i][mask]) / y_true[mask]
        )) * 100
        results[f't+{h}h'] = {
            'RMSE': round(float(rmse), 4), 'MAE': round(float(mae), 4),
            'MAPE': round(float(mape), 2), 'R2' : round(float(r2),  4),
        }
        print(f"  t+{h}h | RMSE={rmse:.4f}m | R²={r2:.4f}")

    pers_norm = np.column_stack([
        scaler_target.transform(y_pers[:, i].reshape(-1, 1))
        for i in range(len(horizons))
    ])
    np.save(model_dir / 'Persistence_y_pred.npy', pers_norm)
    return results

# ============================================================
# VISUALISASI & TABEL
# ============================================================

def save_comparison_table(all_results, results_dir, version):
    rows = []
    for model_name, horizons in all_results.items():
        for horizon, metrics in horizons.items():
            if not horizon.startswith('t+'):
                continue
            rows.append({
                'Model'  : model_name, 'Horizon': horizon,
                'RMSE'   : metrics['RMSE'], 'MAE': metrics['MAE'],
                'MAPE'   : metrics['MAPE'], 'R2' : metrics['R2'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / 'comparison_table.csv', index=False)

    print(f"\n{'='*70}")
    print(f"TABEL PERBANDINGAN — {version.upper()}")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    horizons_list = df['Horizon'].unique()
    print(f"\n  ★ Model terbaik per horizon:")
    for h in horizons_list:
        sub  = df[df['Horizon'] == h]
        best = sub.loc[sub['RMSE'].idxmin()]
        print(f"  {h} → {best['Model']} "
              f"(RMSE={best['RMSE']:.4f}, R²={best['R2']:.4f})")
    return df


def plot_summary(all_results, results_dir, version, config):
    """Bar chart RMSE semua model — ringkasan cepat."""
    rows = []
    for name, horizons in all_results.items():
        for h_key, metrics in horizons.items():
            if h_key.startswith('t+'):
                rows.append({'Model': name, 'Horizon': h_key,
                             'RMSE': metrics['RMSE'], 'R2': metrics['R2']})
    df = pd.DataFrame(rows)

    model_names = df['Model'].unique()
    horizon_keys = df['Horizon'].unique()
    x = np.arange(len(horizon_keys))
    width = 0.15

    colors = {'LSTM': '#3498db', 'GRU': '#2ecc71', 'BiLSTM': '#9b59b6',
              'CNN_LSTM': '#e74c3c', 'XGBoost': '#f39c12',
              'Persistence': '#95a5a6'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Perbandingan Model — {version.upper()}: {config["desc"]}',
                 fontsize=12, fontweight='bold')

    for ax, metric in zip(axes, ['RMSE', 'R2']):
        for j, name in enumerate(model_names):
            sub  = df[df['Model'] == name]
            vals = [sub[sub['Horizon'] == h][metric].values[0]
                    for h in horizon_keys]
            offset = (j - len(model_names)/2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=name, color=colors.get(name, 'gray'), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(horizon_keys)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=7)
        if metric == 'R2':
            ax.set_ylim(0.6, 1.02)

    plt.tight_layout()
    path = results_dir / f'summary_{version}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path.name}")

# ============================================================
# MAIN PER VERSI
# ============================================================

def run_version(version: str):
    print(f"\n{'#'*60}")
    print(f"# EKSPERIMEN {version.upper()}")
    print(f"{'#'*60}")

    model_dir   = Path(f"data/models_{version}")
    results_dir = Path(f"data/results_{version}")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load
    (X_train, y_train, X_val, y_val,
     X_test, y_test, scaler_target,
     config, test_timestamps) = load_preprocessed(version)

    lookback   = config['lookback']
    horizons   = config['horizons']
    n_features = config['n_features']
    n_horizons = config['n_horizons']

    dl_builders = {
        'LSTM'    : build_lstm,
        'GRU'     : build_gru,
        'BiLSTM'  : build_bilstm,
        'CNN_LSTM': build_cnn_lstm,
    }

    all_results   = {}
    all_histories = {}

    # ---- DL Models ----
    print(f"\n{'='*60}")
    print("DEEP LEARNING MODELS")
    print(f"{'='*60}")

    for model_name, builder in dl_builders.items():
        print(f"\n--- {model_name} ---")
        final_path = model_dir / f'{model_name}_final.h5'
        pred_path  = model_dir / f'{model_name}_y_pred.npy'

        if SKIP_IF_EXISTS and final_path.exists() and pred_path.exists():
            print(f"  [SKIP] Load dari file...")
            model = load_model(str(final_path), compile=False)
            model.compile(optimizer=Adam(LR), loss='mse', metrics=['mae'])
            class FakeHistory:
                history = {'loss':[],'val_loss':[],'mae':[],'val_mae':[]}
            all_histories[model_name] = FakeHistory()
        else:
            model = builder(n_features, lookback, n_horizons)
            history, t_elapsed = compile_and_train(
                model, X_train, y_train, X_val, y_val,
                model_name, model_dir
            )
            all_histories[model_name] = history
            all_results[model_name] = {
                'training_time_min': round(t_elapsed/60, 2)
            }

        print(f"  Evaluasi {model_name}...")
        results = evaluate_model(model, X_test, y_test, scaler_target,
                                 model_name, model_dir, horizons)
        all_results[model_name] = {
            **all_results.get(model_name, {}), **results
        }

    # ---- XGBoost ----
    print(f"\n{'='*60}")
    print("XGBOOST")
    print(f"{'='*60}")
    xgb_models, xgb_time = train_xgboost(
        X_train, y_train, X_val, y_val, horizons, model_dir
    )
    print(f"\n  Evaluasi XGBoost...")
    xgb_results = evaluate_model(
        xgb_models, X_test, y_test, scaler_target,
        'XGBoost', model_dir, horizons, is_xgboost=True
    )
    all_results['XGBoost'] = {
        **xgb_results, 'training_time_min': round(xgb_time/60, 2)
    }

    # ---- Persistence ----
    print(f"\n{'='*60}")
    print("Persistence (Baseline)")
    print(f"{'='*60}")
    all_results['Persistence'] = evaluate_persistence(
        X_test, y_test, scaler_target, model_dir, horizons
    )

    # ---- Output ----
    save_comparison_table(all_results, results_dir, version)
    plot_summary(all_results, results_dir, version, config)

    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))

    print(f"\n{'='*60}")
    print(f"✓ {version.upper()} SELESAI!")
    print(f"  Models  : {model_dir.resolve()}")
    print(f"  Results : {results_dir.resolve()}")
    print(f"{'='*60}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str,
                        choices=['v3', 'v4', 'v5'],
                        help='Versi eksperimen')
    parser.add_argument('--all', action='store_true',
                        help='Jalankan semua versi (v3, v4, v5)')
    args = parser.parse_args()

    if args.all:
        for v in ['v3', 'v4', 'v5']:
            run_version(v)
        print("\n✓ SEMUA EKSPERIMEN SELESAI!")
    elif args.version:
        run_version(args.version)
    else:
        print("Gunakan --version v3/v4/v5 atau --all")
        print("Contoh: python train_universal.py --version v3")
