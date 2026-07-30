"""
============================================================
03_train_models.py
Training & Evaluasi Model Deep Learning + XGBoost
LSTM | GRU | CNN-LSTM | BiLSTM | XGBoost
============================================================
"""

import numpy as np
import pandas as pd
import json
import pickle
import time
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

# ============================================================
# KONFIGURASI
# ============================================================

PREP_DIR    = Path("data/preprocessed")
MODEL_DIR   = Path("data/models")
RESULTS_DIR = Path("data/results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(PREP_DIR / 'preprocessing_config.json') as f:
    CONFIG = json.load(f)

LOOKBACK   = CONFIG['lookback']     # 72
HORIZONS   = CONFIG['horizons']     # [6, 12, 24]
N_FEATURES = CONFIG['n_features']   # 8
N_HORIZONS = CONFIG['n_horizons']   # 3

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

def load_preprocessed():
    print("\n[1] Load data preprocessed...")
    X_train = np.load(PREP_DIR / 'X_train.npy')
    y_train = np.load(PREP_DIR / 'y_train.npy')
    X_val   = np.load(PREP_DIR / 'X_val.npy')
    y_val   = np.load(PREP_DIR / 'y_val.npy')
    X_test  = np.load(PREP_DIR / 'X_test.npy')
    y_test  = np.load(PREP_DIR / 'y_test.npy')
    with open(PREP_DIR / 'scaler_target.pkl', 'rb') as f:
        scaler_target = pickle.load(f)
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_target

# ============================================================
# DEFINISI MODEL DEEP LEARNING
# ============================================================

def build_lstm(n_features, lookback, n_horizons):
    """LSTM 2-layer baseline."""
    model = Sequential([
        LSTM(64, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons)
    ], name='LSTM')
    return model


def build_gru(n_features, lookback, n_horizons):
    """GRU 2-layer."""
    model = Sequential([
        GRU(64, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons)
    ], name='GRU')
    return model


def build_bilstm(n_features, lookback, n_horizons):
    """
    Bidirectional LSTM — baca sekuens dari dua arah.
    Biasanya outperform LSTM biasa untuk data oseanografi.
    """
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(lookback, n_features)
        ),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_horizons)
    ], name='BiLSTM')
    return model


def build_cnn_lstm(n_features, lookback, n_horizons):
    """CNN-LSTM hybrid."""
    n_subseq = 6
    n_steps  = lookback // n_subseq   # 12 timesteps per subsequence
    model = Sequential([
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
    return model

# ============================================================
# XGBOOST — perlu reshape data karena tidak pakai sequences 3D
# ============================================================

def prepare_xgboost_data(X, y):
    """
    XGBoost tidak pakai format 3D (samples, timesteps, features).
    Flatten jadi 2D: (samples, timesteps × features)
    """
    n_samples  = X.shape[0]
    X_flat     = X.reshape(n_samples, -1)   # (samples, 72×8 = 576)
    return X_flat, y


def train_xgboost(X_train, y_train, X_val, y_val, horizon_idx: int,
                  horizon_name: str):
    """
    Train satu XGBoost model untuk satu horizon prediksi.
    XGBoost tidak support multi-output langsung, jadi
    kita train satu model per horizon.
    """
    X_tr_flat, _ = prepare_xgboost_data(X_train, y_train)
    X_vl_flat, _ = prepare_xgboost_data(X_val,   y_val)

    y_tr = y_train[:, horizon_idx]
    y_vl = y_val[:, horizon_idx]

    model = XGBRegressor(
        n_estimators    = 500,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        early_stopping_rounds = 20,
        eval_metric     = 'rmse',
        random_state    = SEED,
        n_jobs          = -1,       # pakai semua CPU core
        verbosity       = 0,
    )

    model.fit(
        X_tr_flat, y_tr,
        eval_set=[(X_vl_flat, y_vl)],
        verbose=False
    )

    best_iter = model.best_iteration
    print(f"    {horizon_name}: best iteration = {best_iter}")
    return model


def train_all_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost untuk semua horizon dan simpan.
    Return: dict {horizon_name: model}
    """
    print(f"\n  Training XGBoost (satu model per horizon)...")
    t_start = time.time()

    xgb_models = {}
    for i, h in enumerate(HORIZONS):
        h_name = f't+{h}h'
        model_path = MODEL_DIR / f'XGBoost_{h_name}.pkl'

        if SKIP_IF_EXISTS and model_path.exists():
            print(f"    [SKIP] XGBoost {h_name} sudah ada")
            with open(model_path, 'rb') as f:
                xgb_models[h_name] = pickle.load(f)
            continue

        print(f"    Training XGBoost {h_name}...", end=" ", flush=True)
        model = train_xgboost(X_train, y_train, X_val, y_val, i, h_name)
        xgb_models[h_name] = model

        # Simpan
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"OK")

    t_elapsed = time.time() - t_start
    print(f"  XGBoost selesai! Waktu: {t_elapsed/60:.1f} menit")
    return xgb_models, t_elapsed


def predict_xgboost(xgb_models, X_test):
    """Prediksi XGBoost untuk semua horizon, return array (samples, n_horizons)."""
    X_flat = X_test.reshape(X_test.shape[0], -1)
    preds  = np.zeros((X_test.shape[0], len(HORIZONS)))
    for i, h in enumerate(HORIZONS):
        h_name = f't+{h}h'
        preds[:, i] = xgb_models[h_name].predict(X_flat)
    return preds

# ============================================================
# TRAINING DL MODELS
# ============================================================

def compile_and_train(model, X_train, y_train, X_val, y_val, model_name):
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(MODEL_DIR / f'{model_name}_best.h5'),
                        monitor='val_loss', save_best_only=True, verbose=0)
    ]

    print(f"\n  Training {model_name}...")
    t_start = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    t_elapsed  = time.time() - t_start
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\n  {model_name} selesai!")
    print(f"  Waktu    : {t_elapsed/60:.1f} menit")
    print(f"  Best epoch: {best_epoch}/{len(history.history['loss'])}")
    print(f"  Val loss  : {min(history.history['val_loss']):.6f}")

    model.save(str(MODEL_DIR / f'{model_name}_final.h5'))
    return history, t_elapsed

# ============================================================
# EVALUASI
# ============================================================

def evaluate_model(model, X_test, y_test, scaler_target, model_name,
                   is_xgboost=False):
    """Evaluasi model DL atau XGBoost."""
    if is_xgboost:
        y_pred_norm = predict_xgboost(model, X_test)
    else:
        y_pred_norm = model.predict(X_test, verbose=0)

    results = {}
    for i, h in enumerate(HORIZONS):
        y_true_real = scaler_target.inverse_transform(
            y_test[:, i].reshape(-1, 1)).flatten()
        y_pred_real = scaler_target.inverse_transform(
            y_pred_norm[:, i].reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        mae  = mean_absolute_error(y_true_real, y_pred_real)
        r2   = r2_score(y_true_real, y_pred_real)
        mask = y_true_real > 0.01
        mape = np.mean(np.abs(
            (y_true_real[mask] - y_pred_real[mask]) / y_true_real[mask]
        )) * 100

        results[f't+{h}h'] = {
            'RMSE': round(float(rmse), 4),
            'MAE' : round(float(mae),  4),
            'MAPE': round(float(mape), 2),
            'R2'  : round(float(r2),   4),
        }
        print(f"  t+{h:2d}h | RMSE={rmse:.4f}m | MAE={mae:.4f}m | "
              f"MAPE={mape:.2f}% | R²={r2:.4f}")

    # Simpan prediksi
    np.save(MODEL_DIR / f'{model_name}_y_pred.npy', y_pred_norm)
    return results


def evaluate_persistence(X_test, y_test, scaler_target):
    swh_last = X_test[:, -1, 0]
    swh_last_real = scaler_target.inverse_transform(
        swh_last.reshape(-1, 1)).flatten()
    y_persistence = np.column_stack([swh_last_real] * len(HORIZONS))

    results = {}
    for i, h in enumerate(HORIZONS):
        y_true_real = scaler_target.inverse_transform(
            y_test[:, i].reshape(-1, 1)).flatten()
        y_pred_real = y_persistence[:, i]
        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        mae  = mean_absolute_error(y_true_real, y_pred_real)
        r2   = r2_score(y_true_real, y_pred_real)
        mask = y_true_real > 0.01
        mape = np.mean(np.abs(
            (y_true_real[mask] - y_pred_real[mask]) / y_true_real[mask]
        )) * 100
        results[f't+{h}h'] = {
            'RMSE': round(float(rmse), 4),
            'MAE' : round(float(mae),  4),
            'MAPE': round(float(mape), 2),
            'R2'  : round(float(r2),   4),
        }
    np.save(MODEL_DIR / 'Persistence_y_pred.npy',
            np.column_stack([
                scaler_target.transform(y_persistence[:, i].reshape(-1, 1))
                for i in range(len(HORIZONS))
            ]))
    return results

# ============================================================
# VISUALISASI
# ============================================================

def plot_training_history(histories):
    # Filter hanya model DL (yang punya history)
    dl_histories = {k: v for k, v in histories.items()
                    if v.history.get('loss')}
    if not dl_histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — Loss Curve (DL Models)',
                 fontsize=13, fontweight='bold')

    colors = {'LSTM'    : '#3498db', 'GRU'     : '#2ecc71',
              'BiLSTM'  : '#9b59b6', 'CNN_LSTM': '#e74c3c'}

    for name, history in dl_histories.items():
        if not history.history['loss']:
            continue
        color = colors.get(name, 'gray')
        axes[0].plot(history.history['loss'], color=color,
                     lw=1.5, label=f'{name} train')
        axes[0].plot(history.history['val_loss'], color=color,
                     lw=1.5, ls='--', label=f'{name} val', alpha=0.7)
        axes[1].plot(history.history['mae'], color=color,
                     lw=1.5, label=f'{name} train')
        axes[1].plot(history.history['val_mae'], color=color,
                     lw=1.5, ls='--', label=f'{name} val', alpha=0.7)

    for ax, title, ylabel in zip(axes, ['Loss (MSE)', 'MAE'], ['MSE', 'MAE']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    axes[0].set_yscale('log')

    plt.tight_layout()
    path = RESULTS_DIR / 'training_history.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path.name}")


def plot_predictions(model_names_with_pred, X_test, y_test,
                     scaler_target, test_timestamps):
    n_plot = 24 * 30
    colors = {'LSTM': '#3498db', 'GRU': '#2ecc71', 'BiLSTM': '#9b59b6',
              'CNN_LSTM': '#e74c3c', 'XGBoost': '#f39c12',
              'Persistence': '#95a5a6'}

    fig, axes = plt.subplots(len(HORIZONS), 1,
                              figsize=(16, 4 * len(HORIZONS)), sharex=True)
    fig.suptitle('Prediksi vs Aktual — 30 Hari Pertama Testing',
                 fontsize=13, fontweight='bold')

    for idx, h in enumerate(HORIZONS):
        ax = axes[idx]
        y_true = scaler_target.inverse_transform(
            y_test[:n_plot, idx].reshape(-1, 1)).flatten()
        times = test_timestamps[:n_plot]
        ax.plot(times, y_true, 'k-', lw=1.5, label='Aktual', zorder=5)

        for name in model_names_with_pred:
            pred_path = MODEL_DIR / f'{name}_y_pred.npy'
            if not pred_path.exists():
                continue
            y_pred_norm = np.load(pred_path)
            y_pred = scaler_target.inverse_transform(
                y_pred_norm[:n_plot, idx].reshape(-1, 1)).flatten()
            ax.plot(times, y_pred,
                    color=colors.get(name, 'gray'),
                    lw=1, alpha=0.8, label=name)

        ax.axhline(2.5, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylabel('SWH (m)')
        ax.set_title(f'Horizon t+{h}h')
        ax.legend(fontsize=7, loc='upper right', ncol=3)

    plt.tight_layout()
    path = RESULTS_DIR / 'predictions_vs_actual.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path.name}")


def plot_scatter_all(model_names_with_pred, X_test, y_test, scaler_target):
    names = [n for n in model_names_with_pred
             if (MODEL_DIR / f'{n}_y_pred.npy').exists()]
    n_models = len(names)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle('Scatter Plot: Prediksi vs Aktual (t+24h)',
                 fontsize=13, fontweight='bold')

    horizon_idx = HORIZONS.index(24)
    y_true = scaler_target.inverse_transform(
        y_test[:, horizon_idx].reshape(-1, 1)).flatten()

    for ax, name in zip(axes, names):
        y_pred_norm = np.load(MODEL_DIR / f'{name}_y_pred.npy')
        y_pred = scaler_target.inverse_transform(
            y_pred_norm[:, horizon_idx].reshape(-1, 1)).flatten()
        ax.scatter(y_true, y_pred, alpha=0.1, s=2, c='steelblue')
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=1)
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f'{name}\nR²={r2:.4f}')
        ax.set_xlabel('Aktual SWH (m)')
        ax.set_ylabel('Prediksi SWH (m)')

    plt.tight_layout()
    path = RESULTS_DIR / 'scatter_t24h.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path.name}")


def save_comparison_table(all_results):
    rows = []
    for model_name, horizons in all_results.items():
        for horizon, metrics in horizons.items():
            if not horizon.startswith('t+'):
                continue
            rows.append({
                'Model'  : model_name,
                'Horizon': horizon,
                'RMSE'   : metrics['RMSE'],
                'MAE'    : metrics['MAE'],
                'MAPE'   : metrics['MAPE'],
                'R2'     : metrics['R2'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'comparison_table.csv', index=False)

    print("\n" + "="*70)
    print("TABEL PERBANDINGAN SEMUA MODEL")
    print("="*70)
    print(df.to_string(index=False))

    print("\n  ★ Model terbaik per horizon (RMSE terendah):")
    for h in HORIZONS:
        h_key = f't+{h}h'
        sub   = df[df['Horizon'] == h_key].copy()
        if not sub.empty:
            best = sub.loc[sub['RMSE'].idxmin()]
            print(f"  t+{h}h → {best['Model']} "
                  f"(RMSE={best['RMSE']:.4f}, R²={best['R2']:.4f})")
    return df

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Training: LSTM | GRU | BiLSTM | CNN-LSTM | XGBoost")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 60)

    (X_train, y_train, X_val, y_val,
     X_test, y_test, scaler_target) = load_preprocessed()

    test_timestamps = pd.to_datetime(
        pd.read_csv(PREP_DIR / 'test_timestamps.csv').iloc[:, 0]
    ).values

    # ---- Model DL ----
    dl_builders = {
        'LSTM'    : build_lstm,
        'GRU'     : build_gru,
        'BiLSTM'  : build_bilstm,
        'CNN_LSTM': build_cnn_lstm,
    }

    all_results   = {}
    all_histories = {}

    print("\n" + "="*60)
    print("DEEP LEARNING MODELS")
    print("="*60)

    for model_name, builder in dl_builders.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        final_path = MODEL_DIR / f'{model_name}_final.h5'
        pred_path  = MODEL_DIR / f'{model_name}_y_pred.npy'

        if SKIP_IF_EXISTS and final_path.exists() and pred_path.exists():
            print(f"  [SKIP] {model_name} sudah ada, load...")
            model = load_model(str(final_path), compile=False)
            model.compile(optimizer=Adam(LR), loss='mse', metrics=['mae'])

            class FakeHistory:
                history = {'loss': [], 'val_loss': [],
                           'mae': [], 'val_mae': []}
            all_histories[model_name] = FakeHistory()
        else:
            model = builder(N_FEATURES, LOOKBACK, N_HORIZONS)
            history, t_elapsed = compile_and_train(
                model, X_train, y_train, X_val, y_val, model_name
            )
            all_histories[model_name] = history
            all_results[model_name] = {}
            all_results[model_name]['training_time_min'] = round(
                t_elapsed/60, 2
            )

        print(f"  Evaluasi {model_name}...")
        results = evaluate_model(model, X_test, y_test,
                                 scaler_target, model_name)
        all_results[model_name] = {**all_results.get(model_name, {}),
                                   **results}

    # ---- XGBoost ----
    print(f"\n{'='*60}")
    print("XGBOOST")
    print(f"{'='*60}")

    xgb_models, xgb_time = train_all_xgboost(
        X_train, y_train, X_val, y_val
    )
    print(f"\n  Evaluasi XGBoost...")
    xgb_results = evaluate_model(
        xgb_models, X_test, y_test, scaler_target,
        'XGBoost', is_xgboost=True
    )
    all_results['XGBoost'] = {
        **xgb_results,
        'training_time_min': round(xgb_time/60, 2)
    }

    # ---- Persistence baseline ----
    print(f"\n{'='*60}")
    print("Persistence (Naive Baseline)")
    print(f"{'='*60}")
    all_results['Persistence'] = evaluate_persistence(
        X_test, y_test, scaler_target
    )
    for h, m in all_results['Persistence'].items():
        if h.startswith('t+'):
            print(f"  {h} | RMSE={m['RMSE']:.4f}m | R²={m['R2']:.4f}")

    # ---- Visualisasi ----
    print("\n[VIZ] Membuat plot...")
    all_model_names = list(dl_builders.keys()) + ['XGBoost', 'Persistence']
    plot_training_history(all_histories)
    plot_predictions(all_model_names, X_test, y_test,
                     scaler_target, test_timestamps)
    plot_scatter_all(all_model_names, X_test, y_test, scaler_target)

    # ---- Tabel ----
    save_comparison_table(all_results)

    with open(RESULTS_DIR / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x))

    print(f"\n{'='*60}")
    print("SELESAI!")
    print(f"Models  : {MODEL_DIR.resolve()}")
    print(f"Results : {RESULTS_DIR.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()