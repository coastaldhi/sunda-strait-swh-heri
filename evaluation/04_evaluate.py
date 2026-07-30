"""
============================================================
04_evaluate.py
Evaluasi Mendalam & Visualisasi untuk Paper
============================================================
Jalankan SETELAH 03_train_models.py selesai.

Output: folder data/paper_figures/
  - Semua plot siap masuk paper (resolusi tinggi)
  - Analisis HERI dari hasil prediksi
  - Analisis kondisi mudik Lebaran
  - Tabel lengkap untuk section hasil
============================================================
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ============================================================
# KONFIGURASI
# ============================================================

PREP_DIR    = Path("data/preprocessed")
MODEL_DIR   = Path("data/models")
RESULTS_DIR = Path("data/results")
PAPER_DIR   = Path("data/paper_figures")
PAPER_DIR.mkdir(parents=True, exist_ok=True)

with open(PREP_DIR / 'preprocessing_config.json') as f:
    CONFIG = json.load(f)

HORIZONS   = CONFIG['horizons']    # [6, 12, 24]
LOOKBACK   = CONFIG['lookback']    # 72

# HERI constants
G               = 9.81
SWH_MAX         = 4.0
MWP_MIN         = 3.0
S_MAX           = (2 * np.pi) / (7 * G * MWP_MIN**2)
W1, W2, W3      = 0.50, 0.30, 0.20
HARBOUR_MERAK   = 270.0
HARBOUR_BKHENI  = 315.0

HERI_THRESHOLDS = [0.0, 0.25, 0.50, 0.75, 1.01]
HERI_LABELS     = ['AMAN', 'WASPADA', 'BERBAHAYA', 'SANGAT BERBAHAYA']
HERI_COLORS     = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

MODEL_COLORS = {
    'LSTM'       : '#3498db',
    'GRU'        : '#2ecc71',
    'BiLSTM'     : '#9b59b6',   # ← tambah
    'CNN_LSTM'   : '#e74c3c',
    'XGBoost'    : '#f39c12',   # ← tambah
    'Persistence': '#95a5a6',
    'Aktual'     : '#2c3e50',
}

LEBARAN = {
    2015: '2015-07-17', 2016: '2016-07-06', 2017: '2017-06-26',
    2018: '2018-06-15', 2019: '2019-06-05', 2020: '2020-05-24',
    2021: '2021-05-13', 2022: '2022-05-02', 2023: '2023-04-21',
    2024: '2024-04-10',
}

plt.rcParams.update({
    'figure.dpi'    : 150,
    'font.size'     : 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.grid'     : True,
    'grid.alpha'    : 0.3,
})

# ============================================================
# FUNGSI BANTU
# ============================================================

def save_fig(fig, filename: str, tight=True):
    path = PAPER_DIR / filename
    if tight:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Disimpan: {filename}")


def load_all():
    """Load semua data yang dibutuhkan."""
    print("[LOAD] Membaca data...")

    # Preprocessed arrays
    X_test = np.load(PREP_DIR / 'X_test.npy')
    y_test = np.load(PREP_DIR / 'y_test.npy')

    with open(PREP_DIR / 'scaler_target.pkl', 'rb') as f:
        scaler_target = pickle.load(f)
    with open(PREP_DIR / 'scaler_features.pkl', 'rb') as f:
        scaler_features = pickle.load(f)

    # Timestamps
    test_timestamps = pd.to_datetime(
        pd.read_csv(PREP_DIR / 'test_timestamps.csv').iloc[:, 0]
    )

    # CSV asli untuk data HERI aktual
    csv_files = list(Path("data/era5_processed").glob("era5_titik1*.csv"))
    df_raw = pd.read_csv(csv_files[0], index_col='time', parse_dates=True)

    # Prediksi semua model
    model_names = ['LSTM', 'GRU', 'CNN_LSTM']
    y_preds = {}
    for name in model_names:
        pred_path = MODEL_DIR / f'{name}_y_pred.npy'
        if pred_path.exists():
            y_preds[name] = np.load(pred_path)

    # Persistence
    swh_last = X_test[:, -1, 0]
    swh_last_real = scaler_target.inverse_transform(
        swh_last.reshape(-1, 1)).flatten()
    y_preds['Persistence'] = np.column_stack(
        [scaler_target.transform(swh_last_real.reshape(-1, 1))] * len(HORIZONS)
    )

    # Denormalisasi y_test
    y_test_real = {}
    for i, h in enumerate(HORIZONS):
        y_test_real[f't+{h}h'] = scaler_target.inverse_transform(
            y_test[:, i].reshape(-1, 1)).flatten()

    # Denormalisasi prediksi
    y_preds_real = {}
    for name, preds in y_preds.items():
        y_preds_real[name] = {}
        for i, h in enumerate(HORIZONS):
            y_preds_real[name][f't+{h}h'] = scaler_target.inverse_transform(
                preds[:, i].reshape(-1, 1)).flatten()

    print(f"  Test period: {test_timestamps.iloc[0].date()} "
          f"s/d {test_timestamps.iloc[-1].date()}")
    print(f"  Samples    : {len(test_timestamps):,}")

    return (X_test, y_test, y_test_real, y_preds_real,
            scaler_target, test_timestamps, df_raw)


def compute_heri_from_arrays(swh, mwp, mwd, theta=HARBOUR_MERAK):
    """Hitung HERI dari array numpy."""
    swh_norm  = np.clip(swh / SWH_MAX, 0, 1)
    mwp_safe  = np.where(mwp > 0, mwp, MWP_MIN)
    steepness = swh / (G * mwp_safe**2) * (2 * np.pi)
    s_norm    = np.clip(steepness / S_MAX, 0, 1)
    theta_rel = np.radians(mwd - theta)
    f_dir     = np.abs(np.cos(theta_rel))
    heri      = np.clip(W1*swh_norm + W2*s_norm + W3*f_dir, 0, 1)
    return heri


def categorize_heri(heri):
    cats = np.empty(len(heri), dtype=object)
    for i, (lo, hi) in enumerate(
        zip(HERI_THRESHOLDS[:-1], HERI_THRESHOLDS[1:])
    ):
        cats[(heri >= lo) & (heri < hi)] = HERI_LABELS[i]
    return cats


# ============================================================
# PLOT 1 — Perbandingan Metrik Semua Model (Bar Chart)
# ============================================================

def plot_metrics_comparison(y_test_real, y_preds_real):
    print("\n[1] Plot perbandingan metrik...")

    metrics_data = {}
    for name, preds in y_preds_real.items():
        metrics_data[name] = {}
        for h_key, y_true in y_test_real.items():
            y_pred = preds[h_key]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            r2   = r2_score(y_true, y_pred)
            mask = y_true > 0.01
            mape = np.mean(np.abs(
                (y_true[mask] - y_pred[mask]) / y_true[mask]
            )) * 100
            metrics_data[name][h_key] = {
                'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2
            }

    model_names = list(y_preds_real.keys())
    horizon_keys = [f't+{h}h' for h in HORIZONS]
    x = np.arange(len(horizon_keys))
    width = 0.2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perbandingan Metrik Evaluasi — Semua Model',
                 fontsize=13, fontweight='bold')

    metric_info = [
        ('RMSE', 'RMSE (m)', axes[0, 0], False),
        ('MAE',  'MAE (m)',  axes[0, 1], False),
        ('MAPE', 'MAPE (%)', axes[1, 0], False),
        ('R2',   'R²',       axes[1, 1], True),
    ]

    for metric, ylabel, ax, higher_better in metric_info:
        for j, name in enumerate(model_names):
            vals = [metrics_data[name][hk][metric] for hk in horizon_keys]
            offset = (j - len(model_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=name,
                          color=MODEL_COLORS.get(name, 'gray'),
                          alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.001,
                        f'{v:.3f}', ha='center', fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(horizon_keys)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        if higher_better:
            ax.set_ylim(0.6, 1.02)

    plt.tight_layout()
    save_fig(fig, "fig1_metrics_comparison.png")


# ============================================================
# PLOT 2 — Time Series Prediksi vs Aktual (3 Periode)
# ============================================================

def plot_timeseries_comparison(y_test_real, y_preds_real, test_timestamps):
    print("\n[2] Plot time series prediksi vs aktual...")

    # Pilih 3 periode: Januari, April (mudik), Juli
    periods = {
        'Januari 2024 (Musim Barat)' : ('2024-01-01', '2024-01-31'),
        'April 2024 (Mudik Lebaran)' : ('2024-04-01', '2024-04-30'),
        'Juli 2024 (Musim Timur)'    : ('2024-07-01', '2024-07-31'),
    }

    fig, axes = plt.subplots(len(periods), 1,
                              figsize=(16, 4 * len(periods)))
    fig.suptitle('Prediksi vs Aktual SWH — Horizon t+24h',
                 fontsize=13, fontweight='bold')

    h_key = 't+24h'
    y_true_all = y_test_real[h_key]

    for ax, (period_name, (start, end)) in zip(axes, periods.items()):
        mask = ((test_timestamps >= start) &
                (test_timestamps <= end))
        times = test_timestamps[mask]

        if mask.sum() == 0:
            ax.text(0.5, 0.5, 'Data tidak tersedia',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        y_true = y_true_all[mask]
        ax.plot(times, y_true, color=MODEL_COLORS['Aktual'],
                lw=1.5, label='Aktual', zorder=5)

        for name, preds in y_preds_real.items():
            y_pred = preds[h_key][mask]
            ax.plot(times, y_pred,
                    color=MODEL_COLORS.get(name, 'gray'),
                    lw=1, alpha=0.8, label=name)

        ax.axhline(2.5, color='red', ls='--', lw=0.8,
                   alpha=0.6, label='Threshold KSOP (2.5m)')
        ax.set_ylabel('SWH (m)')
        ax.set_title(period_name)
        ax.legend(fontsize=8, loc='upper right', ncol=3)

    plt.tight_layout()
    save_fig(fig, "fig2_timeseries_t24h.png")


# ============================================================
# PLOT 3 — Scatter Plot + Error Distribution
# ============================================================

def plot_scatter_error(y_test_real, y_preds_real):
    print("\n[3] Plot scatter + distribusi error...")

    model_names = [m for m in y_preds_real if m != 'Persistence']
    h_key = 't+24h'
    y_true = y_test_real[h_key]

    fig, axes = plt.subplots(2, len(model_names),
                              figsize=(5 * len(model_names), 10))
    fig.suptitle('Scatter Plot & Distribusi Error — t+24h',
                 fontsize=13, fontweight='bold')

    for j, name in enumerate(model_names):
        y_pred = y_preds_real[name][h_key]
        error  = y_pred - y_true

        # Scatter
        ax = axes[0, j]
        sc = ax.scatter(y_true, y_pred, alpha=0.15, s=3,
                        c=np.abs(error), cmap='YlOrRd', vmin=0, vmax=0.3)
        plt.colorbar(sc, ax=ax, label='|Error| (m)')
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'b--', lw=1, label='Perfect fit')
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.set_title(f'{name}\nR²={r2:.4f} | RMSE={rmse:.4f}m')
        ax.set_xlabel('Aktual SWH (m)')
        ax.set_ylabel('Prediksi SWH (m)')
        ax.legend(fontsize=8)

        # Distribusi error
        ax = axes[1, j]
        ax.hist(error, bins=80, color=MODEL_COLORS.get(name, 'gray'),
                edgecolor='white', alpha=0.8)
        ax.axvline(0, color='black', lw=1.5, ls='--')
        ax.axvline(error.mean(), color='red', lw=1.5,
                   label=f'Mean={error.mean():.4f}m')
        ax.axvline(error.std(), color='orange', lw=1, ls=':',
                   label=f'Std={error.std():.4f}m')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Frekuensi')
        ax.set_title(f'Distribusi Error — {name}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "fig3_scatter_error.png")


# ============================================================
# PLOT 4 — HERI dari Prediksi CNN-LSTM vs HERI Aktual
# ============================================================

def plot_heri_prediction(y_test_real, y_preds_real, test_timestamps, df_raw):
    print("\n[4] Plot HERI prediksi vs aktual...")

    # Ambil data aktual dari CSV (MWP dan MWD untuk HERI)
    df_test = df_raw.loc[
        (df_raw.index >= str(test_timestamps.iloc[0].date())) &
        (df_raw.index <= str(test_timestamps.iloc[-1].date()))
    ].copy()

    # Sejajarkan index
    df_aligned = df_test.reindex(test_timestamps)

    swh_actual = y_test_real['t+24h']
    mwp_actual = df_aligned['MWP'].values
    mwd_actual = df_aligned['MWD'].values

    # HERI aktual
    valid_mask = (~np.isnan(mwp_actual) & ~np.isnan(mwd_actual))
    heri_actual = np.full(len(swh_actual), np.nan)
    heri_actual[valid_mask] = compute_heri_from_arrays(
        swh_actual[valid_mask],
        mwp_actual[valid_mask],
        mwd_actual[valid_mask]
    )

    # HERI prediksi CNN-LSTM (hanya SWH yang diprediksi,
    # MWP dan MWD pakai nilai aktual sebagai proxy)
    swh_pred_cnn = y_preds_real['CNN_LSTM']['t+24h']
    heri_pred = np.full(len(swh_pred_cnn), np.nan)
    heri_pred[valid_mask] = compute_heri_from_arrays(
        swh_pred_cnn[valid_mask],
        mwp_actual[valid_mask],
        mwd_actual[valid_mask]
    )

    # Ambil periode April 2024 (mudik)
    mask_apr = ((test_timestamps >= '2024-04-01') &
                (test_timestamps <= '2024-04-30'))
    times_apr = test_timestamps[mask_apr]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('HERI Aktual vs Prediksi CNN-LSTM — April 2024 (Mudik Lebaran)',
                 fontsize=13, fontweight='bold')

    # Panel 1: SWH
    ax = axes[0]
    ax.plot(times_apr, swh_actual[mask_apr], 'k-', lw=1.5, label='SWH Aktual')
    ax.plot(times_apr, swh_pred_cnn[mask_apr], color='#e74c3c',
            lw=1, alpha=0.8, label='SWH Prediksi (CNN-LSTM)')
    ax.axhline(2.5, color='red', ls='--', lw=0.8, alpha=0.6,
               label='Threshold KSOP')
    ax.axvline(pd.Timestamp('2024-04-10'), color='green',
               ls=':', lw=1.5, label='Hari Lebaran (10 Apr)')
    ax.set_ylabel('SWH (m)')
    ax.set_title('Significant Wave Height')
    ax.legend(fontsize=8, ncol=2)

    # Panel 2: HERI time series
    ax = axes[1]
    if not np.all(np.isnan(heri_actual[mask_apr])):
        ax.plot(times_apr, heri_actual[mask_apr], 'k-',
                lw=1.5, label='HERI Aktual')
    if not np.all(np.isnan(heri_pred[mask_apr])):
        ax.plot(times_apr, heri_pred[mask_apr], color='#e74c3c',
                lw=1, alpha=0.8, label='HERI Prediksi')

    for i, (lo, hi) in enumerate(
        zip(HERI_THRESHOLDS[:-1], HERI_THRESHOLDS[1:])
    ):
        ax.axhspan(lo, hi, alpha=0.08, color=HERI_COLORS[i])
    for thresh, color, label in zip(
        [0.25, 0.50, 0.75],
        HERI_COLORS[1:],
        HERI_LABELS[1:]
    ):
        ax.axhline(thresh, color=color, ls=':', lw=0.8, alpha=0.7)

    ax.axvline(pd.Timestamp('2024-04-10'), color='green', ls=':', lw=1.5)
    ax.set_ylabel('HERI (0–1)')
    ax.set_ylim(0, 1)
    ax.set_title('Harbour Entry Risk Index (HERI)')
    legend_els = [mpatches.Patch(color=c, alpha=0.5, label=l)
                  for c, l in zip(HERI_COLORS, HERI_LABELS)]
    ax.legend(handles=legend_els + [
        plt.Line2D([0], [0], color='k', lw=1.5, label='Aktual'),
        plt.Line2D([0], [0], color='#e74c3c', lw=1, label='Prediksi'),
    ], fontsize=8, ncol=3)

    # Panel 3: Kategori HERI per jam (stacked bar)
    ax = axes[2]
    heri_act_apr = heri_actual[mask_apr]
    heri_prd_apr = heri_pred[mask_apr]

    # Hitung proporsi kategori per hari
    days = pd.date_range('2024-04-01', '2024-04-30', freq='D')
    pct_act = {cat: [] for cat in HERI_LABELS}
    pct_prd = {cat: [] for cat in HERI_LABELS}

    for day in days:
        day_mask = ((times_apr.date == day.date())
                    if hasattr(times_apr, 'date')
                    else np.array([t.date() == day.date()
                                   for t in times_apr]))
        for cat, pct_dict, heri_arr in [
            (None, pct_act, heri_act_apr),
            (None, pct_prd, heri_prd_apr)
        ]:
            sub = heri_arr[day_mask]
            sub = sub[~np.isnan(sub)]
            for lbl in HERI_LABELS:
                lo = HERI_THRESHOLDS[HERI_LABELS.index(lbl)]
                hi = HERI_THRESHOLDS[HERI_LABELS.index(lbl) + 1]
                pct = ((sub >= lo) & (sub < hi)).mean() * 100 if len(sub) > 0 else 0
                pct_dict[lbl].append(pct)

    x_days = np.arange(len(days))
    bottom = np.zeros(len(days))
    for lbl, color in zip(HERI_LABELS, HERI_COLORS):
        vals = np.array(pct_act[lbl])
        ax.bar(x_days, vals, bottom=bottom, color=color,
               alpha=0.8, width=0.8)
        bottom += vals

    ax.set_xticks(x_days[::2])
    ax.set_xticklabels([d.strftime('%d') for d in days[::2]], fontsize=8)
    ax.set_xlabel('Tanggal April 2024')
    ax.set_ylabel('% Jam per Kategori')
    ax.set_title('Distribusi Kategori HERI Harian (Aktual)')
    ax.axvline(9, color='green', ls=':', lw=1.5, label='Lebaran (10 Apr)')
    legend_els = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(HERI_COLORS, HERI_LABELS)]
    ax.legend(handles=legend_els, fontsize=8, ncol=2)

    plt.tight_layout()
    save_fig(fig, "fig4_heri_prediction_mudik.png")


# ============================================================
# PLOT 5 — Analisis Mudik Multi-tahun
# ============================================================

def plot_mudik_multiyear(df_raw, y_test_real, y_preds_real, test_timestamps):
    print("\n[5] Plot analisis mudik multi-tahun...")

    results = []
    for year, date_str in LEBARAN.items():
        center = pd.Timestamp(date_str)

        # Data historis dari CSV
        for d_start, d_end, periode in [
            (-14, -8, 'Pra-Mudik'),
            (-7,  -1, 'Arus Mudik'),
            (-3,   3, 'Puncak'),
            ( 1,   7, 'Arus Balik'),
        ]:
            mask = (
                (df_raw.index >= center + pd.Timedelta(days=d_start)) &
                (df_raw.index <= center + pd.Timedelta(days=d_end))
            )
            sub = df_raw.loc[mask, 'SWH'].dropna()
            if len(sub) == 0:
                continue

            pct_heri = 0
            if 'HERI' in df_raw.columns:
                pct_heri = df_raw.loc[mask, 'HERI_cat'].isin(
                    ['BERBAHAYA', 'SANGAT BERBAHAYA']
                ).mean() * 100

            results.append({
                'Tahun'   : year,
                'Periode' : periode,
                'Mean SWH': round(sub.mean(), 3),
                'Max SWH' : round(sub.max(), 3),
                'HERI%'   : round(pct_heri, 1),
                '% > 2.5m': round((sub >= 2.5).mean() * 100, 1),
            })

    df_mudik = pd.DataFrame(results)
    df_mudik.to_csv(PAPER_DIR / 'mudik_analysis.csv', index=False)
    print(f"  Tabel mudik disimpan: mudik_analysis.csv")

    # Pivot untuk heatmap
    pivot_heri = df_mudik.pivot(index='Tahun', columns='Periode',
                                 values='HERI%')
    pivot_swh  = df_mudik.pivot(index='Tahun', columns='Periode',
                                 values='Mean SWH')

    # Urutkan kolom
    col_order = ['Pra-Mudik', 'Arus Mudik', 'Puncak', 'Arus Balik']
    pivot_heri = pivot_heri.reindex(
        columns=[c for c in col_order if c in pivot_heri.columns]
    )
    pivot_swh  = pivot_swh.reindex(
        columns=[c for c in col_order if c in pivot_swh.columns]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Analisis Kondisi Gelombang Selama Mudik Lebaran 2015–2024\n'
                 'Selat Sunda (titik -6.0°S, 105.5°E)',
                 fontsize=12, fontweight='bold')

    # Heatmap HERI
    ax = axes[0]
    sns.heatmap(pivot_heri, ax=ax, annot=True, fmt='.1f',
                cmap='YlOrRd', vmin=0, vmax=40,
                cbar_kws={'label': '% Jam HERI ≥ Berbahaya'},
                linewidths=0.5)
    ax.set_title('% Jam HERI ≥ Berbahaya per Sub-Periode')
    ax.set_xlabel('')
    ax.set_ylabel('Tahun')
    ax.tick_params(axis='x', rotation=30)

    # Heatmap SWH
    ax = axes[1]
    sns.heatmap(pivot_swh, ax=ax, annot=True, fmt='.3f',
                cmap='Blues',
                cbar_kws={'label': 'Mean SWH (m)'},
                linewidths=0.5)
    ax.set_title('Mean SWH (m) per Sub-Periode')
    ax.set_xlabel('')
    ax.set_ylabel('Tahun')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    save_fig(fig, "fig5_mudik_heatmap.png")

    # Print ringkasan
    print(f"\n  Tahun mudik paling berisiko (HERI% saat puncak):")
    if 'Puncak' in pivot_heri.columns:
        top3 = pivot_heri['Puncak'].nlargest(3)
        for yr, val in top3.items():
            lebaran_date = LEBARAN.get(yr, '-')
            print(f"    {yr} ({lebaran_date}): {val:.1f}% jam berbahaya")


# ============================================================
# PLOT 6 — Seasonal Error Analysis
# ============================================================

def plot_seasonal_error(y_test_real, y_preds_real, test_timestamps):
    print("\n[6] Plot analisis error per musim...")

    h_key = 't+24h'
    y_true = y_test_real[h_key]

    # Tambahkan info musim
    months = pd.DatetimeIndex(test_timestamps).month
    seasons = {
        'Musim Barat\n(Des-Feb)' : [12, 1, 2],
        'Peralihan I\n(Mar-Mei)' : [3, 4, 5],
        'Musim Timur\n(Jun-Ags)' : [6, 7, 8],
        'Peralihan II\n(Sep-Nov)': [9, 10, 11],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Analisis Error per Musim — Horizon t+24h',
                 fontsize=13, fontweight='bold')

    # RMSE per musim per model
    ax = axes[0]
    season_names = list(seasons.keys())
    x = np.arange(len(season_names))
    width = 0.2

    for j, name in enumerate([m for m in y_preds_real if m != 'Persistence']):
        y_pred = y_preds_real[name][h_key]
        rmses  = []
        for season, months_list in seasons.items():
            mask = np.isin(months, months_list)
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(
                    y_true[mask], y_pred[mask]
                ))
            else:
                rmse = 0
            rmses.append(rmse)

        offset = (j - 1) * width
        ax.bar(x + offset, rmses, width,
               label=name,
               color=MODEL_COLORS.get(name, 'gray'),
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(season_names, fontsize=9)
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE per Musim')
    ax.legend(fontsize=8)

    # Boxplot error per musim (CNN-LSTM)
    ax = axes[1]
    y_pred_cnn = y_preds_real['CNN_LSTM'][h_key]
    errors_by_season = []
    season_labels = []
    for season, months_list in seasons.items():
        mask = np.isin(months, months_list)
        if mask.sum() > 0:
            err = y_pred_cnn[mask] - y_true[mask]
            errors_by_season.append(err)
            season_labels.append(season)

    bp = ax.boxplot(errors_by_season, patch_artist=True,
                    showfliers=False,
                    labels=season_labels)
    colors_season = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    for patch, color in zip(bp['boxes'], colors_season):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set_ylabel('Error (m)')
    ax.set_title('Distribusi Error CNN-LSTM per Musim')
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    save_fig(fig, "fig6_seasonal_error.png")


# ============================================================
# TABEL PAPER — Format siap masuk paper
# ============================================================

def generate_paper_tables(y_test_real, y_preds_real):
    print("\n[TABLE] Generate tabel untuk paper...")

    rows = []
    for name, preds in y_preds_real.items():
        for h in HORIZONS:
            h_key  = f't+{h}h'
            y_true = y_test_real[h_key]
            y_pred = preds[h_key]
            rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
            mae    = mean_absolute_error(y_true, y_pred)
            r2     = r2_score(y_true, y_pred)
            mask   = y_true > 0.01
            mape   = np.mean(np.abs(
                (y_true[mask] - y_pred[mask]) / y_true[mask]
            )) * 100
            rows.append({
                'Model'  : name,
                'Horizon': h_key,
                'RMSE (m)': f'{rmse:.4f}',
                'MAE (m)' : f'{mae:.4f}',
                'MAPE (%)': f'{mape:.2f}',
                'R²'      : f'{r2:.4f}',
            })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(PAPER_DIR / 'table_comparison.csv', index=False)

    print("\n" + "="*70)
    print("TABEL LENGKAP UNTUK PAPER")
    print("="*70)
    print(df_table.to_string(index=False))

    # Highlight terbaik
    print("\n  ★ Model terbaik per horizon (RMSE terendah):")
    for h in HORIZONS:
        h_key = f't+{h}h'
        sub   = df_table[df_table['Horizon'] == h_key].copy()
        sub['RMSE_float'] = sub['RMSE (m)'].astype(float)
        best  = sub.loc[sub['RMSE_float'].idxmin()]
        print(f"    t+{h}h → {best['Model']} "
              f"(RMSE={best['RMSE (m)']}m, R²={best['R²']})")

    return df_table


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Evaluasi Mendalam — Prediksi Gelombang Selat Sunda")
    print("=" * 60)

    # Load semua data
    (X_test, y_test, y_test_real, y_preds_real,
     scaler_target, test_timestamps, df_raw) = load_all()

    # Generate semua plot
    plot_metrics_comparison(y_test_real, y_preds_real)
    plot_timeseries_comparison(y_test_real, y_preds_real, test_timestamps)
    plot_scatter_error(y_test_real, y_preds_real)
    plot_heri_prediction(y_test_real, y_preds_real, test_timestamps, df_raw)
    plot_mudik_multiyear(df_raw, y_test_real, y_preds_real, test_timestamps)
    plot_seasonal_error(y_test_real, y_preds_real, test_timestamps)

    # Tabel untuk paper
    generate_paper_tables(y_test_real, y_preds_real)

    print(f"\n{'='*60}")
    print("SELESAI! Semua output tersimpan di:")
    print(f"  {PAPER_DIR.resolve()}")
    print(f"\nFile yang dihasilkan:")
    for f in sorted(PAPER_DIR.iterdir()):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<45} ({size:.0f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
