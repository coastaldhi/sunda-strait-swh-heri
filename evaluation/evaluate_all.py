"""
============================================================
evaluate_all.py
Evaluasi & Perbandingan Semua Versi Eksperimen

Jalankan kapan saja — otomatis deteksi versi yang sudah selesai.
Tidak perlu argumen apapun, langsung Run saja.

Output: data/evaluation_summary/
  - comparison_all_versions.csv   ← tabel besar semua versi
  - best_model_summary.csv        ← model terbaik per versi
  - fig_compare_versions.png      ← plot perbandingan
  - fig_best_predictions.png      ← prediksi model terbaik
============================================================
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# KONFIGURASI
# ============================================================

OUTPUT_DIR = Path("data/evaluation_summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Deskripsi per versi
VERSION_DESC = {
    'v2': 'Split 86/7/7  | Lookback 72h',
    'v3': 'Split 86/7/7  | Lookback 120h',
    'v4': 'Split 70/15/15 | Lookback 72h',
    'v5': 'Split 70/15/15 | Lookback 120h',
}

MODEL_COLORS = {
    'LSTM'       : '#3498db',
    'GRU'        : '#2ecc71',
    'BiLSTM'     : '#9b59b6',
    'CNN_LSTM'   : '#e74c3c',
    'XGBoost'    : '#f39c12',
    'Persistence': '#95a5a6',
}

plt.rcParams.update({
    'figure.dpi'    : 120,
    'font.size'     : 10,
    'axes.titlesize': 11,
    'axes.grid'     : True,
    'grid.alpha'    : 0.3,
})

# ============================================================
# FUNGSI BANTU
# ============================================================

def get_prep_dir(version):
    """Lokasi preprocessed data per versi."""
    if version == 'v2':
        return Path("data/preprocessed")
    return Path(f"data/preprocessed_{version}")


def get_model_dir(version):
    if version == 'v2':
        return Path("data/models")
    return Path(f"data/models_{version}")


def get_results_dir(version):
    if version == 'v2':
        return Path("data/results")
    return Path(f"data/results_{version}")


def detect_finished_versions():
    """Deteksi otomatis versi mana yang sudah selesai training."""
    finished = []
    for v in ['v2', 'v3', 'v4', 'v5']:
        results_dir = get_results_dir(v)
        table_path  = results_dir / 'comparison_table.csv'
        if table_path.exists():
            finished.append(v)
            print(f"  ✓ {v} ({VERSION_DESC.get(v, '')}) — ditemukan")
        else:
            print(f"  ✗ {v} — belum selesai, dilewati")
    return finished


def load_version_results(version):
    """Load comparison table dari satu versi."""
    path = get_results_dir(version) / 'comparison_table.csv'
    df   = pd.read_csv(path)
    df['Version'] = version
    df['Desc']    = VERSION_DESC.get(version, version)
    return df


def load_predictions(version, model_name):
    """Load array prediksi dari satu versi + model."""
    model_dir = get_model_dir(version)
    path      = model_dir / f'{model_name}_y_pred.npy'
    if path.exists():
        return np.load(path)
    return None


def load_test_data(version):
    """Load X_test, y_test, scaler, timestamps."""
    prep_dir = get_prep_dir(version)

    y_test  = np.load(prep_dir / 'y_test.npy')
    X_test  = np.load(prep_dir / 'X_test.npy')

    with open(prep_dir / 'scaler_target.pkl', 'rb') as f:
        scaler = pickle.load(f)

    timestamps = pd.to_datetime(
        pd.read_csv(prep_dir / 'test_timestamps.csv').iloc[:, 0]
    )

    with open(prep_dir / 'preprocessing_config.json') as f:
        config = json.load(f)

    return X_test, y_test, scaler, timestamps, config


def denorm(arr, scaler):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# ============================================================
# PLOT 1 — Perbandingan RMSE & R² Semua Versi
# ============================================================

def plot_compare_versions(df_all, finished_versions):
    print("\n[PLOT 1] Perbandingan semua versi...")

    horizons = ['t+6h', 't+12h', 't+24h']
    models   = [m for m in df_all['Model'].unique()
                if m != 'Persistence']

    fig, axes = plt.subplots(2, len(horizons),
                              figsize=(6 * len(horizons), 10))
    fig.suptitle('Perbandingan RMSE & R² — Semua Versi Eksperimen',
                 fontsize=13, fontweight='bold')

    for col, h in enumerate(horizons):
        sub = df_all[df_all['Horizon'] == h]

        # RMSE
        ax = axes[0, col]
        x  = np.arange(len(finished_versions))
        w  = 0.8 / len(models)
        for j, model in enumerate(models):
            m_sub = sub[sub['Model'] == model]
            vals  = [m_sub[m_sub['Version'] == v]['RMSE'].values[0]
                     if len(m_sub[m_sub['Version'] == v]) > 0 else 0
                     for v in finished_versions]
            offset = (j - len(models)/2 + 0.5) * w
            ax.bar(x + offset, vals, w,
                   label=model,
                   color=MODEL_COLORS.get(model, 'gray'),
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(finished_versions)
        ax.set_ylabel('RMSE (m)')
        ax.set_title(f'RMSE — {h}')
        if col == 0:
            ax.legend(fontsize=7)

        # R²
        ax = axes[1, col]
        for j, model in enumerate(models):
            m_sub = sub[sub['Model'] == model]
            vals  = [m_sub[m_sub['Version'] == v]['R2'].values[0]
                     if len(m_sub[m_sub['Version'] == v]) > 0 else 0
                     for v in finished_versions]
            offset = (j - len(models)/2 + 0.5) * w
            ax.bar(x + offset, vals, w,
                   label=model,
                   color=MODEL_COLORS.get(model, 'gray'),
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(finished_versions)
        ax.set_ylabel('R²')
        ax.set_title(f'R² — {h}')
        ax.set_ylim(0.6, 1.02)
        if col == 0:
            ax.legend(fontsize=7)

    # Tambahkan deskripsi versi di bawah
    desc_text = ' | '.join([f"{v}: {VERSION_DESC.get(v,'')}"
                             for v in finished_versions])
    fig.text(0.5, -0.01, desc_text, ha='center', fontsize=8,
             style='italic', color='gray')

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig_compare_versions.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan: {path.name}")


# ============================================================
# PLOT 2 — Heatmap RMSE: Version × Model (per horizon)
# ============================================================

def plot_heatmap_versions(df_all, finished_versions):
    print("\n[PLOT 2] Heatmap RMSE per versi × model...")

    horizons = ['t+6h', 't+12h', 't+24h']
    models   = [m for m in df_all['Model'].unique()
                if m != 'Persistence']

    fig, axes = plt.subplots(1, len(horizons),
                              figsize=(6 * len(horizons), 5))
    fig.suptitle('Heatmap RMSE — Versi vs Model',
                 fontsize=13, fontweight='bold')

    for ax, h in zip(axes, horizons):
        sub   = df_all[df_all['Horizon'] == h]
        pivot = sub[sub['Model'].isin(models)].pivot(
            index='Version', columns='Model', values='RMSE'
        )
        pivot = pivot.reindex(finished_versions)

        sns.heatmap(pivot, ax=ax, annot=True, fmt='.4f',
                    cmap='YlOrRd_r',
                    cbar_kws={'label': 'RMSE (m)'},
                    linewidths=0.5)
        ax.set_title(f'RMSE — {h}')
        ax.set_xlabel('')
        ax.set_ylabel('Versi')

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig_heatmap_rmse.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan: {path.name}")


# ============================================================
# PLOT 3 — Prediksi Model Terbaik (CNN-LSTM/BiLSTM) per Versi
# ============================================================

def plot_best_predictions(finished_versions):
    print("\n[PLOT 3] Plot prediksi model terbaik per versi...")

    # Model terbaik berdasarkan hasil v2
    best_model = 'CNN_LSTM'
    h_idx      = 2   # t+24h

    fig, axes = plt.subplots(len(finished_versions), 1,
                              figsize=(16, 4 * len(finished_versions)),
                              sharex=False)
    if len(finished_versions) == 1:
        axes = [axes]

    fig.suptitle(f'Prediksi {best_model} vs Aktual (t+24h) — Januari 2024',
                 fontsize=13, fontweight='bold')

    for ax, version in zip(axes, finished_versions):
        try:
            X_test, y_test, scaler, timestamps, config = load_test_data(version)
        except Exception as e:
            ax.text(0.5, 0.5, f'{version}: data tidak tersedia\n{e}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Filter Januari 2024
        mask = ((timestamps >= '2024-01-01') &
                (timestamps <= '2024-01-31'))
        if mask.sum() == 0:
            # Fallback: 30 hari pertama
            mask = np.zeros(len(timestamps), dtype=bool)
            mask[:min(720, len(timestamps))] = True

        times  = timestamps[mask]
        y_true = denorm(y_test[mask, h_idx], scaler)

        ax.plot(times, y_true, 'k-', lw=1.5, label='Aktual', zorder=5)

        # Plot semua model yang tersedia
        for model_name in ['LSTM', 'GRU', 'BiLSTM', 'CNN_LSTM', 'XGBoost']:
            pred_path = get_model_dir(version) / f'{model_name}_y_pred.npy'
            if not pred_path.exists():
                continue
            y_pred_norm = np.load(pred_path)
            y_pred = denorm(y_pred_norm[mask, h_idx], scaler)
            ax.plot(times, y_pred,
                    color=MODEL_COLORS.get(model_name, 'gray'),
                    lw=1, alpha=0.8, label=model_name)

        ax.axhline(2.5, color='red', ls='--', lw=0.8,
                   alpha=0.5, label='Threshold KSOP')
        ax.set_ylabel('SWH (m)')
        ax.set_title(f'{version}: {VERSION_DESC.get(version, "")}')
        ax.legend(fontsize=7, loc='upper right', ncol=4)

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig_best_predictions.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan: {path.name}")


# ============================================================
# TABEL RINGKASAN
# ============================================================

def generate_summary_tables(df_all, finished_versions):
    print("\n[TABLE] Generate tabel ringkasan...")

    # Tabel 1: Semua hasil gabungan
    df_all.to_csv(OUTPUT_DIR / 'comparison_all_versions.csv', index=False)
    print(f"  Disimpan: comparison_all_versions.csv")

    # Tabel 2: Model terbaik per versi per horizon
    rows = []
    for version in finished_versions:
        sub = df_all[(df_all['Version'] == version) &
                     (df_all['Model'] != 'Persistence')]
        for h in ['t+6h', 't+12h', 't+24h']:
            h_sub = sub[sub['Horizon'] == h]
            if h_sub.empty:
                continue
            best = h_sub.loc[h_sub['RMSE'].idxmin()]
            rows.append({
                'Version' : version,
                'Desc'    : VERSION_DESC.get(version, ''),
                'Horizon' : h,
                'Best Model': best['Model'],
                'RMSE'    : best['RMSE'],
                'MAE'     : best['MAE'],
                'MAPE'    : best['MAPE'],
                'R2'      : best['R2'],
            })

    df_best = pd.DataFrame(rows)
    df_best.to_csv(OUTPUT_DIR / 'best_model_summary.csv', index=False)

    print(f"\n{'='*75}")
    print("RINGKASAN MODEL TERBAIK PER VERSI")
    print(f"{'='*75}")
    print(df_best.to_string(index=False))

    # Cari versi + model terbaik secara keseluruhan (t+24h)
    t24 = df_best[df_best['Horizon'] == 't+24h']
    if not t24.empty:
        overall_best = t24.loc[t24['RMSE'].idxmin()]
        print(f"\n  🏆 TERBAIK KESELURUHAN (t+24h):")
        print(f"     Versi : {overall_best['Version']} "
              f"({overall_best['Desc']})")
        print(f"     Model : {overall_best['Best Model']}")
        print(f"     RMSE  : {overall_best['RMSE']:.4f} m")
        print(f"     R²    : {overall_best['R2']:.4f}")

    return df_best


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Evaluate & Compare — Semua Versi Eksperimen")
    print("=" * 60)

    # Deteksi versi yang sudah selesai
    print("\nDeteksi versi yang sudah selesai:")
    finished = detect_finished_versions()

    if not finished:
        print("\n[ERROR] Tidak ada versi yang selesai!")
        print("Jalankan train_universal.py terlebih dahulu.")
        return

    print(f"\nTotal versi ditemukan: {len(finished)} — {finished}")

    # Load semua hasil
    print("\n[LOAD] Membaca semua comparison_table...")
    dfs = []
    for v in finished:
        df_v = load_version_results(v)
        dfs.append(df_v)
        print(f"  {v}: {len(df_v)} baris")

    df_all = pd.concat(dfs, ignore_index=True)

    # Generate semua output
    plot_compare_versions(df_all, finished)
    plot_heatmap_versions(df_all, finished)
    plot_best_predictions(finished)
    df_best = generate_summary_tables(df_all, finished)

    print(f"\n{'='*60}")
    print("SELESAI!")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"\nFile yang dihasilkan:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<45} ({size:.0f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
