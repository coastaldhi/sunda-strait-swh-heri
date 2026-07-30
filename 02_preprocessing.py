"""
============================================================
02_preprocessing.py
Preprocessing data ERA5 → array siap training model DL
============================================================
Jalankan SETELAH 01_eda.py selesai.

Output: folder data/preprocessed/ berisi file .npy
  X_train.npy, y_train.npy
  X_val.npy,   y_val.npy
  X_test.npy,  y_test.npy
  scaler_features.pkl, scaler_target.pkl
  preprocessing_config.json
============================================================
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI — sesuaikan sesuai kebutuhan
# ============================================================

PROCESSED_DIR = Path("data/era5_processed")
PREP_DIR      = Path("data/preprocessed")
PREP_DIR.mkdir(parents=True, exist_ok=True)

# Titik grid yang dipakai untuk training
# Gunakan titik 1 (tengah selat) sebagai titik utama
POINT_NAME = "titik1_tengah_selat"

# Kolom fitur input model
FEATURE_COLS = [
    'SWH',        # target sekaligus fitur (lag)
    'WS10',       # kecepatan angin
    'MWP',        # mean wave period
    'MWD_sin',    # arah gelombang (sin encoding)
    'MWD_cos',    # arah gelombang (cos encoding)
    'MSLP',       # tekanan permukaan laut
    'U10',        # komponen angin U
    'V10',        # komponen angin V
]

# Kolom target output
TARGET_COL = 'SWH'

# Parameter sliding window
LOOKBACK  = 72    # jam ke belakang yang dilihat model (3 hari)
HORIZONS  = [6, 12, 24]   # jam ke depan yang diprediksi

# Pembagian dataset
# Training: 2010–2022 | Validasi: 2023 | Testing: 2024–2025
TRAIN_END = '2022-12-31 23:00'
VAL_END   = '2023-12-31 23:00'
# Test: sisa data (2024–2025)

# ============================================================
# FUNGSI BANTU
# ============================================================

def load_data(point_name: str) -> pd.DataFrame:
    """Load CSV hasil merge_era5.py."""
    for suffix in ['2010_2025', '2010_2023']:
        path = PROCESSED_DIR / f"era5_{point_name}_{suffix}.csv"
        if path.exists():
            print(f"  Membaca: {path.name}")
            df = pd.read_csv(path, index_col='time', parse_dates=True)
            return df
    raise FileNotFoundError(
        f"File CSV tidak ditemukan untuk {point_name}.\n"
        f"Jalankan merge_era5.py terlebih dahulu."
    )


def check_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan semua kolom fitur tersedia.
    Kolom yang tidak ada diisi nol dengan peringatan.
    """
    print("\n  Cek ketersediaan fitur:")
    for col in FEATURE_COLS:
        if col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            status = "✓" if missing_pct < 1 else f"⚠ {missing_pct:.1f}% missing"
            print(f"    {col:<12}: {status}")
        else:
            print(f"    {col:<12}: ✗ tidak ada — diisi 0")
            df[col] = 0.0
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Isi missing values dengan interpolasi."""
    total_missing = df[FEATURE_COLS].isnull().sum().sum()
    if total_missing == 0:
        print("  ✓ Tidak ada missing values")
        return df

    print(f"  Mengisi {total_missing} missing values...")
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(
        method='linear', limit=6
    )
    # Isi sisa dengan forward/backward fill
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    remaining = df[FEATURE_COLS].isnull().sum().sum()
    print(f"  Sisa missing setelah interpolasi: {remaining}")
    return df


def normalize_data(df_train: pd.DataFrame,
                   df_val: pd.DataFrame,
                   df_test: pd.DataFrame):
    """
    Fit scaler HANYA pada data training, transform semua split.
    Penting: jangan fit pada val/test untuk mencegah data leakage.
    """
    print("\n  Normalisasi (MinMaxScaler, fit pada training saja)...")

    # Scaler untuk fitur
    scaler_feat = MinMaxScaler(feature_range=(0, 1))
    scaler_feat.fit(df_train[FEATURE_COLS])

    # Scaler terpisah untuk target (SWH) — untuk inverse transform prediksi
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(df_train[[TARGET_COL]])

    # Transform semua split
    def transform(df):
        arr = scaler_feat.transform(df[FEATURE_COLS])
        return pd.DataFrame(arr, columns=FEATURE_COLS, index=df.index)

    df_train_norm = transform(df_train)
    df_val_norm   = transform(df_val)
    df_test_norm  = transform(df_test)

    # Simpan scaler
    with open(PREP_DIR / 'scaler_features.pkl', 'wb') as f:
        pickle.dump(scaler_feat, f)
    with open(PREP_DIR / 'scaler_target.pkl', 'wb') as f:
        pickle.dump(scaler_target, f)

    print(f"  Scaler disimpan di {PREP_DIR}/")
    print(f"  Range SWH setelah normalisasi: "
          f"[{df_train_norm[TARGET_COL].min():.3f}, "
          f"{df_train_norm[TARGET_COL].max():.3f}]")

    return df_train_norm, df_val_norm, df_test_norm, scaler_target


def create_sequences(data: np.ndarray,
                     lookback: int,
                     horizons: list,
                     target_idx: int) -> tuple:
    """
    Buat sliding window sequences.

    Parameters
    ----------
    data       : array (n_timesteps, n_features) yang sudah dinormalisasi
    lookback   : jumlah timestep ke belakang sebagai input
    horizons   : list horizon prediksi, misal [6, 12, 24]
    target_idx : index kolom target (SWH) dalam array fitur

    Returns
    -------
    X : array (n_samples, lookback, n_features)
    y : array (n_samples, len(horizons))
        setiap kolom = prediksi untuk horizon ke-i
    """
    max_horizon = max(horizons)
    X, y = [], []

    for i in range(lookback, len(data) - max_horizon + 1):
        # Input: lookback jam ke belakang, semua fitur
        X.append(data[i - lookback:i, :])

        # Output: nilai SWH pada t+h untuk setiap horizon
        targets = []
        for h in horizons:
            targets.append(data[i + h - 1, target_idx])
        y.append(targets)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def print_shapes(split_name: str, X: np.ndarray, y: np.ndarray):
    """Cetak info shape array."""
    print(f"\n  {split_name}:")
    print(f"    X: {X.shape}  "
          f"({X.shape[0]:,} samples × {X.shape[1]} timesteps × {X.shape[2]} fitur)")
    print(f"    y: {y.shape}  "
          f"({y.shape[0]:,} samples × {y.shape[1]} horizons {HORIZONS})")
    mem_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
    print(f"    Ukuran memori: {mem_mb:.1f} MB")


def save_arrays(name: str, X: np.ndarray, y: np.ndarray):
    """Simpan array ke file .npy."""
    np.save(PREP_DIR / f"X_{name}.npy", X)
    np.save(PREP_DIR / f"y_{name}.npy", y)


def save_config():
    """Simpan konfigurasi preprocessing ke JSON."""
    config = {
        "point_name"  : POINT_NAME,
        "feature_cols": FEATURE_COLS,
        "target_col"  : TARGET_COL,
        "lookback"    : LOOKBACK,
        "horizons"    : HORIZONS,
        "train_end"   : TRAIN_END,
        "val_end"     : VAL_END,
        "n_features"  : len(FEATURE_COLS),
        "n_horizons"  : len(HORIZONS),
    }
    with open(PREP_DIR / 'preprocessing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config disimpan: preprocessing_config.json")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print("Preprocessing ERA5 → Array siap training")
    print("=" * 55)

    # ---- Load data ----
    print(f"\n[1] Load data: {POINT_NAME}")
    df = load_data(POINT_NAME)
    print(f"  Total: {len(df):,} baris | "
          f"{df.index.min().date()} s/d {df.index.max().date()}")

    # ---- Cek & siapkan fitur ----
    print("\n[2] Cek fitur...")
    df = check_features(df)

    # ---- Handle missing values ----
    print("\n[3] Handle missing values...")
    df = handle_missing(df)

    # ---- Split berdasarkan waktu ----
    print("\n[4] Split dataset...")
    df_train = df.loc[:TRAIN_END]
    df_val   = df.loc[TRAIN_END:VAL_END].iloc[1:]   # hindari overlap
    df_test  = df.loc[VAL_END:].iloc[1:]

    print(f"  Training : {len(df_train):>8,} baris "
          f"({df_train.index.min().date()} s/d {df_train.index.max().date()})")
    print(f"  Validasi : {len(df_val):>8,} baris "
          f"({df_val.index.min().date()} s/d {df_val.index.max().date()})")
    print(f"  Testing  : {len(df_test):>8,} baris "
          f"({df_test.index.min().date()} s/d {df_test.index.max().date()})")

    pct_train = len(df_train) / len(df) * 100
    pct_val   = len(df_val)   / len(df) * 100
    pct_test  = len(df_test)  / len(df) * 100
    print(f"  Proporsi : {pct_train:.1f}% / {pct_val:.1f}% / {pct_test:.1f}%")

    # ---- Normalisasi ----
    print("\n[5] Normalisasi...")
    df_train_n, df_val_n, df_test_n, scaler_target = normalize_data(
        df_train, df_val, df_test
    )

    # ---- Buat sliding window sequences ----
    print(f"\n[6] Buat sequences "
          f"(lookback={LOOKBACK}h, horizons={HORIZONS}h)...")

    target_idx = FEATURE_COLS.index(TARGET_COL)

    print("  Proses training set...", end=" ", flush=True)
    X_train, y_train = create_sequences(
        df_train_n.values, LOOKBACK, HORIZONS, target_idx
    )
    print("OK")

    print("  Proses validasi set...", end=" ", flush=True)
    X_val, y_val = create_sequences(
        df_val_n.values, LOOKBACK, HORIZONS, target_idx
    )
    print("OK")

    print("  Proses testing set...", end=" ", flush=True)
    X_test, y_test = create_sequences(
        df_test_n.values, LOOKBACK, HORIZONS, target_idx
    )
    print("OK")

    # ---- Info shape ----
    print("\n[7] Shape hasil sequences:")
    print_shapes("Training", X_train, y_train)
    print_shapes("Validasi",  X_val,   y_val)
    print_shapes("Testing",   X_test,  y_test)

    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"\n  Total samples: {total_samples:,}")

    # ---- Sanity check ----
    print("\n[8] Sanity check...")
    print(f"  Range X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Range y_train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  NaN di X_train: {np.isnan(X_train).sum()}")
    print(f"  NaN di y_train: {np.isnan(y_train).sum()}")

    # ---- Simpan ----
    print("\n[9] Menyimpan array...")
    save_arrays('train', X_train, y_train)
    save_arrays('val',   X_val,   y_val)
    save_arrays('test',  X_test,  y_test)

    # Simpan juga index waktu testing (untuk plot prediksi vs aktual nanti)
    test_times = df_test_n.index[LOOKBACK + max(HORIZONS) - 1:]
    pd.Series(test_times[:len(X_test)]).to_csv(
        PREP_DIR / 'test_timestamps.csv', index=False
    )

    save_config()

    # ---- Ringkasan ----
    print(f"\n{'='*55}")
    print(f"Preprocessing selesai!")
    print(f"File tersimpan di: {PREP_DIR.resolve()}")
    print(f"\nFile yang dihasilkan:")
    for f in sorted(PREP_DIR.iterdir()):
        size = f.stat().st_size / (1024*1024)
        print(f"  {f.name:<35} ({size:.1f} MB)")
    print(f"\nLangkah selanjutnya: jalankan 03_train_models.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
