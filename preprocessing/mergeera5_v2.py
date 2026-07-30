"""
============================================================
merge_era5.py
Menggabungkan semua file ERA5 .nc per bulan menjadi
CSV siap pakai untuk training model

Urutan penggunaan:
  1. download_era5_fixed.py   ← download semua .nc per bulan
  2. merge_era5.py            ← script ini, gabung jadi CSV
  3. 01_eda.py                ← eksplorasi data
  4. 02_preprocessing.py      ← sliding window, normalisasi
  5. 03_train_models.py       ← training LSTM/GRU/CNN-LSTM

Output:
  data/era5_processed/
  ├── era5_titik1_tengah_selat_2010_2023.csv
  ├── era5_titik2_selatan_samudera_2010_2023.csv
  └── era5_titik3_utara_merak_2010_2023.csv
============================================================
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI — sesuaikan dengan lokasi folder kamu
# ============================================================

RAW_DIR       = Path("data/era5_raw")        # lokasi file .nc hasil download
PROCESSED_DIR = Path("data/era5_processed")  # output CSV
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tiga titik grid representatif Selat Sunda
GRID_POINTS = {
    "titik1_tengah_selat" : {"lat": -6.00, "lon": 105.5},  # sudah ada, skip otomatis
    "titik2_lampung_timur": {"lat": -5.50, "lon": 106.0},
    "titik3_teluk_semangka": {"lat": -6.00, "lon": 105.0},
    "titik4_pintu_samudera": {"lat": -6.50, "lon": 105.0},
    "titik5_pandeglang"    : {"lat": -6.50, "lon": 105.5},
}

YEARS  = list(range(2010, 2026))
MONTHS = list(range(1, 13))

# Orientasi mulut pelabuhan untuk HERI (derajat)
HARBOUR_MERAK     = 270.0
HARBOUR_BAKAUHENI = 315.0

# Konstanta fisika untuk HERI
G       = 9.81
SWH_MAX = 4.0
MWP_MIN = 3.0
S_MAX   = (2 * np.pi) / (7 * G * MWP_MIN**2)
W1, W2, W3 = 0.50, 0.30, 0.20

# ============================================================
# PETA NAMA VARIABEL ERA5 → nama standar kita
# ERA5 kadang pakai nama pendek atau panjang tergantung versi
# ============================================================

RENAME_WAVE = {
    # nama pendek
    'swh' : 'SWH',
    'mwp' : 'MWP',
    'mwd' : 'MWD',
    'pp1d': 'PWP',
    # nama panjang (versi baru CDS)
    'significant_height_of_combined_wind_waves_and_swell': 'SWH',
    'mean_wave_period'   : 'MWP',
    'mean_wave_direction': 'MWD',
    'peak_wave_period'   : 'PWP',
}

RENAME_SURFACE = {
    'u10' : 'U10',
    'v10' : 'V10',
    'msl' : 'MSLP',
    'sst' : 'SST',
    '10m_u_component_of_wind'  : 'U10',
    '10m_v_component_of_wind'  : 'V10',
    'mean_sea_level_pressure'  : 'MSLP',
    'sea_surface_temperature'  : 'SST',
}

# ============================================================
# FUNGSI BANTU
# ============================================================

def load_nc_point(nc_path: Path, lat: float, lon: float,
                  rename_map: dict) -> pd.DataFrame:
    try:
        ds = xr.open_dataset(nc_path, engine='netcdf4')
        ds_pt = ds.sel(latitude=lat, longitude=lon, method='nearest')

        df = ds_pt.to_dataframe().reset_index()

        # Set index waktu
        if 'valid_time' in df.columns:
            df = df.set_index('valid_time')
        elif 'time' in df.columns:
            df = df.set_index('time')
        else:
            print(f"    [ERROR] Tidak ada kolom waktu di {nc_path.name}")
            ds.close()
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index)
        df.index.name = 'time'
        df = df.drop(columns=['latitude', 'longitude', 'number', 'expver'],
                     errors='ignore')
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                if k in df.columns})
        keep = [c for c in df.columns if c in rename_map.values()]
        df = df[keep]

        ds.close()
        return df

    except Exception as e:
        print(f"    [ERROR] Gagal baca {nc_path.name}: {e}")
        return pd.DataFrame()


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung fitur turunan dari variabel dasar ERA5."""
    df = df.copy()

    # Kecepatan & arah angin
    if 'U10' in df.columns and 'V10' in df.columns:
        df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
        df['WD10'] = (np.degrees(np.arctan2(df['U10'], df['V10'])) + 360) % 360

    # Sin/cos encoding arah gelombang (hindari discontinuity 359→0)
    if 'MWD' in df.columns:
        df['MWD_sin'] = np.sin(np.radians(df['MWD']))
        df['MWD_cos'] = np.cos(np.radians(df['MWD']))

    # Sin/cos encoding arah angin
    if 'WD10' in df.columns:
        df['WD10_sin'] = np.sin(np.radians(df['WD10']))
        df['WD10_cos'] = np.cos(np.radians(df['WD10']))

    # Konversi MSLP dari Pa ke hPa
    if 'MSLP' in df.columns:
        df['MSLP'] = df['MSLP'] / 100.0

    return df


def compute_heri(df: pd.DataFrame,
                 theta_harbour: float = HARBOUR_MERAK) -> pd.DataFrame:
    """Tambahkan kolom HERI dan kategori ke DataFrame."""
    df = df.copy()

    required = ['SWH', 'MWP', 'MWD']
    if not all(c in df.columns for c in required):
        print("    [WARNING] Kolom SWH/MWP/MWD tidak lengkap, HERI dilewati.")
        return df

    # Komponen 1: SWH ternormalisasi
    swh_norm = np.clip(df['SWH'] / SWH_MAX, 0.0, 1.0)

    # Komponen 2: Steepness ternormalisasi
    mwp_safe  = df['MWP'].clip(lower=MWP_MIN)
    steepness = df['SWH'] / (G * mwp_safe**2) * (2 * np.pi)
    s_norm    = np.clip(steepness / S_MAX, 0.0, 1.0)

    # Komponen 3: Faktor arah
    theta_rel = np.radians(df['MWD'] - theta_harbour)
    f_dir     = np.abs(np.cos(theta_rel))

    # HERI
    df['HERI'] = np.clip(W1*swh_norm + W2*s_norm + W3*f_dir, 0.0, 1.0)

    # Kategori
    conditions = [
        df['HERI'] < 0.25,
        df['HERI'] < 0.50,
        df['HERI'] < 0.75,
    ]
    choices = ['AMAN', 'WASPADA', 'BERBAHAYA']
    df['HERI_cat'] = np.select(conditions, choices, default='SANGAT BERBAHAYA')

    return df


def check_gaps(df: pd.DataFrame, point_name: str):
    """
    Cek apakah ada gap dalam time series (jam yang hilang).
    ERA5 harusnya sangat lengkap, tapi tetap perlu dicek.
    """
    expected = pd.date_range(df.index.min(), df.index.max(), freq='h')
    missing  = expected.difference(df.index)

    if len(missing) == 0:
        print(f"    ✓ Tidak ada gap waktu")
    else:
        print(f"    ⚠ {len(missing)} jam hilang — mengisi dengan interpolasi linear")
        df = df.reindex(expected)
        df = df.interpolate(method='linear', limit=6)  # max gap 6 jam
        df = df.ffill().bfill()  # isi sisa dengan forward/backward fill

    return df


def print_summary(df: pd.DataFrame, point_name: str):
    """Cetak ringkasan statistik dataset."""
    print(f"\n  {'='*50}")
    print(f"  Ringkasan: {point_name}")
    print(f"  {'='*50}")
    print(f"  Periode : {df.index.min()} s/d {df.index.max()}")
    print(f"  Jumlah  : {len(df):,} baris ({len(df)/8760:.1f} tahun)")
    print(f"  Kolom   : {list(df.columns)}")

    if 'SWH' in df.columns:
        swh = df['SWH'].dropna()
        print(f"\n  SWH (m):")
        print(f"    Min    : {swh.min():.3f}")
        print(f"    Mean   : {swh.mean():.3f}")
        print(f"    Median : {swh.median():.3f}")
        print(f"    Max    : {swh.max():.3f}")
        print(f"    Std    : {swh.std():.3f}")
        print(f"    > 2.5m : {(swh >= 2.5).sum():,} jam "
              f"({(swh >= 2.5).mean()*100:.1f}%)")

    if 'HERI' in df.columns:
        print(f"\n  HERI kategori:")
        for cat in ['AMAN', 'WASPADA', 'BERBAHAYA', 'SANGAT BERBAHAYA']:
            n   = (df['HERI_cat'] == cat).sum()
            pct = n / len(df) * 100
            bar = '█' * int(pct / 2)
            print(f"    {cat:<18}: {n:>7,} jam ({pct:>5.1f}%) {bar}")

    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"\n  Missing values: {total_missing:,} total")
        for col in df.columns:
            if missing[col] > 0:
                print(f"    {col}: {missing[col]:,}")
    else:
        print(f"\n  Missing values: tidak ada ✓")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print("Menggabungkan file ERA5 .nc → CSV")
    print("=" * 55)

    # Cek file yang tersedia
    nc_wave_files    = sorted(RAW_DIR.glob("era5_wave_*.nc"))
    nc_surface_files = sorted(RAW_DIR.glob("era5_surface_*.nc"))

    print(f"\nFile ditemukan:")
    print(f"  Wave   : {len(nc_wave_files)} file")
    print(f"  Surface: {len(nc_surface_files)} file")

    if len(nc_wave_files) == 0:
        print("\n[ERROR] Tidak ada file .nc ditemukan!")
        print(f"Pastikan folder ini benar: {RAW_DIR.resolve()}")
        print("Jalankan download_era5_fixed.py terlebih dahulu.")
        return

    # Proses per titik grid
    for point_name, coords in GRID_POINTS.items():
        lat = coords['lat']
        lon = coords['lon']

        print(f"\n{'='*55}")
        print(f"Proses: {point_name} (lat={lat}, lon={lon})")
        print(f"{'='*55}")

        output_path = PROCESSED_DIR / f"era5_{point_name}_2010_2025.csv"

        # Skip kalau sudah ada
        if output_path.exists():
            print(f"  [SKIP] {output_path.name} sudah ada.")
            print(f"  Hapus file tersebut jika ingin proses ulang.")
            continue

        # ---- Kumpulkan semua bulan ----
        all_dfs = []
        total_files = len(YEARS) * len(MONTHS)
        processed = 0

        for year in YEARS:
            for month in MONTHS:
                processed += 1
                month_str = f"{month:02d}"

                # Baca wave
                wave_path = RAW_DIR / f"era5_wave_{year}_{month_str}.nc"
                df_w = pd.DataFrame()
                if wave_path.exists():
                    df_w = load_nc_point(wave_path, lat, lon, RENAME_WAVE)
                    if year == 2010 and month == 1:
                        print(f"DEBUG wave 2010-01: shape={df_w.shape}, kolom={df_w.columns.tolist()}")
                        print(df_w.head(3))
                else:
                    print(f"  [MISSING] {wave_path.name}")

                # Baca surface
                surf_path = RAW_DIR / f"era5_surface_{year}_{month_str}.nc"
                df_s = pd.DataFrame()
                if surf_path.exists():
                    df_s = load_nc_point(surf_path, lat, lon, RENAME_SURFACE)

                # Gabung wave + surface untuk bulan ini
                if not df_w.empty and not df_s.empty:
                    df_month = df_w.join(df_s, how='outer')
                elif not df_w.empty:
                    df_month = df_w
                elif not df_s.empty:
                    df_month = df_s
                else:
                    continue

                all_dfs.append(df_month)

                # Progress setiap 12 bulan
                if processed % 12 == 0:
                    pct = processed / total_files * 100
                    print(f"  Progress: {processed}/{total_files} "
                          f"bulan ({pct:.0f}%) — selesai s/d {year}-{month_str}")

        if not all_dfs:
            print(f"  [ERROR] Tidak ada data untuk {point_name}")
            continue

        # ---- Gabungkan semua bulan ----
        print(f"\n  Menggabungkan {len(all_dfs)} bulan data...", end=" ")
        df_all = pd.concat(all_dfs, axis=0)
        df_all = df_all.sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        print("OK")

        # ---- Cek dan isi gap ----
        print(f"  Cek gap waktu...", end=" ")
        df_all = check_gaps(df_all, point_name)

        # ---- Fitur turunan ----
        print(f"  Hitung fitur turunan...", end=" ")
        df_all = compute_derived_features(df_all)
        print("OK")

        # ---- HERI ----
        print(f"  Hitung HERI (Pelabuhan Merak)...", end=" ")
        df_all = compute_heri(df_all, theta_harbour=HARBOUR_MERAK)
        print("OK")

        # ---- Ringkasan ----
        print_summary(df_all, point_name)

        # ---- Simpan CSV ----
        print(f"\n  Menyimpan ke {output_path.name}...", end=" ")
        df_all.to_csv(output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")

    # ---- Selesai ----
    print(f"\n{'='*55}")
    print("SELESAI! File CSV tersimpan di:")
    print(f"  {PROCESSED_DIR.resolve()}")
    print(f"\nLangkah selanjutnya:")
    print(f"  → Jalankan 01_eda.py untuk eksplorasi data")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
