"""
============================================================
Script Download Data ERA5 untuk Prediksi Gelombang Selat Sunda
============================================================
Studi: Prediksi Tinggi Gelombang (SWH) Selat Sunda Berbasis Deep Learning
       untuk Mendukung Keselamatan Penyeberangan Merak-Bakauheni

Data  : ERA5 Reanalysis (ECMWF/Copernicus)
Variabel yang diunduh:
  - swh   : Significant Wave Height (target/output)
  - u10   : 10m U-component of wind
  - v10   : 10m V-component of wind
  - mwp   : Mean Wave Period
  - mwd   : Mean Wave Direction
  - msl   : Mean Sea Level Pressure

Area  : Selat Sunda (Lat: -7.0 s/d -5.5 | Lon: 104.5 s/d 106.5)
Periode: 2010 - 2023 (training & testing)
Resolusi: grid asli ERA5 (gelombang 0.5°, atmosfer 0.25°) — jangan isi parameter "grid"; hourly

============================================================
PERSIAPAN SEBELUM MENJALANKAN SCRIPT INI:
============================================================
1. Daftar akun gratis di https://cds.climate.copernicus.eu
2. Setelah login, buka: https://cds.climate.copernicus.eu/how-to-api
3. Salin API key Anda, lalu buat file ~/.cdsapirc berisi:
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_API_KEY_HERE
4. Install library yang diperlukan:
   pip install cdsapi netCDF4 xarray pandas numpy matplotlib seaborn

============================================================
"""

import cdsapi
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI UTAMA — sesuaikan di sini jika perlu
# ============================================================

# Direktori output
OUTPUT_DIR = Path("data/era5_raw")
PROCESSED_DIR = Path("data/era5_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Area studi: Selat Sunda
# Format: [North, West, South, East]
AREA = [-5.5, 104.5, -7.0, 106.5]

# Periode data
YEARS_TRAIN = list(range(2010, 2022))   # 2010–2021 untuk training
YEARS_TEST  = [2022, 2023]              # 2022–2023 untuk testing

# Titik grid representatif Selat Sunda (untuk ekstraksi time series)
# Titik 1: Tengah Selat Sunda (dekat jalur feri Merak-Bakauheni)
# Titik 2: Mulut selat sisi Samudra Hindia
# Titik 3: Bagian utara dekat Pelabuhan Merak
GRID_POINTS = {
    "titik1_tengah_selat":  {"lat": -6.0,  "lon": 105.75},
    "titik2_selatan_samudera": {"lat": -6.75, "lon": 105.25},
    "titik3_utara_merak":   {"lat": -5.9,  "lon": 106.0},
}

# Variabel ERA5 yang diunduh
VARIABLES_WAVE = [
    'significant_height_of_combined_wind_waves_and_swell',  # SWH (target)
    'mean_wave_period',                                      # MWP
    'mean_wave_direction',                                   # MWD
    'peak_wave_period',                                      # PWP
]

VARIABLES_SURFACE = [
    '10m_u_component_of_wind',   # U10
    '10m_v_component_of_wind',   # V10
    'mean_sea_level_pressure',   # MSLP
    'sea_surface_temperature',   # SST (opsional, bisa tambah insight)
]

# Jam pengambilan data (hourly = semua jam)
HOURS = [f"{h:02d}:00" for h in range(0, 24)]

# ============================================================
# FUNGSI DOWNLOAD
# ============================================================

def download_era5_wave(year: int, output_path: Path):
    """
    Download variabel gelombang ERA5 untuk satu tahun.
    Dataset: reanalysis-era5-single-levels
    """
    c = cdsapi.Client()
    print(f"  [WAVE] Mengunduh tahun {year}...")
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': VARIABLES_WAVE,
            'year': str(year),
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': HOURS,
            'area': AREA,
            'format': 'netcdf',
        },
        str(output_path)
    )
    print(f"  [WAVE] Selesai: {output_path}")


def download_era5_surface(year: int, output_path: Path):
    """
    Download variabel permukaan ERA5 (angin, tekanan) untuk satu tahun.
    """
    c = cdsapi.Client()
    print(f"  [SURFACE] Mengunduh tahun {year}...")
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': VARIABLES_SURFACE,
            'year': str(year),
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': HOURS,
            'area': AREA,
            'format': 'netcdf',
        },
        str(output_path)
    )
    print(f"  [SURFACE] Selesai: {output_path}")


# ============================================================
# FUNGSI PENGOLAHAN DATA
# ============================================================

def extract_timeseries(nc_file: Path, lat: float, lon: float) -> pd.DataFrame:
    """
    Ekstrak time series dari file NetCDF pada titik grid terdekat.
    """
    ds = xr.open_dataset(nc_file)
    
    # Pilih titik terdekat
    ds_point = ds.sel(
        latitude=lat,
        longitude=lon,
        method='nearest'
    )
    
    df = ds_point.to_dataframe().reset_index()
    df = df.set_index('valid_time') if 'valid_time' in df.columns else df.set_index('time')
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=['latitude', 'longitude'], errors='ignore')
    ds.close()
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarisasi nama kolom ERA5 menjadi nama yang lebih mudah dibaca.
    """
    rename_map = {
        'swh': 'SWH',
        'mwp': 'MWP',
        'mwd': 'MWD',
        'pp1d': 'PWP',
        'u10': 'U10',
        'v10': 'V10',
        'msl': 'MSLP',
        'sst': 'SST',
        # nama alternatif yang mungkin muncul
        'significant_height_of_combined_wind_waves_and_swell': 'SWH',
        'mean_wave_period': 'MWP',
        'mean_wave_direction': 'MWD',
        '10m_u_component_of_wind': 'U10',
        '10m_v_component_of_wind': 'V10',
        'mean_sea_level_pressure': 'MSLP',
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def compute_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung kecepatan angin dari komponen U dan V."""
    if 'U10' in df.columns and 'V10' in df.columns:
        df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
        df['WD10'] = (np.degrees(np.arctan2(df['U10'], df['V10'])) + 360) % 360
    return df


def check_data_quality(df: pd.DataFrame, point_name: str):
    """Cek kualitas data: missing values, outlier, statistik dasar."""
    print(f"\n{'='*55}")
    print(f"Laporan Kualitas Data — {point_name}")
    print(f"{'='*55}")
    print(f"Periode : {df.index.min()} s/d {df.index.max()}")
    print(f"Jumlah  : {len(df):,} baris")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        print(f"  {col:<8}: {missing[col]:>6} ({pct[col]:.2f}%)")
    
    if 'SWH' in df.columns:
        print(f"\nStatistik SWH (Significant Wave Height):")
        print(f"  Min    : {df['SWH'].min():.3f} m")
        print(f"  Max    : {df['SWH'].max():.3f} m")
        print(f"  Mean   : {df['SWH'].mean():.3f} m")
        print(f"  Median : {df['SWH'].median():.3f} m")
        print(f"  Std    : {df['SWH'].std():.3f} m")
        print(f"\nDistribusi SWH berdasarkan kategori BMKG:")
        kategori = [
            ("Tenang",   0.0,  0.5),
            ("Rendah",   0.5,  1.25),
            ("Sedang",   1.25, 2.5),
            ("Tinggi",   2.5,  4.0),
            ("Sangat Tinggi", 4.0, 99),
        ]
        for nama, lo, hi in kategori:
            n = ((df['SWH'] >= lo) & (df['SWH'] < hi)).sum()
            pct_kat = n / len(df) * 100
            print(f"  {nama:<18}: {n:>7,} ({pct_kat:.1f}%)")


def plot_overview(df: pd.DataFrame, point_name: str, save_dir: Path):
    """
    Buat 4 plot ringkasan: time series, distribusi, seasonal, scatter angin-gelombang.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Overview Data ERA5 — {point_name}\nSelat Sunda (2010–2023)',
                 fontsize=14, fontweight='bold')

    # 1. Time series SWH
    ax = axes[0, 0]
    monthly = df['SWH'].resample('ME').mean()
    ax.plot(monthly.index, monthly.values, color='steelblue', linewidth=1)
    ax.axhline(2.5, color='red', linestyle='--', alpha=0.7, label='Batas aman feri (2.5m)')
    ax.set_title('Time Series SWH Bulanan (Rata-rata)')
    ax.set_ylabel('SWH (m)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 2. Distribusi SWH
    ax = axes[0, 1]
    ax.hist(df['SWH'].dropna(), bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(2.5, color='red', linestyle='--', label='Batas aman (2.5m)')
    ax.axvline(df['SWH'].mean(), color='orange', linestyle='--',
               label=f"Mean ({df['SWH'].mean():.2f}m)")
    ax.set_title('Distribusi SWH')
    ax.set_xlabel('SWH (m)')
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)

    # 3. SWH rata-rata per bulan (seasonal pattern)
    ax = axes[1, 0]
    nama_bulan = ['Jan','Feb','Mar','Apr','Mei','Jun',
                  'Jul','Ags','Sep','Okt','Nov','Des']
    monthly_mean = df['SWH'].groupby(df.index.month).mean()
    monthly_std  = df['SWH'].groupby(df.index.month).std()
    bars = ax.bar(range(1, 13), monthly_mean.values, color='steelblue',
                  alpha=0.8, yerr=monthly_std.values, capsize=4)
    ax.axhline(2.5, color='red', linestyle='--', alpha=0.7, label='Batas aman (2.5m)')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(nama_bulan, fontsize=9)
    ax.set_title('Pola Musiman SWH (Rata-rata per Bulan, 2010–2023)')
    ax.set_ylabel('SWH (m)')
    ax.legend(fontsize=8)

    # 4. Scatter: kecepatan angin vs SWH
    ax = axes[1, 1]
    if 'WS10' in df.columns:
        sample = df[['WS10', 'SWH']].dropna().sample(min(5000, len(df)))
        sc = ax.scatter(sample['WS10'], sample['SWH'],
                        alpha=0.2, s=5, c=sample['SWH'],
                        cmap='YlOrRd')
        plt.colorbar(sc, ax=ax, label='SWH (m)')
        ax.set_xlabel('Kecepatan Angin 10m (m/s)')
        ax.set_ylabel('SWH (m)')
        ax.set_title('Hubungan Kecepatan Angin vs SWH')
    else:
        ax.text(0.5, 0.5, 'Data angin tidak tersedia',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    save_path = save_dir / f"overview_{point_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot disimpan: {save_path}")


def plot_mudik_analysis(df: pd.DataFrame, point_name: str, save_dir: Path):
    """
    Analisis khusus kondisi gelombang selama periode mudik Lebaran.
    Estimasi jadwal Lebaran (H-7 s/d H+7):
      2015: 17 Jul | 2016: 6 Jul | 2017: 26 Jun | 2018: 15 Jun
      2019: 5 Jun  | 2020: 24 Mei| 2021: 13 Mei | 2022: 2 Mei
      2023: 21 Apr
    """
    lebaran_dates = {
        2015: '2015-07-17', 2016: '2016-07-06', 2017: '2017-06-26',
        2018: '2018-06-15', 2019: '2019-06-05', 2020: '2020-05-24',
        2021: '2021-05-13', 2022: '2022-05-02', 2023: '2023-04-21',
    }

    mudik_stats = []
    for year, date_str in lebaran_dates.items():
        try:
            center = pd.Timestamp(date_str)
            start  = center - pd.Timedelta(days=7)
            end    = center + pd.Timedelta(days=7)
            mask   = (df.index >= start) & (df.index <= end)
            subset = df.loc[mask, 'SWH'].dropna()
            if len(subset) > 0:
                pct_berbahaya = (subset >= 2.5).mean() * 100
                mudik_stats.append({
                    'Tahun': year,
                    'Tanggal Lebaran': date_str,
                    'Mean SWH (m)': round(subset.mean(), 3),
                    'Max SWH (m)': round(subset.max(), 3),
                    'Std SWH (m)': round(subset.std(), 3),
                    '% Jam > 2.5m': round(pct_berbahaya, 1),
                })
        except Exception:
            continue

    if not mudik_stats:
        print("  Data mudik tidak tersedia dalam rentang dataset.")
        return

    df_mudik = pd.DataFrame(mudik_stats)
    print(f"\n{'='*55}")
    print(f"Analisis Kondisi Gelombang Selama Mudik Lebaran")
    print(f"Lokasi: {point_name}")
    print(f"{'='*55}")
    print(df_mudik.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Kondisi Gelombang Saat Mudik Lebaran — {point_name}',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    colors = ['red' if v > 10 else 'steelblue' for v in df_mudik['% Jam > 2.5m']]
    ax.bar(df_mudik['Tahun'], df_mudik['% Jam > 2.5m'], color=colors, alpha=0.85)
    ax.set_title('% Jam Gelombang > 2.5m (H-7 s/d H+7 Lebaran)')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Persentase Jam (%)')
    ax.set_xticks(df_mudik['Tahun'])
    ax.tick_params(axis='x', rotation=45)
    for i, (_, row) in enumerate(df_mudik.iterrows()):
        ax.text(row['Tahun'], row['% Jam > 2.5m'] + 0.3,
                f"{row['% Jam > 2.5m']:.1f}%", ha='center', fontsize=8)

    ax = axes[1]
    ax.bar(df_mudik['Tahun'], df_mudik['Mean SWH (m)'],
           alpha=0.7, color='steelblue', label='Rata-rata SWH')
    ax.plot(df_mudik['Tahun'], df_mudik['Max SWH (m)'],
            'ro--', markersize=6, label='Maks SWH')
    ax.axhline(2.5, color='red', linestyle='--', alpha=0.5, label='Batas aman (2.5m)')
    ax.set_title('Rata-rata & Maks SWH Saat Mudik')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('SWH (m)')
    ax.legend(fontsize=8)
    ax.set_xticks(df_mudik['Tahun'])
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = save_dir / f"mudik_analysis_{point_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot mudik disimpan: {save_path}")

    # Simpan tabel
    csv_path = save_dir / f"mudik_stats_{point_name}.csv"
    df_mudik.to_csv(csv_path, index=False)
    print(f"  Tabel disimpan: {csv_path}")


# ============================================================
# FUNGSI UTAMA
# ============================================================

def main():
    print("=" * 60)
    print("Download & Preprocessing ERA5 — Selat Sunda")
    print("=" * 60)

    all_years = YEARS_TRAIN + YEARS_TEST

    # ---- STEP 1: Download data ----
    print("\n[STEP 1] Memulai download ERA5...")
    for year in all_years:
        wave_path   = OUTPUT_DIR / f"era5_wave_{year}.nc"
        surface_path = OUTPUT_DIR / f"era5_surface_{year}.nc"

        if not wave_path.exists():
            download_era5_wave(year, wave_path)
        else:
            print(f"  [SKIP] {wave_path.name} sudah ada.")

        if not surface_path.exists():
            download_era5_surface(year, surface_path)
        else:
            print(f"  [SKIP] {surface_path.name} sudah ada.")

    # ---- STEP 2: Gabungkan file per titik grid ----
    print("\n[STEP 2] Mengekstrak time series per titik grid...")
    for point_name, coords in GRID_POINTS.items():
        lat, lon = coords['lat'], coords['lon']
        print(f"\n  Titik: {point_name} (lat={lat}, lon={lon})")
        
        dfs_all = []
        for year in all_years:
            wave_path    = OUTPUT_DIR / f"era5_wave_{year}.nc"
            surface_path = OUTPUT_DIR / f"era5_surface_{year}.nc"
            
            if wave_path.exists():
                df_w = extract_timeseries(wave_path, lat, lon)
                df_w = rename_columns(df_w)
                dfs_all.append(df_w)
            
            if surface_path.exists():
                df_s = extract_timeseries(surface_path, lat, lon)
                df_s = rename_columns(df_s)
                # Gabung jika ada data gelombang di tahun yang sama
                if dfs_all:
                    last_df = dfs_all[-1]
                    shared_idx = last_df.index.intersection(df_s.index)
                    for col in df_s.columns:
                        if col not in last_df.columns:
                            last_df.loc[shared_idx, col] = df_s.loc[shared_idx, col]

        if not dfs_all:
            print(f"  [WARNING] Tidak ada data untuk {point_name}. Lewati.")
            continue

        df_combined = pd.concat(dfs_all).sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        df_combined = compute_wind_speed(df_combined)

        # ---- STEP 3: Cek kualitas & simpan ----
        check_data_quality(df_combined, point_name)

        # Simpan CSV
        csv_path = PROCESSED_DIR / f"era5_{point_name}_2010_2023.csv"
        df_combined.to_csv(csv_path)
        print(f"  Data disimpan: {csv_path}")

        # ---- STEP 4: Plot overview & analisis mudik ----
        print(f"\n[STEP 4] Membuat visualisasi untuk {point_name}...")
        if 'SWH' in df_combined.columns:
            plot_overview(df_combined, point_name, PROCESSED_DIR)
            plot_mudik_analysis(df_combined, point_name, PROCESSED_DIR)

    print("\n" + "=" * 60)
    print("SELESAI! Semua file tersimpan di folder:")
    print(f"  Raw    : {OUTPUT_DIR.resolve()}")
    print(f"  Proses : {PROCESSED_DIR.resolve()}")
    print("=" * 60)
    print("\nLangkah selanjutnya:")
    print("  1. Cek folder data/era5_processed/ untuk file CSV")
    print("  2. Buka notebook 01_eda.ipynb untuk eksplorasi data")
    print("  3. Lanjut ke 02_preprocessing.ipynb untuk sliding window")


if __name__ == "__main__":
    main()
