r"""
09_incidents_and_neighbors.py
=============================
(A) Tabel 16 insiden dengan HERI v2 FINAL + override (untuk Tabel Suplemen S1)
(B) Perbandingan titik 1 dengan 4 titik grid tetangga (representativitas spasial)
(C) Cek grid ERA5 dari file .nc mentah (resolusi gelombang vs atmosfer, sel darat/laut)

CARA PAKAI (dari folder DL_SelatSunda):
    .\.venv\Scripts\python.exe 09_incidents_and_neighbors.py
Output di data/revision_round2/  -> upload ke Claude:
    incident_table.csv, neighbor_comparison.csv, neighbor_incidents.csv, grid_check.txt

CARA MENGUBAH:
    - Koordinat pelabuhan (untuk jarak): PORTS (perkiraan; ganti dari Selat_sunda.kml)
    - Daftar insiden & metadata: INCIDENTS (kolom sumber bisa dilengkapi URL)
    - Jendela insiden: 00:00 WIB tanggal mulai s/d 23:00 WIB tanggal selesai (WINDOW_*)

ASUMSI:
    - Kolom waktu CSV ERA5 = UTC; WIB = UTC + 7 jam.
    - HERI v2 + override identik dengan heri_v2_FINAL.py / heri_override.py.
    - "Kalibrasi" = 6 insiden bertag 'merak' (2020–2024) yang dipakai menetapkan ambang 0.50.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ═════════ KONFIGURASI ═════════
PROC = Path("data/era5_processed")
RAW = Path("data/era5_raw")
OUT = Path("data/revision_round2")
POINTS = {  # dari mergeera5_v2.py (GRID_POINTS)
    "titik1_tengah_selat":    (-6.0, 105.5),
    "titik2_lampung_timur":   (-5.5, 106.0),
    "titik3_teluk_semangka":  (-6.0, 105.0),
    "titik4_pintu_samudera":  (-6.5, 105.0),
    "titik5_pandeglang":      (-6.5, 105.5),
}
PORTS = {"Merak": (-5.93, 106.00), "Bakauheni": (-5.87, 105.75)}  # PERKIRAAN, cek KML

SWH_MAX, T_N, BETA, FLOOR, W_H, W_D = 2.5, 5.5, 0.30, 0.30, 0.70, 0.30
THETA = {"merak": 243.8, "bakauheni": 130.8}
CAT_BOUNDS = [0.382, 0.50, 0.727]
CAT_LABELS = ["SAFE", "CAUTION", "DANGEROUS", "VERY DANGEROUS"]
SWH_LIMIT, WS10_LIMIT = 2.5, 21.0 * 0.514444

# no, mulai, selesai (WIB), lokasi, kapal/jenis, mekanisme menurut sumber, sumber, keyakinan
INCIDENTS = [
    (1, "2015-12-19", "2015-12-21", "unknown", "multiple vessels; 3-km truck queue",
     "berthing difficulty, bad weather", "Suara.com 19 Dec 2015 (opened by author)", "LOW"),
    (2, "2017-11-30", "2017-11-30", "route_wide", "full closure (TC Dahlia)",
     "wave ~5 m, wind >45 kn (ASDP)", "Tribunnews 30 Nov 2017; Wartakota 1 Dec 2017", "HIGH"),
    (3, "2018-05-23", "2018-05-23", "bakauheni", "KMP Titian Nusantara grounding",
     "grounded ~2 nm off Bakauheni (P. Kandangluni)", "Selatsunda.com; Sindonews 24 May 2018", "MEDIUM"),
    (4, "2019-06-07", "2019-06-07", "bakauheni", "KMP Mutiara Persada II grounding",
     "grounded entering Bakauheni channel (P. Rimau Balak)", "CNN Indonesia; Antara 904939", "HIGH"),
    (5, "2020-12-20", "2020-12-20", "merak", "KMP Safira Nusantara grounding at berth approach",
     "strong current + low tide + buoy light failure (Dermaga IV)", "Titiknol; Beritatrans 203354", "MEDIUM"),
    (6, "2021-03-08", "2021-03-08", "unknown", "departure delay 15-20 min",
     "manoeuvring difficulty, waves ~1 m", "Ujaran.co (opened by author; citizen journalism)", "LOW"),
    (7, "2021-11-28", "2021-11-28", "merak", "KMP Jatra 3 unable to hold berth (Dermaga 6)",
     "bad weather; trucks toppled on KMP Nusa Dharma", "Liputan6 4722911, 4722918", "MEDIUM"),
    (8, "2021-12-06", "2021-12-07", "merak", "KMP Batu Mandi 8 h unable to berth; ramp door broken",
     "waves ~2.5 m at Dermaga Eksekutif", "Sindonews 620713; Kompas TV", "MEDIUM"),
    (9, "2022-02-09", "2022-02-10", "merak", "KMP Nusa Dharma drifted, struck KMP Panorama",
     "bad weather, waves 1.25-2.5 m", "Bantennews; Detakbanten", "MEDIUM"),
    (10, "2022-12-22", "2022-12-22", "route_wide", "departures postponed from 20:40 WIB (route-wide)",
     "bad weather; berthing difficulty at Merak", "Bisnis.com 1611412 (ASDP); Sindonews 976763; PMJ News 50011", "HIGH"),
    (11, "2022-12-28", "2022-12-31", "route_wide", "route closed ~6 h",
     "waves 1.25-2.5 m, wind to 25 kn", "Kompas.id; Liputan6 5166265", "HIGH"),
    (12, "2024-03-07", "2024-03-08", "merak", "Dermaga 4 & 7 closed; berthing difficulty",
     "current 1.5-2 m/s dominant; wind 20-25 kn", "Detik; Antara Banten", "MEDIUM"),
    (13, "2024-03-08", "2024-03-09", "merak", "4-km vehicle queue (berthing delays)",
     "waves ~2.5 m, wind 20-25 kn", "Detik; Antara", "MEDIUM"),
    (14, "2024-03-15", "2024-03-16", "route_wide", "all ferries suspended ~27 h",
     "wind 22-30 kn SW, waves 2.5 m", "ASDP press release", "HIGH"),
    (15, "2024-12-03", "2024-12-03", "route_wide", "Merak and Bakauheni suspended",
     "wind 17-32 kn, waves 1.5-3.0 m (BMKG)", "CNN Indonesia; Antara 4508353", "HIGH"),
    (16, "2024-12-07", "2024-12-07", "bakauheni", "KMP Jagantara grounding (P. Kandang Lunik)",
     "rain, poor visibility, current", "Antara 4518631; Kompas TV; KSOP Bakauheni", "HIGH"),
]
# ═══════════════════════════════


def heri(swh, mwp, mwd, theta):
    swh_norm = np.clip(swh / SWH_MAX, 0, 1)
    R = FLOOR + (1 - FLOOR) * np.exp(-(((mwp - T_N) / (BETA * T_N)) ** 2))
    f_dir = np.maximum(0, np.cos(np.radians(mwd - theta)))
    return np.clip(W_H * swh_norm * R + W_D * f_dir, 0, 1)


def level(df, theta):
    lvl = np.digitize(heri(df.SWH.values, df.MWP.values, df.MWD.values, theta), CAT_BOUNDS)
    return np.where((df.SWH.values > SWH_LIMIT) | (df.WS10.values > WS10_LIMIT), 3, lvl)


def load(name):
    df = pd.read_csv(PROC / f"era5_{name}_2010_2025.csv", index_col='time', parse_dates=True)
    df = df.loc[:"2024-12-31 23:00"]
    if 'WS10' not in df.columns:
        df['WS10'] = np.hypot(df.U10, df.V10)
    return df


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (*a, *b))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def incident_rows(df, point):
    wib = df.copy()
    wib.index = wib.index + pd.Timedelta(hours=7)
    rows = []
    for no, d0, d1, loc, what, mech, src, conf in INCIDENTS:
        w = wib.loc[f"{d0} 00:00":f"{d1} 23:00"]
        port = 'bakauheni' if loc == 'bakauheni' else 'merak'
        h = heri(w.SWH.values, w.MWP.values, w.MWD.values, THETA[port])
        lv = level(w, THETA[port])
        k = int(np.argmax(h))
        exceed = np.where(lv >= 2)[0]
        rows.append({
            'point': point, 'no': no, 'start_WIB': f"{d0} 00:00", 'end_WIB': f"{d1} 23:00",
            'start_UTC': str(pd.Timestamp(f"{d0} 00:00") - pd.Timedelta(hours=7)),
            'location': loc, 'event': what, 'reported_mechanism': mech, 'sources': src,
            'confidence': conf, 'calibration_set': loc == 'merak',
            'port_theta_used': port, 'n_hours': len(w),
            'SWH_max_m': round(w.SWH.max(), 3), 'WS10_max_kn': round(w.WS10.max() / 0.514444, 1),
            'HERI_max': round(h.max(), 3), 'HERI_mean': round(h.mean(), 3),
            'time_HERI_max_WIB': str(w.index[k]),
            'SWH_at_max': round(w.SWH.iloc[k], 3), 'MWP_at_max_s': round(w.MWP.iloc[k], 2),
            'MWD_at_max_deg': round(w.MWD.iloc[k], 1),
            'hours_DANGEROUS_plus': int((lv >= 2).sum()),
            'first_DANGEROUS_WIB': str(w.index[exceed[0]]) if len(exceed) else '',
            'max_category': CAT_LABELS[int(lv.max())],
            'detected_HERI_only': bool((h >= 0.50).any()),          # tanpa override
            'detected_HERI': bool((lv >= 2).any()),                 # dengan override
            'detected_SWH_2p5': bool((w.SWH > 2.5).any()),
            'margin_vs_0p50': round(h.max() - 0.50, 3),
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {p: load(p) for p in POINTS}
    ref = data["titik1_tengah_selat"]
    ref_lv = level(ref, THETA['merak'])

    # (A) tabel insiden di titik 1
    inc = pd.DataFrame(incident_rows(ref, "titik1_tengah_selat"))
    inc.to_csv(OUT / 'incident_table.csv', index=False)

    # (B) perbandingan titik tetangga
    comp, ninc = [], []
    for p, df in data.items():
        df = df.reindex(ref.index)
        lv = level(df, THETA['merak'])
        a, b = ref_lv >= 2, lv >= 2
        po = (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        agree = np.mean(a == b)
        row = {'point': p, 'lat': POINTS[p][0], 'lon': POINTS[p][1],
               **{f'dist_{k}_km': round(haversine(POINTS[p], v), 1) for k, v in PORTS.items()},
               'n_valid_hours': int(df.SWH.notna().sum()),
               'SWH_mean': round(df.SWH.mean(), 3), 'SWH_p99': round(df.SWH.quantile(.99), 3),
               'SWH_max': round(df.SWH.max(), 3), 'MWP_mean': round(df.MWP.mean(), 2),
               'MWD_circmean': round(np.degrees(np.arctan2(np.sin(np.radians(df.MWD)).mean(),
                                                           np.cos(np.radians(df.MWD)).mean())) % 360, 1),
               'WS10_mean': round(df.WS10.mean(), 2),
               'r_SWH_vs_titik1': round(df.SWH.corr(ref.SWH), 3),
               'pct_DANGEROUS_plus_merak': round(100 * b.mean(), 2),
               'pct_hours_SWH_gt_2p5': round(100 * (df.SWH > 2.5).mean(), 3),
               'agreement_DANGEROUS_plus_vs_titik1': round(agree, 3),
               'kappa_vs_titik1': round((agree - po) / (1 - po), 3) if po < 1 else np.nan}
        comp.append(row)
        r = pd.DataFrame(incident_rows(df.dropna(subset=['SWH']), p))
        ninc.append(r[['point', 'no', 'location', 'calibration_set', 'SWH_max_m',
                       'HERI_max', 'detected_HERI_only', 'detected_HERI', 'detected_SWH_2p5']])
    pd.DataFrame(comp).to_csv(OUT / 'neighbor_comparison.csv', index=False)
    pd.concat(ninc).to_csv(OUT / 'neighbor_incidents.csv', index=False)

    # (C) cek grid ERA5 mentah
    lines = []
    try:
        import xarray as xr
        for kind in ['wave', 'surface']:
            f = sorted(RAW.glob(f"era5_{kind}_*.nc"))
            if not f:
                lines.append(f"{kind}: file .nc tidak ditemukan di {RAW}")
                continue
            ds = xr.open_dataset(f[0])
            lat, lon = ds.latitude.values, ds.longitude.values
            lines.append(f"{kind}: file={f[0].name} | lat {lat.min()}..{lat.max()} "
                         f"step={np.round(np.diff(lat)[0], 4)} | lon {lon.min()}..{lon.max()} "
                         f"step={np.round(np.diff(lon)[0], 4)} | vars={list(ds.data_vars)}")
            if kind == 'wave':
                v = list(ds.data_vars)[0]
                first = ds[v].isel({d: 0 for d in ds[v].dims if d not in ('latitude', 'longitude')})
                for la in lat:
                    lines.append("  lat %6.2f: " % la + " ".join(
                        "%7.2f:%s" % (lo, 'sea' if np.isfinite(float(first.sel(latitude=la, longitude=lo)))
                                      else 'LAND') for lo in lon))
            ds.close()
    except Exception as e:
        lines.append(f"Cek grid gagal: {e}")
    (OUT / 'grid_check.txt').write_text("\n".join(lines), encoding='utf-8')

    pd.set_option('display.width', 250)
    print(inc[['no', 'location', 'calibration_set', 'SWH_max_m', 'HERI_max', 'max_category',
               'hours_DANGEROUS_plus', 'detected_HERI_only', 'detected_HERI']].to_string(index=False))
    print(pd.DataFrame(comp).to_string(index=False))
    print("\n".join(lines))
    print(f"\nUpload: {OUT}\\incident_table.csv, neighbor_comparison.csv, "
          f"neighbor_incidents.csv, grid_check.txt")


if __name__ == "__main__":
    main()
