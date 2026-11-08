r"""
10_base_rate.py
===============
Cek "deteksi kebetulan": seberapa sering HERI Merak (v2 + override) mencapai DANGEROUS+
pada hari-hari BIASA, dibanding pada hari insiden. Menjawab R-A soal spesifisitas:
kalau hampir semua hari Desember sudah DANGEROUS, deteksi 6/6 tidak bermakna.

Untuk tiap insiden (jendela L hari, bulan m): base rate = proporsi semua jendela L hari
yang dimulai pada bulan m (2010–2024, di luar tanggal insiden) yang punya ≥1 jam DANGEROUS+.
Peluang mendeteksi semua insiden secara kebetulan = perkalian base rate (asumsi independen).

CARA PAKAI (dari folder DL_SelatSunda):
    .\.venv\Scripts\python.exe 10_base_rate.py
Output: data/revision_round2/base_rate.csv  -> upload ke Claude

Juga membandingkan HERI dengan aturan tinggi-saja yang dikalibrasi lokal (diminta R-A):
    SWH > P90 / P95 lokal (persentil 2010–2024 di titik 1).

CARA MENGUBAH:
    - Daftar insiden: INCIDENTS (no, mulai WIB, selesai WIB, kalibrasi?)
    - Aturan pembanding: RULE_PCTS (persentil SWH)
"""

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv")
OUT = Path("data/revision_round2")
SWH_MAX, T_N, BETA, FLOOR, W_H, W_D, THETA = 2.5, 5.5, 0.30, 0.30, 0.70, 0.30, 243.8
SWH_LIMIT, WS10_LIMIT = 2.5, 21.0 * 0.514444
RULE_PCTS = [90, 95]
INCIDENTS = [  # no, mulai, selesai (WIB), set
    (5, "2020-12-20", "2020-12-20", "calibration"), (7, "2021-11-28", "2021-11-28", "calibration"),
    (8, "2021-12-06", "2021-12-07", "calibration"), (9, "2022-02-09", "2022-02-10", "calibration"),
    (12, "2024-03-07", "2024-03-08", "calibration"), (13, "2024-03-08", "2024-03-09", "calibration"),
    (1, "2015-12-19", "2015-12-21", "other"), (2, "2017-11-30", "2017-11-30", "other"),
    (6, "2021-03-08", "2021-03-08", "other"), (10, "2022-12-22", "2022-12-22", "other"), (11, "2022-12-28", "2022-12-31", "other"),
    (14, "2024-03-15", "2024-03-16", "other"), (15, "2024-12-03", "2024-12-03", "other"),
]


def main():
    df = pd.read_csv(CSV, index_col='time', parse_dates=True).loc[:"2024-12-31 23:00"]
    if 'WS10' not in df.columns:
        df['WS10'] = np.hypot(df.U10, df.V10)
    df.index = df.index + pd.Timedelta(hours=7)                       # UTC -> WIB
    R = FLOOR + (1 - FLOOR) * np.exp(-(((df.MWP - T_N) / (BETA * T_N)) ** 2))
    h = np.clip(W_H * np.clip(df.SWH / SWH_MAX, 0, 1) * R
                + W_D * np.maximum(0, np.cos(np.radians(df.MWD - THETA))), 0, 1)
    rules = {'HERI': (h >= 0.50) | (df.SWH > SWH_LIMIT) | (df.WS10 > WS10_LIMIT)}
    for q in RULE_PCTS:
        thr = df.SWH.quantile(q / 100)
        rules[f'SWH>P{q} ({thr:.2f} m)'] = df.SWH > thr
    all_rows, all_monthly = [], []
    for rname, danger in rules.items():
        r, m = evaluate(danger)
        all_rows.append(r.assign(rule=rname)); all_monthly.append(m.rename(rname))
    res = pd.concat(all_rows)
    summ = res.groupby(['rule', 'set']).agg(detected=('detected', 'sum'), n=('detected', 'size'),
                                            mean_base_rate=('base_rate', 'mean'),
                                            p_all_by_chance=('base_rate', 'prod')).round(4)
    monthly = pd.concat(all_monthly, axis=1).round(3)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / 'base_rate.csv', index=False)
    summ.to_csv(OUT / 'base_rate_summary.csv')
    monthly.to_csv(OUT / 'base_rate_monthly.csv')
    pd.set_option('display.width', 200)
    print(summ.to_string()); print(monthly.to_string())
    print(f"\nUpload: {OUT}\\base_rate_summary.csv, base_rate.csv, base_rate_monthly.csv")


def evaluate(danger):
    daily = danger.groupby(danger.index.normalize()).any()             # hari dengan ≥1 jam

    inc_days = set()
    for _, a, b, _ in INCIDENTS:
        inc_days |= set(pd.date_range(a, b, freq='D'))
    rows = []
    for no, a, b, s in INCIDENTS:
        L = len(pd.date_range(a, b, freq='D'))
        m = pd.Timestamp(a).month
        hits = []
        for d in daily.index[daily.index.month == m]:
            win = pd.date_range(d, periods=L, freq='D')
            if win[-1] > daily.index[-1] or any(x in inc_days for x in win):
                continue
            hits.append(daily.reindex(win).fillna(False).any())
        rows.append({'no': no, 'set': s, 'month': m, 'window_days': L,
                     'base_rate': round(float(np.mean(hits)), 3), 'n_windows': len(hits),
                     'detected': bool(daily.reindex(pd.date_range(a, b, freq='D')).any())})
    return pd.DataFrame(rows), daily.groupby(daily.index.month).mean()

if __name__ == "__main__":
    main()
