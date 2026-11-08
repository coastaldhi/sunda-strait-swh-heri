r"""
12_param_sensitivity.py
=======================
Sensitivitas parameter HERI (diminta R1 #5/#11, R3 2.5, R4 #3). Tanpa training, hanya pandas.

Untuk tiap konfigurasi (ubah SATU parameter dari baseline, sisanya tetap):
    Tn, beta, Rmin, bobot Wh/Wd, normalisasi SWH, dan dua ablation:
      - no_period    : R = 1 (period weighting dimatikan)  -> aturan tinggi + arah
      - no_direction : Wh = 1, Wd = 0                      -> aturan tinggi + periode
dihitung: deteksi 6 insiden kalibrasi & 6 insiden lain, margin terkecil di atas ambang,
base rate & P(chance), dan % hari ter-flag per musim (Des–Jan vs Mei–Okt).

Dua mode ambang DANGEROUS:
    fixed   : ambang tetap 0.50 (seperti di naskah)
    matched : ambang diatur supaya % jam DANGEROUS+ sama dengan baseline
              -> perbandingan adil: "dengan jumlah peringatan sama, siapa lebih tepat?"
Override (SWH > 2.5 m atau WS10 > 21 kn -> VD) selalu dipakai.

Tabel kedua: sensitivitas batas persentil P75 (SAFE/CAUTION) dan P99.9 (DANGEROUS/VD).

CARA PAKAI (dari folder DL_SelatSunda):
    .\.venv\Scripts\python.exe 12_param_sensitivity.py
Output (upload ke Claude):
    data/revision_round2/param_sensitivity.csv
    data/revision_round2/param_sensitivity_percentiles.csv

CARA MENGUBAH:
    - Nilai yang diuji: GRID (nama parameter -> daftar nilai)
    - Baseline: BASE
    - Insiden: INCIDENTS (sama dengan 10_base_rate.py); UNVERIFIED = insiden tanpa sumber
      yang berhasil dibuka (dihitung terpisah di kolom other_det_verified)
"""

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv")
OUT = Path("data/revision_round2")
BASE = dict(Tn=5.5, beta=0.30, Rmin=0.30, Wh=0.70, Wd=0.30, Smax=2.5, theta=243.8)
GRID = {
    'Tn':    [4.5, 5.0, 6.0, 6.5, 7.0, 8.0],
    'beta':  [0.20, 0.40, 0.50],
    'Rmin':  [0.10, 0.50],
    'Wh/Wd': [(0.6, 0.4), (0.8, 0.2), (0.9, 0.1)],
    'Smax':  [2.0, 3.0],
    'no_period':    [dict(Rmin=1.0)],
    'no_direction': [dict(Wh=1.0, Wd=0.0)],
}
THR_FIXED = 0.50
SWH_LIMIT, WS10_LIMIT = 2.5, 21.0 * 0.514444
INCIDENTS = [  # no, mulai, selesai (WIB), set
    (5, "2020-12-20", "2020-12-20", "calibration"), (7, "2021-11-28", "2021-11-28", "calibration"),
    (8, "2021-12-06", "2021-12-07", "calibration"), (9, "2022-02-09", "2022-02-10", "calibration"),
    (12, "2024-03-07", "2024-03-08", "calibration"), (13, "2024-03-08", "2024-03-09", "calibration"),
    (1, "2015-12-19", "2015-12-21", "other"), (2, "2017-11-30", "2017-11-30", "other"),
    (6, "2021-03-08", "2021-03-08", "other"), (10, "2022-12-22", "2022-12-22", "other"), (11, "2022-12-28", "2022-12-31", "other"),
    (14, "2024-03-15", "2024-03-16", "other"), (15, "2024-12-03", "2024-12-03", "other"),
]
UNVERIFIED = {1, 6}   # satu sumber saja / citizen journalism (keyakinan LOW, lihat Tabel S1)


def heri(df, Tn, beta, Rmin, Wh, Wd, Smax, theta):
    R = Rmin + (1 - Rmin) * np.exp(-(((df.MWP - Tn) / (beta * Tn)) ** 2))
    fd = np.maximum(0, np.cos(np.radians(df.MWD - theta)))
    return np.clip(Wh * np.clip(df.SWH / Smax, 0, 1) * R + Wd * fd, 0, 1)


def evaluate(h, thr, override):
    flag = (h >= thr) | override
    daily = flag.groupby(flag.index.normalize()).any()
    inc = pd.Series(False, index=daily.index)
    for _, a, b, _ in INCIDENTS:
        inc.loc[a:b] = True
    rows = []
    for no, a, b, s in INCIDENTS:
        L = len(pd.date_range(a, b, freq='D'))
        win_any = daily.astype(float).rolling(L).max().shift(-(L - 1))     # window mulai hari d
        win_inc = inc.astype(float).rolling(L).max().shift(-(L - 1))
        ok = (daily.index.month == pd.Timestamp(a).month) & (win_inc == 0) & win_any.notna()
        hw = h.loc[f"{a} 00:00":f"{b} 23:00"]
        rows.append(dict(no=no, set=s, base_rate=win_any[ok].mean(),
                         detected=bool(daily.loc[a:b].any()), margin=hw.max() - thr))
    r = pd.DataFrame(rows)
    cal, oth = r[r.set == 'calibration'], r[r.set == 'other']
    m = daily.index.month
    return dict(threshold=round(thr, 3),
                calib_det=f"{cal.detected.sum()}/{len(cal)}",
                calib_min_margin=round(cal.margin.min(), 3),
                calib_base_rate=round(cal.base_rate.mean(), 3),
                calib_p_chance=float(f"{cal.base_rate.prod():.2g}"),
                other_det=f"{oth.detected.sum()}/{len(oth)}",
                other_det_verified=f"{oth[~oth.no.isin(UNVERIFIED)].detected.sum()}/"
                                   f"{(~oth.no.isin(UNVERIFIED)).sum()}",
                pct_hours_flag=round(100 * flag.mean(), 1),
                pct_days_year=round(100 * daily.mean(), 1),
                pct_days_DecJan=round(100 * daily[np.isin(m, [12, 1])].mean(), 1),
                pct_days_MayOct=round(100 * daily[np.isin(m, range(5, 11))].mean(), 1),
                missed=",".join(str(n) for n in r.no[~r.detected]) or "-")


def main():
    df = pd.read_csv(CSV, index_col='time', parse_dates=True).loc[:"2024-12-31 23:00"]
    if 'WS10' not in df.columns:
        df['WS10'] = np.hypot(df.U10, df.V10)
    df.index = df.index + pd.Timedelta(hours=7)                       # UTC -> WIB
    override = (df.SWH > SWH_LIMIT) | (df.WS10 > WS10_LIMIT)

    h0 = heri(df, **BASE)
    frac0 = ((h0 >= THR_FIXED) | override).mean()                     # % jam DANGEROUS+ baseline
    configs = [('baseline', '-', BASE)]
    for name, vals in GRID.items():
        for v in vals:
            p = dict(BASE)
            if name == 'Wh/Wd':
                p.update(Wh=v[0], Wd=v[1])
            elif isinstance(v, dict):
                p.update(v); v = ", ".join(f"{k}={x}" for k, x in v.items())
            else:
                p[name] = v
            configs.append((name, v, p))

    rows = []
    for name, v, p in configs:
        h = heri(df, **p)
        thr_m = solve_matched(h, override, frac0)
        for mode, thr in (('fixed', THR_FIXED), ('matched', thr_m)):
            if name == 'baseline' and mode == 'matched':
                continue
            rows.append(dict(parameter=name, value=v, mode=mode, **evaluate(h, thr, override)))
    res = pd.DataFrame(rows)

    # --- batas persentil (baseline HERI) ---
    prow = []
    for q_lo in (70, 75, 80):
        t = h0.quantile(q_lo / 100)
        prow.append(dict(boundary='SAFE/CAUTION', percentile=q_lo, threshold=round(t, 3),
                         pct_hours_SAFE=round(100 * ((h0 < t) & ~override).mean(), 1)))
    cal_win = [(a, b) for _, a, b, s in INCIDENTS if s == 'calibration']
    for q_hi in (99.5, 99.9, 99.95):
        t = h0.quantile(q_hi / 100)
        vd = (h0 >= t) | override
        prow.append(dict(boundary='DANGEROUS/VD', percentile=q_hi, threshold=round(t, 3),
                         pct_hours_VD=round(100 * vd.mean(), 2),
                         calib_events_with_VD=sum(vd.loc[f"{a} 00:00":f"{b} 23:00"].any()
                                                  for a, b in cal_win)))
    pct = pd.DataFrame(prow)

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / 'param_sensitivity.csv', index=False)
    pct.to_csv(OUT / 'param_sensitivity_percentiles.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(f"Baseline: {frac0:.2%} jam DANGEROUS+\n"); print(res.to_string(index=False))
    print(); print(pct.to_string(index=False))
    print(f"\nUpload: {OUT}\\param_sensitivity.csv, param_sensitivity_percentiles.csv")


def solve_matched(h, override, frac):
    """Ambang t sehingga mean((h >= t) | override) ~= frac (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if ((h >= mid) | override).mean() > frac:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    main()
