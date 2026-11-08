"""
heri_override.py
================
Menambahkan aturan OVERRIDE regulasi ke kategori HERI v2 (nilai HERI TIDAK diubah).

ATURAN (level kategori):
    Cat_final = max( Cat_HERI , Cat_override )
    Cat_override = SANGAT BERBAHAYA jika  SWH > 2.5 m  ATAU  WS10 > 21 knot (10.80 m/s)
    Sumber ambang: BMKG, risiko tinggi keselamatan pelayaran untuk kapal ferry
    (siaran pers BMKG 31 Juli 2018): "angin > 21 knot dan gelombang > 2.5 m".

CARA PAKAI:
    1. Taruh skrip ini satu folder dengan era5_HERI_v2_FINAL.csv (output heri_v2_FINAL.py).
    2. Jalankan:   python heri_override.py
    3. Output:
       - era5_HERI_v2_override.csv      -> kolom baru *_final dan flag override
       - override_summary.txt           -> jumlah jam override & perubahan kategori
       - fig_heri_operating_range.png   -> peta kategori seluruh rentang operasi (untuk R-A poin 5)
    4. Lalu jalankan ulang match_insiden_heri.py dengan kolom HERI_v2_cat_final_merak
       untuk cek apakah status 6 insiden kalibrasi berubah.

CARA MENGUBAH:
    - Ambang: ubah SWH_LIMIT / WS10_LIMIT_KNOT di blok CONFIG.
    - Pakai ">=" alih-alih ">" : ubah STRICT = False.
    - Kalau angin cukup menaikkan ke BERBAHAYA saja (bukan SANGAT BERBAHAYA):
      ubah WIND_OVERRIDE_LEVEL = 2.
    - Nama kolom angin: skrip mencari WS10; kalau tidak ada, dihitung dari U10/V10
      (tidak peka huruf besar/kecil). Sesuaikan WIND_COLS kalau nama kolom berbeda.

ASUMSI:
    - Kolom yang ada: time, SWH [m], MWP [s], MWD [deg, nautical "from"],
      dan WS10 [m/s] atau U10/V10 [m/s].
    - WS10 ERA5 = rata-rata per jam pada 10 m, bukan hembusan (gust).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ═════════ CONFIG ═════════
CSV_IN = "era5_HERI_v2_FINAL.csv"
CSV_OUT = "era5_HERI_v2_override.csv"
SUMMARY_OUT = "override_summary.txt"
FIG_OUT = "fig_heri_operating_range.png"

SWH_LIMIT = 2.5            # m
WS10_LIMIT_KNOT = 21.0     # knot
KNOT_TO_MS = 0.514444
STRICT = True              # True: ">" (sesuai teks BMKG "lebih dari"); False: ">="
SWH_OVERRIDE_LEVEL = 3     # 0=AMAN 1=WASPADA 2=BERBAHAYA 3=SANGAT BERBAHAYA
WIND_OVERRIDE_LEVEL = 3
WIND_COLS = ("WS10", "U10", "V10")

# Parameter HERI v2 (identik dengan heri_v2_FINAL.py — jangan diubah di sini saja)
SWH_MAX = 2.5
T_N, BETA, FLOOR = 5.5, 0.30, 0.30
W_H, W_D = 0.70, 0.30
THETA = {"merak": 243.8, "bakauheni": 130.8}
CAT_BOUNDS = [0.0, 0.382, 0.50, 0.727, 1.01]
CAT_LABELS = ["AMAN", "WASPADA", "BERBAHAYA", "SANGAT BERBAHAYA"]
# ══════════════════════════


def compute_heri_v2(swh, mwp, mwd, theta):
    swh_norm = np.clip(swh / SWH_MAX, 0.0, 1.0)
    R = FLOOR + (1 - FLOOR) * np.exp(-(((mwp - T_N) / (BETA * T_N)) ** 2))
    f_dir = np.maximum(0.0, np.cos(np.radians(mwd - theta)))
    return np.clip(W_H * swh_norm * R + W_D * f_dir, 0.0, 1.0)


def heri_level(h):
    """HERI -> level integer 0..3"""
    return np.digitize(h, CAT_BOUNDS[1:-1])


def exceed(x, lim):
    return x > lim if STRICT else x >= lim


def override_level(swh, ws10):
    lvl = np.zeros(len(swh), dtype=int)
    lvl = np.where(exceed(swh, SWH_LIMIT), np.maximum(lvl, SWH_OVERRIDE_LEVEL), lvl)
    lvl = np.where(exceed(ws10, WS10_LIMIT_KNOT * KNOT_TO_MS),
                   np.maximum(lvl, WIND_OVERRIDE_LEVEL), lvl)
    return lvl


def get_ws10(df):
    cols = {c.lower(): c for c in df.columns}
    if WIND_COLS[0].lower() in cols:
        return df[cols[WIND_COLS[0].lower()]].values
    u, v = cols.get(WIND_COLS[1].lower()), cols.get(WIND_COLS[2].lower())
    if u is None or v is None:
        raise KeyError(f"Kolom angin tidak ditemukan. Kolom tersedia: {list(df.columns)}")
    return np.hypot(df[u].values, df[v].values)


def operating_range_figure():
    """Peta kategori SWH 0–3 m × MWP 3–14 s, F_dir=0 dan 1, sebelum/sesudah override SWH."""
    swh = np.linspace(0, 3.0, 301)
    mwp = np.linspace(3, 14, 221)
    S, T = np.meshgrid(swh, mwp)
    cmap = ListedColormap(["#2e7d32", "#f9a825", "#ef6c00", "#c62828"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for i, fdir in enumerate([0.0, 1.0]):
        h = np.clip(W_H * np.clip(S / SWH_MAX, 0, 1)
                    * (FLOOR + (1 - FLOOR) * np.exp(-(((T - T_N) / (BETA * T_N)) ** 2)))
                    + W_D * fdir, 0, 1)
        base = heri_level(h)
        final = np.where(exceed(S, SWH_LIMIT), np.maximum(base, SWH_OVERRIDE_LEVEL), base)
        for j, (lvl, ttl) in enumerate([(base, "HERI (no override)"),
                                        (final, "HERI + SWH override")]):
            ax = axes[i, j]
            ax.pcolormesh(S, T, lvl, cmap=cmap, norm=norm, shading="auto")
            ax.axvline(SWH_LIMIT, color="k", ls="--", lw=1)
            ax.axvline(2.405, color="k", ls=":", lw=1)  # maks SWH ERA5 di titik studi
            ax.set_title(f"{ttl}, F_dir = {fdir:.0f}", fontsize=10)
            if i == 1:
                ax.set_xlabel("SWH (m)")
            if j == 0:
                ax.set_ylabel("MWP (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(4)]
    fig.legend(handles, ["SAFE", "CAUTION", "DANGEROUS", "VERY DANGEROUS"],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_OUT, dpi=300)
    print(f"Saved: {FIG_OUT}")


def main():
    df = pd.read_csv(CSV_IN, parse_dates=["time"]).set_index("time")
    ws10 = get_ws10(df)
    df["WS10_knot"] = ws10 / KNOT_TO_MS
    df["override_swh"] = exceed(df["SWH"].values, SWH_LIMIT)
    df["override_wind"] = exceed(ws10, WS10_LIMIT_KNOT * KNOT_TO_MS)
    ov = override_level(df["SWH"].values, ws10)

    lines = [f"Data: {len(df):,} jam, {df.index.min()} – {df.index.max()}",
             f"SWH maks = {df['SWH'].max():.3f} m ; WS10 maks = {df['WS10_knot'].max():.1f} knot",
             f"Jam override SWH  : {int(df['override_swh'].sum())}",
             f"Jam override angin: {int(df['override_wind'].sum())}",
             f"Jam override total: {int((df['override_swh'] | df['override_wind']).sum())}", ""]

    for port, theta in THETA.items():
        h = compute_heri_v2(df["SWH"].values, df["MWP"].values, df["MWD"].values, theta)
        base = heri_level(h)
        final = np.maximum(base, ov)
        df[f"HERI_v2_{port}"] = h
        df[f"HERI_v2_cat_{port}"] = np.array(CAT_LABELS)[base]
        df[f"HERI_v2_cat_final_{port}"] = np.array(CAT_LABELS)[final]
        changed = base != final
        lines.append(f"== {port.upper()} ==")
        lines.append(f"Jam kategori berubah: {int(changed.sum())} "
                     f"({100 * changed.mean():.3f}%)")
        lines.append("Distribusi sebelum -> sesudah (%):")
        for k, lab in enumerate(CAT_LABELS):
            lines.append(f"  {lab:17s} {100*(base==k).mean():6.2f} -> {100*(final==k).mean():6.2f}")
        if changed.any():
            ch = df.loc[changed]
            lines.append("Per tahun (jam berubah):")
            lines.append(ch.groupby(ch.index.year).size().to_string())
            lines.append("Per bulan (jam berubah):")
            lines.append(ch.groupby(ch.index.month).size().to_string())
            lines.append("Asal kategori yang naik:")
            lines.append(pd.Series(np.array(CAT_LABELS)[base[changed]]).value_counts().to_string())
        lines.append("")

    txt = "\n".join(lines)
    print(txt)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write(txt)
    df.to_csv(CSV_OUT)
    print(f"Saved: {CSV_OUT}, {SUMMARY_OUT}")
    operating_range_figure()


if __name__ == "__main__":
    main()
