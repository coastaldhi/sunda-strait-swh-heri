r"""
07_predictive_heri.py
=====================
Evaluasi predictive HERI (Merak) dari prediksi rolling-origin fold 2024.
Tidak ada training — hanya membaca prediksi yang sudah tersimpan.

REFERENSI (label "benar"):
    HERI dari SWH, MWP, MWD aktual ERA5 pada jam target (origin + h),
    + override (SWH > 2.5 m atau WS10 > 21 kn -> SANGAT BERBAHAYA).

SKENARIO PREDIKSI:
    S1_perfect   : SWH prediksi + MWP/MWD/WS10 AKTUAL di jam target
                   (benchmark kondisional = setup paper lama)
    S2_persist   : SWH prediksi + MWP/MWD/WS10 dari jam origin (persistence)
                   (realistis: hanya info yang tersedia saat prakiraan dibuat)
    S0_allpersist: SEMUA variabel (termasuk SWH) dari jam origin
                   (baseline "persistence HERI", diminta R-A)
    SENS_*       : S1 dengan MWP/MWD diganggu (±0.5/1/2 s; ±10/20/30°) -> sensitivitas

CARA PAKAI (dari folder DL_SelatSunda):
    .\.venv\Scripts\python.exe 07_predictive_heri.py
Output di data/predictive_heri/  -> upload 3 file ini ke Claude:
    predictive_heri_summary.csv     (mean ± std antar seed, per skenario/model/horizon)
    predictive_heri_confusion.csv   (confusion matrix 4 kelas, rata-rata antar seed)
    predictive_heri_sensitivity.csv (efek perturbasi MWP/MWD)

CARA MENGUBAH:
    - Model: MODELS = ['GRU', 'XGBoost']   (nama folder: fold_2024/<MODEL>_seed_<s>;
      CNN-LSTM pakai folder seed_<s>)
    - Pelabuhan: THETA (default Merak 243.8°)
    - Blok bootstrap (jam): BLOCK = 168 (7 hari), N_BOOT = 500
    - Parameter HERI & override identik dengan heri_v2_FINAL.py + heri_override.py.

ASUMSI:
    - CSV ERA5 punya kolom time, SWH, MWP, MWD, dan WS10 (atau U10/V10).
    - test_timestamps.csv = jam origin; jam target = origin + h jam.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ═════════ KONFIGURASI ═════════
CSV_PATH = Path("data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv")
RO_DIR = Path("data/rolling_origin/fold_2024")
OUT_DIR = Path("data/predictive_heri")
MODELS = ['GRU', 'XGBoost']
SEEDS = [42, 123, 2024, 7, 99]
HORIZONS = [6, 12, 24]
THETA = 243.8                 # Merak

SWH_MAX = 2.5
T_N, BETA, FLOOR = 5.5, 0.30, 0.30
W_H, W_D = 0.70, 0.30
CAT_BOUNDS = [0.382, 0.50, 0.727]
CAT_LABELS = ["SAFE", "CAUTION", "DANGEROUS", "VERY DANGEROUS"]
SWH_LIMIT, WS10_LIMIT = 2.5, 21.0 * 0.514444      # m, m/s (BMKG ferry)

PERTURB_MWP = [-2, -1, -0.5, 0.5, 1, 2]           # s
PERTURB_MWD = [-30, -20, -10, 10, 20, 30]         # derajat
BLOCK, N_BOOT, RNG_SEED = 168, 500, 0
# ═══════════════════════════════


def heri(swh, mwp, mwd):
    swh_norm = np.clip(swh / SWH_MAX, 0, 1)
    R = FLOOR + (1 - FLOOR) * np.exp(-(((mwp - T_N) / (BETA * T_N)) ** 2))
    f_dir = np.maximum(0, np.cos(np.radians(mwd - THETA)))
    return np.clip(W_H * swh_norm * R + W_D * f_dir, 0, 1)


def category(swh, mwp, mwd, ws10):
    """Level 0..3 dengan override regulasi."""
    lvl = np.digitize(heri(swh, mwp, mwd), CAT_BOUNDS)
    over = (swh > SWH_LIMIT) | (ws10 > WS10_LIMIT)
    return np.where(over, 3, lvl)


def load_era5():
    df = pd.read_csv(CSV_PATH, index_col='time', parse_dates=True)
    cols = {c.lower(): c for c in df.columns}
    if 'ws10' not in cols:
        df['WS10'] = np.hypot(df[cols['u10']], df[cols['v10']])
    else:
        df['WS10'] = df[cols['ws10']]
    return df[['SWH', 'MWP', 'MWD', 'WS10']]


def pred_dir(name, seed):
    return RO_DIR / (f"seed_{seed}" if name == 'CNN_LSTM' else f"{name}_seed_{seed}")


def metrics(ref, pred):
    out = {'accuracy': np.mean(ref == pred)}
    f1s = []
    for k, lab in enumerate(CAT_LABELS):
        tp = np.sum((pred == k) & (ref == k))
        p = tp / max(np.sum(pred == k), 1)
        r = tp / max(np.sum(ref == k), 1)
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        out[f'prec_{lab}'], out[f'rec_{lab}'], out[f'n_{lab}'] = p, r, int(np.sum(ref == k))
        f1s.append(f)
    out['macro_F1'] = np.mean(f1s)
    rb, pb = ref >= 2, pred >= 2                       # DANGEROUS+ biner
    tp = np.sum(rb & pb)
    out['bin_precision'] = tp / max(pb.sum(), 1)
    out['bin_recall'] = tp / max(rb.sum(), 1)
    out['bin_FNR'] = 1 - out['bin_recall']
    out['skip2'] = int(np.sum(ref - pred >= 2))        # under-prediksi ≥ 2 kelas
    out['over2'] = int(np.sum(pred - ref >= 2))        # over-prediksi ≥ 2 kelas
    return out


def block_boot_ci(ref, pred, rng):
    """CI 95% recall & precision DANGEROUS+ dengan moving-block bootstrap."""
    n = len(ref)
    starts = np.arange(0, n - BLOCK + 1)
    nb = int(np.ceil(n / BLOCK))
    rec, prec = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, nb)])[:n]
        rb, pb = ref[idx] >= 2, pred[idx] >= 2
        tp = np.sum(rb & pb)
        rec.append(tp / max(rb.sum(), 1))
        prec.append(tp / max(pb.sum(), 1))
    return np.percentile(rec, [2.5, 97.5]), np.percentile(prec, [2.5, 97.5])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    era = load_era5()
    rng = np.random.default_rng(RNG_SEED)
    rows, conf_rows, sens_rows = [], [], []

    for name in MODELS:
        for seed in SEEDS:
            d = pred_dir(name, seed)
            y_pred = np.load(d / 'y_pred.npy')
            origin = pd.to_datetime(pd.read_csv(d / 'test_timestamps.csv').iloc[:, 0])
            o = era.loc[origin].to_numpy()                       # kondisi di jam origin
            for i, h in enumerate(HORIZONS):
                tgt = era.loc[origin + pd.Timedelta(hours=h)].to_numpy()
                swh_t, mwp_t, mwd_t, ws_t = tgt.T
                ref = category(swh_t, mwp_t, mwd_t, ws_t)
                swh_p = y_pred[:, i]

                scen = {
                    'S1_perfect': category(swh_p, mwp_t, mwd_t, ws_t),
                    'S2_persist': category(swh_p, o[:, 1], o[:, 2], o[:, 3]),
                    'S0_allpersist': category(o[:, 0], o[:, 1], o[:, 2], o[:, 3]),
                }
                for sname, pred in scen.items():
                    if sname == 'S0_allpersist' and (name != MODELS[0] or seed != SEEDS[0]):
                        continue                                 # cukup sekali
                    m = metrics(ref, pred)
                    (rl, rh), (pl, ph) = block_boot_ci(ref, pred, rng)
                    mdl = 'Persistence' if sname == 'S0_allpersist' else name
                    rows.append({'scenario': sname, 'model': mdl, 'seed': seed,
                                 'horizon': f"t+{h}h", **m,
                                 'bin_recall_lo': rl, 'bin_recall_hi': rh,
                                 'bin_prec_lo': pl, 'bin_prec_hi': ph})
                    cm = pd.crosstab(pd.Categorical(ref, range(4)),
                                     pd.Categorical(pred, range(4)), dropna=False)
                    for a in range(4):
                        for b in range(4):
                            conf_rows.append({'scenario': sname, 'model': mdl, 'seed': seed,
                                              'horizon': f"t+{h}h", 'ref': CAT_LABELS[a],
                                              'pred': CAT_LABELS[b], 'count': int(cm.iloc[a, b])})

                for dp in PERTURB_MWP:
                    m = metrics(ref, category(swh_p, mwp_t + dp, mwd_t, ws_t))
                    sens_rows.append({'model': name, 'seed': seed, 'horizon': f"t+{h}h",
                                      'perturb': f"MWP{dp:+g}s", **m})
                for dd in PERTURB_MWD:
                    m = metrics(ref, category(swh_p, mwp_t, (mwd_t + dd) % 360, ws_t))
                    sens_rows.append({'model': name, 'seed': seed, 'horizon': f"t+{h}h",
                                      'perturb': f"MWD{dd:+d}deg", **m})
        print(f"{name} selesai")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'predictive_heri_all_seeds.csv', index=False)
    keep = ['accuracy', 'macro_F1', 'bin_precision', 'bin_recall', 'bin_FNR',
            'rec_DANGEROUS', 'rec_VERY DANGEROUS', 'prec_VERY DANGEROUS',
            'n_DANGEROUS', 'n_VERY DANGEROUS', 'skip2', 'over2',
            'bin_recall_lo', 'bin_recall_hi', 'bin_prec_lo', 'bin_prec_hi']
    summ = res.groupby(['scenario', 'model', 'horizon'])[keep].agg(['mean', 'std']).round(3)
    summ.to_csv(OUT_DIR / 'predictive_heri_summary.csv')

    conf = (pd.DataFrame(conf_rows).groupby(['scenario', 'model', 'horizon', 'ref', 'pred'])
            ['count'].mean().round(1).reset_index())
    conf.to_csv(OUT_DIR / 'predictive_heri_confusion.csv', index=False)

    sens = pd.DataFrame(sens_rows)
    sens_s = (sens.groupby(['model', 'horizon', 'perturb'])
              [['accuracy', 'macro_F1', 'bin_precision', 'bin_recall']].mean().round(3))
    sens_s.to_csv(OUT_DIR / 'predictive_heri_sensitivity.csv')

    pd.set_option('display.width', 200)
    print("\n=== Biner DANGEROUS+ (mean antar seed) ===")
    print(res.groupby(['scenario', 'model', 'horizon'])
          [['accuracy', 'bin_precision', 'bin_recall', 'rec_VERY DANGEROUS', 'skip2']]
          .mean().round(3).to_string())
    print(f"\nUpload: {OUT_DIR}\\predictive_heri_summary.csv, "
          f"predictive_heri_confusion.csv, predictive_heri_sensitivity.csv")


if __name__ == "__main__":
    main()
