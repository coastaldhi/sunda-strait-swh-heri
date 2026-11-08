r"""
06_dm_test.py
=============
Uji Diebold–Mariano (DM) berpasangan antar model, memakai prediksi yang disimpan
05_rolling_origin.py di data/rolling_origin/fold_<tahun>/...

Loss = squared error, dirata-ratakan antar seed per jam:
    d_t = mean_s(e_A,s,t^2) - mean_s(e_B,s,t^2)
    d < 0  -> model A lebih akurat
Varians memakai Newey–West (HAC) dengan lag NW_LAG jam, plus koreksi
Harvey–Leybourne–Newbold (HLN) untuk sampel terbatas. p-value dua sisi (distribusi t).

CARA PAKAI (dari folder DL_SelatSunda):
    .\.venv\Scripts\python.exe 06_dm_test.py
Output:
    data/rolling_origin/dm_test.csv   -> upload ke Claude

CARA MENGUBAH:
    - Lag autokorelasi: NW_LAG (default 48 jam; error per jam berautokorelasi kuat,
      jadi lag kecil seperti h-1 terlalu optimistis).
    - Model/seed/fold: MODELS, SEEDS, TEST_YEARS (harus sudah ada hasilnya).
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ═════════ KONFIGURASI ═════════
OUT_DIR = Path("data/rolling_origin")
TEST_YEARS = [2021, 2022, 2023, 2024]
MODELS = ['CNN_LSTM', 'BiLSTM', 'XGBoost', 'LSTM', 'GRU', 'Persistence']
SEEDS = [42, 123, 2024, 7, 99]
HORIZONS = [6, 12, 24]
NW_LAG = 48
# ═══════════════════════════════


def run_dir(ty, name, seed):
    sub = f"seed_{seed}" if name == 'CNN_LSTM' else f"{name}_seed_{seed}"
    return OUT_DIR / f"fold_{ty}" / sub


def sq_err(ty, name):
    """Squared error per jam (n, n_horizon), rata-rata antar seed."""
    y_true = np.load(run_dir(ty, 'CNN_LSTM', SEEDS[0]) / 'y_true.npy')
    if name == 'Persistence':
        raise RuntimeError("persistence dihitung terpisah")
    errs = [(np.load(run_dir(ty, name, s) / 'y_pred.npy') - y_true) ** 2 for s in SEEDS]
    return np.mean(errs, axis=0)


def persistence_sq_err(ty):
    """SWH(t+h) = SWH(t); SWH(t) direkonstruksi dari y_true jam-jam sebelumnya."""
    y_true = np.load(run_dir(ty, 'CNN_LSTM', SEEDS[0]) / 'y_true.npy')
    out = np.full_like(y_true, np.nan)
    for i, h in enumerate(HORIZONS):
        # y_true[k, i] = SWH(origin_k + h); SWH(origin_k) = y_true[k - h, i]
        out[h:, i] = (y_true[:-h, i] - y_true[h:, i]) ** 2
    return out


def dm_test(d, h, lag):
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return dbar, stat, p, n


def main():
    rows = []
    for ty in TEST_YEARS:
        se = {m: (persistence_sq_err(ty) if m == 'Persistence' else sq_err(ty, m))
              for m in MODELS}
        for i, h in enumerate(HORIZONS):
            for a, b in combinations(MODELS, 2):
                d = se[a][:, i] - se[b][:, i]
                dbar, stat, p, n = dm_test(d, h, NW_LAG)
                rows.append({'test_year': ty, 'horizon': f"t+{h}h", 'A': a, 'B': b,
                             'mean_d': dbar, 'DM_HLN': stat, 'p': p, 'n': n,
                             'lebih_akurat': a if dbar < 0 else b,
                             'signifikan_5pct': p < 0.05})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'dm_test.csv', index=False)

    tab = (res.assign(pair=res.A + ' vs ' + res.B)
              .pivot_table(index=['horizon', 'pair'], columns='test_year',
                           values='p', aggfunc='first').round(3))
    print("p-value DM (HLN), per tahun uji:")
    print(tab.to_string())
    print(f"\nSaved: {OUT_DIR / 'dm_test.csv'}")


if __name__ == "__main__":
    main()
