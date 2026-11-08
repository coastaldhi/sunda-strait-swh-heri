r"""
11_make_figures.py
==================
Membuat ulang SEMUA gambar berbasis data untuk naskah revisi (v6), memakai standar fig_style.py.
Gambar yang datanya belum ada otomatis dilewati (dicetak [SKIP]), yang lain tetap dibuat.

CARA PAKAI (dari folder DL_SelatSunda; fig_style.py harus satu folder):
    .\.venv\Scripts\python.exe 11_make_figures.py
Output: folder figures\  (PDF vektor untuk submit + PNG 600 dpi untuk Word)

PETA GAMBAR (nomor = nomor di naskah v6):
    Fig03_historical_swh        <- data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv
    Fig04_rolling_origin        <- data/rolling_origin/rolling_origin_results.csv
    Fig05_timeseries_gru        <- data/rolling_origin/fold_2024/GRU_seed_*/ (y_pred, y_true, timestamps)
                                   (garis/scatter = rata-rata prediksi 5 seed; angka RMSE di Fig06
                                    = rata-rata RMSE per seed, konsisten dengan Tabel 3)
    Fig06_scatter_tail          <- idem
    Fig07_operating_range       <- (tanpa data; dari rumus HERI)
    Fig08_monthly_frequency     <- data/revision_round2/base_rate_monthly.csv + incident_table.csv
    Fig10_predictive_scenarios  <- data/predictive_heri/predictive_heri_summary.csv
                                   + data/multioutput_gru/mo_heri_summary.csv
    Fig11_sensitivity           <- data/predictive_heri/predictive_heri_sensitivity.csv
    Fig12_confusion             <- data/predictive_heri/predictive_heri_confusion.csv
    FigS1_monthly_error         <- data/rolling_origin/fold_2024/GRU_seed_*/
    Fig01 (peta), Fig02 (flowchart HERI), Fig09 (mudik) = dibuat manual, lihat rencana gambar.

CARA MENGUBAH:
    - Model utama untuk Fig05/06/S1/10–12: MAIN_MODEL
    - Horizon yang ditampilkan di Fig12: CONF_HORIZON
    - Hanya membuat sebagian gambar: ubah daftar JOBS di bagian bawah
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm

from fig_style import (apply_style, new_fig, save_fig, panel_label, MODEL_STYLE, MODEL_ORDER,
                       CAT_LABELS, CAT_COLORS, SCENARIO_STYLE)

ERA5 = Path("data/era5_processed/era5_titik1_tengah_selat_2010_2025.csv")
RO = Path("data/rolling_origin")
PH = Path("data/predictive_heri")
MO = Path("data/multioutput_gru")
RV = Path("data/revision_round2")
MAIN_MODEL = 'GRU'
SEEDS = [42, 123, 2024, 7, 99]
HORIZONS = [6, 12, 24]
CONF_HORIZON = 't+12h'
SWH_MAX, T_N, BETA, FLOOR, W_H, W_D = 2.5, 5.5, 0.30, 0.30, 0.70, 0.30
BOUNDS = [0.382, 0.50, 0.727]


def need(*paths):
    miss = [str(p) for p in paths if not Path(p).exists()]
    if miss:
        raise FileNotFoundError(", ".join(miss))


def load_gru_fold2024():
    preds = []
    for s in SEEDS:
        d = RO / "fold_2024" / f"{MAIN_MODEL}_seed_{s}"
        if (d / 'y_pred.npy').exists():
            preds.append(np.load(d / 'y_pred.npy'))
            y = np.load(d / 'y_true.npy')
            t = pd.to_datetime(pd.read_csv(d / 'test_timestamps.csv').iloc[:, 0])
    if not preds:
        raise FileNotFoundError(f"{RO}/fold_2024/{MAIN_MODEL}_seed_*")
    return np.mean(preds, axis=0), y, t, preds


# ─────────────────────────── gambar ───────────────────────────
def fig03_historical_swh():
    need(ERA5)
    df = pd.read_csv(ERA5, index_col='time', parse_dates=True).loc[:"2024-12-31 23:00"]
    fig, (a, b) = new_fig('double', 62, 1, 2, gridspec_kw=dict(width_ratios=[2.2, 1]))
    m = df.SWH.resample('MS').mean()
    mx = df.SWH.resample('MS').max()
    a.fill_between(m.index, m, mx, color='#0072B2', alpha=0.18, lw=0, label='Monthly max')
    a.plot(m.index, m, color='#0072B2', label='Monthly mean')
    for q, ls in [(0.95, ':'), (0.99, '--')]:
        a.axhline(df.SWH.quantile(q), color='#555555', ls=ls, lw=0.8,
                  label=f'P{int(q*100)} = {df.SWH.quantile(q):.2f} m')
    a.axhline(2.5, color='#C62828', lw=0.8, label='2.5 m (BMKG ferry)')
    a.set_ylabel('SWH (m)'); a.set_ylim(0, 2.6)
    panel_label(a, 'a')
    clim = df.SWH.groupby(df.index.month)
    b.plot(range(1, 13), clim.mean(), marker='o', color='#0072B2', label='Mean')
    b.plot(range(1, 13), clim.quantile(0.99), marker='^', color='#555555', ls='--', label='P99')
    b.set_xticks(range(1, 13)); b.set_xticklabels('JFMAMJJASOND')
    b.set_xlabel('Month'); b.set_ylabel('SWH (m)'); b.legend(); panel_label(b, 'b')
    h_, l_ = a.get_legend_handles_labels()
    fig.legend(h_, l_, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, 'Fig03_historical_swh')


def fig04_rolling_origin():
    need(RO / 'rolling_origin_results.csv')
    r = pd.read_csv(RO / 'rolling_origin_results.csv')
    fig, axes = new_fig('double', 62, 1, 3, sharey=False)
    for ax, h, L in zip(axes, ['t+6h', 't+12h', 't+24h'], 'abc'):
        sub = r[r.horizon == h]
        for mname in MODEL_ORDER:
            g = sub[sub.model == mname]
            if g.empty:
                continue
            st = MODEL_STYLE[mname]
            agg = g.groupby('test_year').RMSE.agg(['mean', 'std'])
            ax.errorbar(agg.index, agg['mean'], yerr=agg['std'].fillna(0), color=st['color'],
                        marker=st['marker'], ls=st.get('linestyle', '-'), capsize=1.5,
                        lw=0.9, label=st['label'])
        ax.set_title(f'Lead time {h[2:]}'); ax.set_xlabel('Test year')
        ax.set_xticks(sorted(sub.test_year.unique()))
        panel_label(ax, L)
    axes[0].set_ylabel('RMSE (m)')
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, 'Fig04_rolling_origin')


def fig05_timeseries():
    yp, yt, t, ns = load_gru_fold2024()
    windows = [('2024-01-01', '2024-01-31', 'January (west monsoon)'),
               ('2024-04-01', '2024-04-30', 'April (Lebaran period)'),
               ('2024-07-01', '2024-07-31', 'July (east monsoon)')]
    fig, axes = new_fig('double', 110, 3, 1, sharey=True)
    for ax, (a, b, title), L in zip(axes, windows, 'abc'):
        for i, (h, c, ls) in enumerate(zip(HORIZONS, ['#0072B2', '#009E73', '#D55E00'],
                                           ['-', '-', '-'])):
            tt = t + pd.Timedelta(hours=h)
            msk = (tt >= a) & (tt <= f"{b} 23:00")
            if i == 0:
                obs = pd.Series(yt[:, 0], index=t + pd.Timedelta(hours=6))
                o = obs[(obs.index >= a) & (obs.index <= f"{b} 23:00")]
                ax.plot(o.index, o.values, color='black', lw=1.1, label='ERA5')
            ax.plot(tt[msk], yp[msk, i], color=c, lw=0.7, label=f'{MAIN_MODEL} t+{h} h')
        ax.set_title(title, loc='left'); ax.set_ylabel('SWH (m)'); panel_label(ax, L)
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, 'Fig05_timeseries_gru')


def fig06_scatter_tail():
    yp, yt, t, ns = load_gru_fold2024()
    fig, axes = new_fig('double', 62, 1, 4)
    lim = (0, max(yt.max(), yp.max()) * 1.05)
    for ax, i, h, L in zip(axes[:3], range(3), HORIZONS, 'abc'):
        ax.scatter(yt[:, i], yp[:, i], s=1.5, color='#0072B2', alpha=0.25, lw=0, rasterized=True)
        ax.plot(lim, lim, color='#555555', lw=0.7)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        rmse = np.mean([np.sqrt(np.mean((p_[:, i] - yt[:, i]) ** 2)) for p_ in ns])
        ax.text(0.96, 0.04, f't+{h} h\nRMSE = {rmse:.3f} m', transform=ax.transAxes,
                va='bottom', ha='right')
        ax.set_xlabel('ERA5 SWH (m)'); panel_label(ax, L)
    axes[0].set_ylabel(f'{MAIN_MODEL} SWH (m)')
    ax = axes[3]
    edges = np.quantile(yt[:, 0], [0, .5, .9, .95, .99, 1])
    names = ['<P50', 'P50–90', 'P90–95', 'P95–99', '>P99']
    x = np.arange(len(names))
    for k, (i, h, c) in enumerate(zip(range(3), HORIZONS, ['#0072B2', '#009E73', '#D55E00'])):
        bias = [np.mean((yp[:, i] - yt[:, i])[(yt[:, i] >= lo) & (yt[:, i] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.bar(x + (k - 1) * 0.26, bias, 0.24, color=c, label=f't+{h} h')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Mean error (m)'); ax.set_xlabel('Observed SWH class')
    ax.legend(); panel_label(ax, 'd')
    fig.tight_layout()
    save_fig(fig, 'Fig06_scatter_tail')


def fig07_operating_range():
    swh = np.linspace(0, 3, 301); mwp = np.linspace(3, 14, 221)
    S, T = np.meshgrid(swh, mwp)
    cmap = ListedColormap(CAT_COLORS); norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    fig, axes = new_fig('double', 70, 1, 3, sharey=True)
    cases = [(0.0, False, 'F_dir = 0, no override'), (0.0, True, 'F_dir = 0, with override'),
             (1.0, True, 'F_dir = 1, with override')]
    for ax, (fd, ov, title), L in zip(axes, cases, 'abc'):
        R = FLOOR + (1 - FLOOR) * np.exp(-(((T - T_N) / (BETA * T_N)) ** 2))
        lv = np.digitize(np.clip(W_H * np.clip(S / SWH_MAX, 0, 1) * R + W_D * fd, 0, 1), BOUNDS)
        if ov:
            lv = np.where(S > 2.5, 3, lv)
        ax.pcolormesh(S, T, lv, cmap=cmap, norm=norm, shading='auto', rasterized=True)
        ax.axvline(2.405, color='black', ls=':', lw=0.8)
        ax.set_title(title, loc='left'); ax.set_xlabel('SWH (m)'); ax.grid(False)
        panel_label(ax, L)
    axes[0].set_ylabel('MWP (s)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLORS]
    fig.legend(handles, CAT_LABELS, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, 'Fig07_operating_range')


def fig08_monthly_frequency():
    need(RV / 'base_rate_monthly.csv', RV / 'incident_table.csv')
    m = pd.read_csv(RV / 'base_rate_monthly.csv', index_col=0)
    inc = pd.read_csv(RV / 'incident_table.csv')
    inc['month'] = pd.to_datetime(inc.start_WIB).dt.month
    fig, (a, b) = new_fig('single', 95, 2, 1, sharex=True)
    styles = [('#0072B2', 'o', '-'), ('#555555', 's', '--'), ('#999999', '^', ':')]
    for (col, (c, mk, ls)) in zip(m.columns, styles):
        a.plot(m.index, 100 * m[col], color=c, marker=mk, ls=ls,
               label='HERI DANGEROUS+' if col.startswith(('HERI', 'pct_days')) else col)
    a.set_ylabel('Days with ≥1 h flagged (%)'); a.set_ylim(0, 58)
    a.legend(loc='upper center'); panel_label(a, 'a')
    order = ['merak', 'route_wide', 'unknown', 'bakauheni']
    cols = {'merak': '#0072B2', 'route_wide': '#E69F00', 'unknown': '#BBBBBB', 'bakauheni': '#009E73'}
    bottom = np.zeros(12)
    for loc in order:
        cnt = inc[inc.location == loc].month.value_counts().reindex(range(1, 13), fill_value=0)
        b.bar(range(1, 13), cnt.values, bottom=bottom, color=cols[loc], width=0.7,
              edgecolor='white', lw=0.5, label=loc.replace('_', '-').capitalize())
        bottom += cnt.values
    b.set_ylabel('Incidents (n)'); b.set_xticks(range(1, 13)); b.set_xticklabels('JFMAMJJASOND')
    b.set_xlabel('Month'); b.legend(ncol=2); panel_label(b, 'b')
    fig.tight_layout()
    save_fig(fig, 'Fig08_monthly_frequency')


def _read_summary(path):
    s = pd.read_csv(path, header=[0, 1], index_col=[0, 1, 2] if 'mo_' not in str(path) else [0, 1])
    return s


def fig10_predictive_scenarios():
    need(PH / 'predictive_heri_summary.csv')
    s = pd.read_csv(PH / 'predictive_heri_summary.csv', header=[0, 1], index_col=[0, 1, 2])
    rows = {}
    for sc in ['S1_perfect', 'S2_persist', 'S0_allpersist']:
        mdl = 'Persistence' if sc == 'S0_allpersist' else MAIN_MODEL
        rows[sc] = s.xs((sc, mdl), level=(0, 1))
    if (MO / 'mo_heri_summary.csv').exists():
        mo = pd.read_csv(MO / 'mo_heri_summary.csv', header=[0, 1], index_col=[0, 1])
        rows['S3_allpredicted'] = mo.xs('S3_allpredicted', level=0)
    order = [k for k in ['S1_perfect', 'S2_persist', 'S3_allpredicted', 'S0_allpersist'] if k in rows]
    fig, axes = new_fig('double', 62, 1, 3, sharey=True)
    for ax, (met, lab), L in zip(axes, [('bin_recall', 'Recall (DANGEROUS+)'),
                                         ('bin_precision', 'Precision (DANGEROUS+)'),
                                         ('macro_F1', 'Macro-F1 (4 classes)')], 'abc'):
        x = np.arange(3); w = 0.8 / len(order)
        for k, sc in enumerate(order):
            st = SCENARIO_STYLE[sc]
            df = rows[sc].reindex(['t+6h', 't+12h', 't+24h'])
            ax.bar(x + (k - (len(order) - 1) / 2) * w, df[(met, 'mean')], w * 0.92,
                   yerr=df[(met, 'std')].fillna(0), color=st['color'], hatch=st['hatch'],
                   edgecolor='white', lw=0.4, capsize=1, label=st['label'])
        ax.set_xticks(x); ax.set_xticklabels(['6 h', '12 h', '24 h']); ax.set_xlabel('Lead time')
        ax.set_title(lab, loc='left'); ax.set_ylim(0, 1); panel_label(ax, L)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save_fig(fig, 'Fig10_predictive_scenarios')


def fig11_sensitivity():
    need(PH / 'predictive_heri_sensitivity.csv')
    s = pd.read_csv(PH / 'predictive_heri_sensitivity.csv')
    s = s[s.model == MAIN_MODEL]
    s['var'] = s.perturb.str[:3]
    s['delta'] = s.perturb.str.extract(r'([+-][\d.]+)')[0].astype(float)
    base = pd.read_csv(PH / 'predictive_heri_summary.csv', header=[0, 1], index_col=[0, 1, 2])
    fig, axes = new_fig('double', 62, 1, 2)
    for ax, var, unit, L in zip(axes, ['MWP', 'MWD'], ['s', '°'], 'ab'):
        for h, c, mk in zip(['t+6h', 't+12h', 't+24h'], ['#0072B2', '#009E73', '#D55E00'], 'o^s'):
            g = s[(s['var'] == var) & (s.horizon == h)].sort_values('delta')
            b0 = base.loc[('S1_perfect', MAIN_MODEL, h)]
            xs = np.r_[g.delta.values, 0]; order = np.argsort(xs)
            rec = np.r_[g.bin_recall.values, b0[('bin_recall', 'mean')]][order]
            pre = np.r_[g.bin_precision.values, b0[('bin_precision', 'mean')]][order]
            ax.plot(xs[order], rec, color=c, marker=mk, label=f'Recall {h[2:]}')
            ax.plot(xs[order], pre, color=c, marker=mk, ls='--', mfc='white',
                    label=f'Precision {h[2:]}')
        ax.axvline(0, color='#555555', lw=0.6)
        ax.set_xlabel(f'Perturbation of {var} ({unit})'); ax.set_ylim(0, 1.02)
        panel_label(ax, L)
    axes[0].set_ylabel('Score (DANGEROUS+)')
    axes[1].legend(ncol=2, loc='lower center')
    fig.tight_layout()
    save_fig(fig, 'Fig11_sensitivity')


def fig12_confusion():
    need(PH / 'predictive_heri_confusion.csv')
    c = pd.read_csv(PH / 'predictive_heri_confusion.csv')
    fig, axes = new_fig('double', 70, 1, 2)
    for ax, sc, title, L in zip(axes, ['S1_perfect', 'S2_persist'],
                                ['S1: future MWP/MWD known', 'S2: persisted MWP/MWD'], 'ab'):
        g = c[(c.scenario == sc) & (c.model == MAIN_MODEL) & (c.horizon == CONF_HORIZON)]
        M = g.pivot(index='ref', columns='pred', values='count').reindex(
            index=CAT_LABELS, columns=CAT_LABELS).fillna(0).values
        P = M / M.sum(1, keepdims=True)
        ax.imshow(P, cmap='Blues', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{P[i, j]*100:.0f}%\n({M[i, j]:.0f})', ha='center', va='center',
                        color='white' if P[i, j] > 0.6 else 'black', fontsize=6.5)
        short = ['SAFE', 'CAUT.', 'DANG.', 'V. DANG.']
        ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_yticks(range(4))
        ax.set_yticklabels(short); ax.set_xlabel('Predicted'); ax.grid(False)
        ax.set_title(f'{title} ({CONF_HORIZON[2:]})', loc='left'); panel_label(ax, L)
    axes[0].set_ylabel('Reference (ERA5)')
    fig.tight_layout()
    save_fig(fig, 'Fig12_confusion')


def figS1_monthly_error():
    yp, yt, t, ns = load_gru_fold2024()
    fig, axes = new_fig('double', 55, 1, 2)
    for i, (h, c, mk) in enumerate(zip(HORIZONS, ['#0072B2', '#009E73', '#D55E00'], 'o^s')):
        tt = (t + pd.Timedelta(hours=h)).dt.month.values
        e = yp[:, i] - yt[:, i]
        rm = [np.sqrt(np.mean(e[tt == m] ** 2)) for m in range(1, 13)]
        bi = [np.mean(e[tt == m]) for m in range(1, 13)]
        axes[0].plot(range(1, 13), rm, color=c, marker=mk, label=f't+{h} h')
        axes[1].plot(range(1, 13), bi, color=c, marker=mk, label=f't+{h} h')
    for ax, lab, L in zip(axes, ['RMSE (m)', 'Bias (m)'], 'ab'):
        ax.set_xticks(range(1, 13)); ax.set_xticklabels('JFMAMJJASOND'); ax.set_xlabel('Month')
        ax.set_ylabel(lab); panel_label(ax, L)
    axes[1].axhline(0, color='black', lw=0.6); axes[0].legend()
    fig.tight_layout()
    save_fig(fig, 'FigS1_monthly_error')


JOBS = [fig03_historical_swh, fig04_rolling_origin, fig05_timeseries, fig06_scatter_tail,
        fig07_operating_range, fig08_monthly_frequency, fig10_predictive_scenarios,
        fig11_sensitivity, fig12_confusion, figS1_monthly_error]

if __name__ == "__main__":
    apply_style()
    for job in JOBS:
        print(job.__name__)
        try:
            job()
        except FileNotFoundError as e:
            print(f"  [SKIP] data tidak ditemukan: {e}")
