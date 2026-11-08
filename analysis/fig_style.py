"""
fig_style.py
============
Standar gambar untuk revisi JOEME (Springer). Semua skrip gambar WAJIB import file ini
supaya font, ukuran, warna, dan format output seragam.

CARA PAKAI:
    from fig_style import apply_style, new_fig, save_fig, MODEL_STYLE, CAT_COLORS
    apply_style()
    fig, ax = new_fig('single', height_mm=60)      # 'single' = 84 mm, 'double' = 174 mm
    ...
    save_fig(fig, 'Fig05_rolling_origin')          # -> figures/Fig05_rolling_origin.pdf + .png

STANDAR (sesuai panduan artwork Springer; cek ulang di halaman "Submission guidelines" JOEME):
    - Lebar: 84 mm (1 kolom) atau 174 mm (2 kolom); tinggi maks 234 mm
    - Font sans-serif (Arial/Helvetica; fallback DejaVu Sans) 8 pt, judul panel 8 pt bold,
      tick 7 pt. Tidak ada judul gambar di dalam gambar (judul = caption di naskah).
    - Semua label bahasa Inggris, satuan dalam kurung: "SWH (m)", "MWP (s)", "Lead time (h)"
    - Panel diberi label (a), (b), (c) di pojok kiri atas
    - Output: PDF vektor (untuk submit) + PNG 600 dpi (untuk preview/Word)
    - Warna model tetap (tidak berubah antar gambar) + marker berbeda (aman buta warna)
    - Kategori HERI: hijau–kuning–oranye–merah, SELALU disertai label teks

CARA MENGUBAH:
    - Ukuran font: BASE_PT
    - Warna/marker model: MODEL_STYLE
    - Folder output: OUT_DIR
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = Path("figures")
BASE_PT = 8
MM = 1 / 25.4
WIDTH = {'single': 84 * MM, 'double': 174 * MM}

# Okabe–Ito (divalidasi; CVD aman bila disertai marker -> marker wajib dipakai)
MODEL_STYLE = {
    'GRU':         dict(color='#0072B2', marker='o', label='GRU'),
    'XGBoost':     dict(color='#D55E00', marker='s', label='XGBoost'),
    'BiLSTM':      dict(color='#009E73', marker='^', label='BiLSTM'),
    'CNN_LSTM':    dict(color='#E69F00', marker='D', label='CNN-LSTM'),
    'LSTM':        dict(color='#CC79A7', marker='v', label='LSTM'),
    'Persistence': dict(color='#555555', marker='x', label='Persistence', linestyle='--'),
}
MODEL_ORDER = ['GRU', 'LSTM', 'BiLSTM', 'CNN_LSTM', 'XGBoost', 'Persistence']

CAT_LABELS = ['SAFE', 'CAUTION', 'DANGEROUS', 'VERY DANGEROUS']
CAT_COLORS = ['#3B8B3B', '#F2C230', '#E8711A', '#C62828']

SCENARIO_STYLE = {   # skenario predictive HERI
    'S1_perfect':      dict(color='#9E9E9E', hatch='//', label='S1 future MWP/MWD known'),
    'S2_persist':      dict(color='#0072B2', hatch='',   label='S2 SWH forecast + persisted MWP/MWD'),
    'S3_allpredicted': dict(color='#56B4E9', hatch='',   label='S3 all inputs forecast'),
    'S0_allpersist':   dict(color='#555555', hatch='..', label='S0 all persisted (no ML)'),
}


def apply_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': BASE_PT, 'axes.titlesize': BASE_PT, 'axes.labelsize': BASE_PT,
        'xtick.labelsize': BASE_PT - 1, 'ytick.labelsize': BASE_PT - 1,
        'legend.fontsize': BASE_PT - 1, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 1.0, 'lines.markersize': 3.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#E0E0E0', 'grid.linewidth': 0.4,
        'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # font tertanam (syarat penerbit)
    })


def new_fig(width='single', height_mm=60, nrows=1, ncols=1, **kw):
    return plt.subplots(nrows, ncols, figsize=(WIDTH[width], height_mm * MM), **kw)


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, fontweight='bold',
            ha='right', va='bottom')


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'{name}.{ext}')
    plt.close(fig)
    print(f"  saved figures/{name}.pdf + .png")
