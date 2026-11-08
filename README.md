# Sunda Strait SWH forecasting and the Harbour Entry Risk Index (HERI)

Code, derived data and results for:

> Fauzi MAR, Tarigan TA, Amri U. *Significant Wave Height Forecasting and a Harbour Entry Risk Index (HERI) for the Merak–Bakauheni Ferry Crossing, Sunda Strait: A Rolling-Origin Evaluation Using ERA5.* Submitted to the Journal of Ocean Engineering and Marine Energy (revised version, November 2026).

The numbers in the revised manuscript are produced by the scripts in `analysis/`.
The scripts in `preprocessing/`, `models/` and `evaluation/` belong to the original submission and are kept for reference.

## Data

ERA5 hourly data on single levels (Copernicus Climate Change Service, https://doi.org/10.24381/cds.adbb2d47), 1 Jan 2010 – 31 Dec 2024, area 5.5–7.0°S, 104.5–106.5°E, requested **without** a `grid` keyword, so that wave variables are on the native 0.5° grid and atmospheric variables on 0.25°. See `data/README.md`.

Analysis node P1 = 6.0°S, 105.5°E; neighbouring nodes P2–P5 = (5.5°S, 106.0°E), (6.0°S, 105.0°E), (6.5°S, 105.0°E), (6.5°S, 105.5°E). Values are taken at the exact grid coordinates (no interpolation).

## Software

Python 3.12.3, TensorFlow 2.16.1, Keras 3.14.1, XGBoost 3.2.0 (see `requirements.txt`). Neural networks were trained on CPU.

```
pip install -r requirements.txt
```

## How to reproduce (run from the repository root)

| Step | Script | Output | Manuscript |
|---|---|---|---|
| 1 | `preprocessing/mergeera5_v2.py` | `data/era5_processed/era5_titik*_*.csv` (hourly series at P1–P5) | Sect. 3.2 |
| 2 | `analysis/heri_override.py` | HERI with the regulatory override; operating-range figure | Sect. 3.6, Fig. 7 |
| 3 | `analysis/05_rolling_origin.py` | `data/rolling_origin/` (forecasts per fold, model and seed; metrics) | Sect. 3.5, 4.2, Table 3 |
| 4 | `analysis/06_dm_test.py` | `data/rolling_origin/dm_test.csv` | Sect. 4.2 |
| 5 | `analysis/07_predictive_heri.py` | `data/predictive_heri/` (scenarios S0–S2, sensitivity) | Sect. 4.5, Table 7 |
| 6 | `analysis/08_multioutput_gru.py` | `data/multioutput_gru/` (scenario S3) | Sect. 4.5 |
| 7 | `analysis/09_incidents_and_neighbors.py` | `data/revision_round2/incident_table.csv`, neighbour comparison | Tables 4, 6, S1 |
| 8 | `analysis/10_base_rate.py` | `data/revision_round2/base_rate*.csv` | Table 5, Fig. 8 |
| 9 | `analysis/12_param_sensitivity.py` | `data/revision_round2/param_sensitivity*.csv` | Sect. 4.3, Table S2 |
| 10 | `analysis/11_make_figures.py` | `figures/` (PDF + PNG) | Figs. 3–11, S1 |

Step 3 trains 4 neural networks × 5 seeds × 4 folds plus XGBoost; on a CPU this takes roughly 1–2 days. Existing runs are skipped, so the provided forecasts in `data/rolling_origin/` can be used directly for steps 4–10.

## Contents of `data/rolling_origin/fold_<year>/<model>_seed_<seed>/`

- `y_pred.npy`, `y_true.npy`: forecasts and ERA5 targets (m), shape (n, 3) for lead times 6, 12, 24 h
- `test_timestamps.csv`: forecast origin times (UTC)
- `train_info.json`: training time, number of epochs and epoch with the lowest validation loss

## Incident catalogue

`data/revision_round2/incident_table.csv` (16 events, 2015–2024) and Table S1 of the manuscript list dates, windows (WIB and UTC), locations, sources and ERA5/HERI values. The catalogue is compiled from media reports and is incomplete.

## License

Code: MIT (see `LICENSE`). ERA5-derived data: Copernicus licence; please cite Hersbach et al. (2020) and the CDS dataset DOI above.

## Citation

To be updated after publication.
