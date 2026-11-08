# Data

## ERA5 download (Copernicus CDS)

Dataset `reanalysis-era5-single-levels`, product type `reanalysis`, hourly, 2010-01-01 to 2024-12-31,
area [N, W, S, E] = [-5.5, 104.5, -7.0, 106.5], format NetCDF, **no `grid` keyword** (native grids).

- Wave variables (0.5° grid): `significant_height_of_combined_wind_waves_and_swell`, `mean_wave_period`, `mean_wave_direction`
- Atmospheric variables (0.25° grid): `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`

Raw monthly files go to `data/era5_raw/` (not included; download with the CDS API).
`preprocessing/mergeera5_v2.py` extracts the series at P1–P5 to `data/era5_processed/`.

## Included derived data

- `era5_processed/` – hourly series at P1–P5 (ERA5-derived; Copernicus licence)
- `rolling_origin/`, `predictive_heri/`, `multioutput_gru/`, `revision_round2/` – results of the scripts in `analysis/`

All times are UTC. Incident times in the catalogue are given in WIB (UTC + 7 h) and UTC.
