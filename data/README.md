# Significant Wave Height Prediction and HERI
## Sunda Strait (Merak-Bakauheni Ferry Corridor)

Code for the paper:
> Aldhi et al. (2025). *Significant Wave Height Prediction and
> Harbour Entry Risk Index (HERI) in the Sunda Strait using
> ERA5-Based Deep Learning.* [Journal name]

## Data
ERA5 reanalysis data (2010–2024) downloaded via Copernicus CDS API.
- Grid point: −6.0°S, 105.5°E
- Variables: SWH, U10, V10, MWP, MWD, MSLP
- See `data/README_data.md` for download instructions.

## Requirements
```
pip install -r requirements.txt
```

## Usage
1. Download ERA5 data (see data/)
2. Run preprocessing: `python preprocessing/mergeera5_v2.py`
3. Train models: `python models/train_models.py`
4. Evaluate: `python evaluation/04_evaluate.py`

## Citation
[To be updated after publication]
