# XGBoostLSS experiments

Runtime-distribution prediction with XGBoostLSS (my thesis contribution).

- `discrete/` — algorithm runtime distributions on the DistNet scenarios (SAT, CSP, planning).
- `continuous/` — the same pipeline on continuous black-box optimisation data.

Both use the modified XGBoostLSS in [`../third_party/xgboostlss`](../third_party/xgboostlss)
(imported via `sys.path`, no installation needed).

## Attribution

The data loading, preprocessing and evaluation code (`helper*/`, `scripts*/eval_model.py`,
`notebooks/CreateTable_*`) is adapted from **DistNet** (Eggensperger, Lindauer & Hoos, IJCAI 2018,
<https://github.com/KEggensperger/DistNet>; no license stated, rights remain with its authors).
The XGBoostLSS models, distribution choices, SHAP analysis (`discrete/notebooks/visualizeSHAP.ipynb`)
and cluster launchers are mine. See [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for BibTeX.
