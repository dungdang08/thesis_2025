# Continuous black-box optimisation experiments

Applies DistNet and forest baselines to performance data of continuous optimisation algorithms.

## Attribution

- `src_DistNet/`, `helper_DistNet/`, `scripts_DistNet/`, `notebooks/` — adapted from **DistNet**
  (Eggensperger, Lindauer & Hoos, IJCAI 2018, <https://github.com/KEggensperger/DistNet>; no license
  stated, rights remain with its authors). My changes: continuous-data loader
  (`helper_DistNet/load_data_continuous.py`), TF2 compatibility, Weibull distribution, cluster launchers.
- `train_RForest_regression_performance_prediction_models.py` and
  `train_XGBoost_regression_performance_prediction_models.py` — written by **Ana Kostovska et al.**
  for their GECCO 2023 work and shared with me privately; published here with their permission.
  My only change: hard-coded cluster paths replaced by the `DATA_DIR` / `RESULTS_DIR` environment
  variables (default `./data`, `./results`).

The underlying data is not included. Please cite the papers in
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
