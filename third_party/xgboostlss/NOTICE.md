# NOTICE — modified copy of XGBoostLSS

This directory contains a **modified** copy of XGBoostLSS v0.4.0
(<https://github.com/StatMixedML/XGBoostLSS>), Copyright (c) Alexander März,
licensed under the Apache License 2.0 (see [LICENSE.txt](LICENSE.txt)). The upstream README is kept as
[README.md](README.md).

Changes made by Hoang Dung Dang for a bachelor thesis (2025), as required by Apache-2.0 §4(b):

- **Removed** `docs/`, `tests/`, `figures/`, `mkdocs.yml`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`
  and the `xgboostlss/datasets/` example data (not needed for the thesis experiments).
- **Added** `xgboostlss/distributions/Exponential.py` — Exponential distribution class.
- **Added** `xgboostlss/distributions/InverseGaussian.py` — Inverse Gaussian distribution class.
- **Added** `xgboostlss/distributions/inverse_gaussian_torch.py` — PyTorch Inverse Gaussian
  distribution. Its sampling routine is ported from TensorFlow Probability
  (`tensorflow_probability/python/distributions/inverse_gaussian.py`, Copyright 2018 The TensorFlow
  Probability Authors, Apache-2.0).
- **Modified** `xgboostlss/distributions/__init__.py` to import the new distributions.

All other files are unchanged from upstream.
