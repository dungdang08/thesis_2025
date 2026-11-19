# Attribution — DistNet

This folder is a modified copy of **DistNet**, <https://github.com/KEggensperger/DistNet>,
the code accompanying:

> K. Eggensperger, M. Lindauer, H. H. Hoos. *Neural Networks for Predicting Algorithm Runtime
> Distributions*. IJCAI 2018. doi:10.24963/ijcai.2018/200

The original repository states no license; all rights to the original code remain with its authors.
The upstream README is kept as [README.md](README.md). Please cite the paper above (BibTeX in
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## My modifications (Hoang Dung Dang, bachelor thesis)

- `scripts/eval_model.py`
  - import `tensorflow.python.keras.backend` instead of `keras.backend`; use `tf.compat.v1` calls (TensorFlow 2 compatibility)
  - added the **Weibull** distribution (parameter choice, NN, RF/RF2, distfit, L-BFGS)
- `helper/dist_helper.py`: pass `dist.pdf` instead of `dist.cdf` for the Kolmogorov–Smirnov test of `weibull_min`
- `scripts/run_scripts*.py`, `scripts/dev_test.sh`: SLURM/submitit launchers for the RWTH cluster
- `notebooks/`: additional analysis notebooks (`*-all`, `*_cutInst`, `Kolmogorov-Smirnov`, `VisualizeFit`)
