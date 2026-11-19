# Third-party code and citations

Large parts of this repository are **adapted from code released with other papers**. That code is
**not** my work and remains under its original terms (or, where no license was given, under the
original authors' copyright). My own additions and modifications are listed in each folder's
`ATTRIBUTION.md`. If you use any of these parts, please cite the original papers below.

| Folder in this repo | Original source | License | Paper |
|---|---|---|---|
| [`distnet/`](distnet/) | [KEggensperger/DistNet](https://github.com/KEggensperger/DistNet) | none stated (all rights reserved by the authors) | Eggensperger, Lindauer & Hoos, IJCAI 2018 |
| [`bayes_distnet/`](bayes_distnet/) | [tuero/BayesianDistNet](https://github.com/tuero/BayesianDistNet) | none stated (all rights reserved by the authors) | Tuero & Buro, AAAI 2021 |
| [`bayes_distnet/src/layers/`](bayes_distnet/src/layers/) (`BBBLinear*.py`, `misc.py`) | [kumar-shridhar/PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) (via Bayes DistNet) | MIT | Shridhar, Laumann & Liwicki, 2019 |
| [`third_party/xgboostlss/`](third_party/xgboostlss/) | [StatMixedML/XGBoostLSS](https://github.com/StatMixedML/XGBoostLSS) v0.4.0 (trimmed, modified) | Apache-2.0 ([LICENSE.txt](third_party/xgboostlss/LICENSE.txt)) | März 2019; März & Kneib 2022 |
| `third_party/xgboostlss/xgboostlss/distributions/inverse_gaussian_torch.py` (`rsample`) | ported from [TensorFlow Probability](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/inverse_gaussian.py) | Apache-2.0 | Dillon et al., 2017 |
| XGBoost search space in `xgboostlss_experiments/*/eval_model.py` | [NAS-Bench-301 `xgb_configspace.json`](https://github.com/automl/nasbench301) | hyperparameter ranges only (no code copied) | Zela et al., ICLR 2022 |
| [`xgboostlss_experiments/`](xgboostlss_experiments/) | experiment scripts built on DistNet's `helper/` and `scripts/` | as DistNet | Eggensperger et al. 2018 |
| [`continuous_optimization/`](continuous_optimization/) | DistNet scripts + training scripts shared privately by A. Kostovska (GECCO 2023 work), used with permission | as original authors | Kostovska et al., GECCO 2023 |

Data is **not** included. The algorithm-runtime data comes from the DistNet release
(<http://www.ml4aad.org/wp-content/uploads/2018/04/DistNetData.zip>).

## BibTeX

```bibtex
@inproceedings{ijcai2018p200,
  title     = {Neural Networks for Predicting Algorithm Runtime Distributions},
  author    = {Katharina Eggensperger and Marius Lindauer and Frank Hutter},
  booktitle = {Proceedings of the 27th International Joint Conference on
               Artificial Intelligence (IJCAI 2018)},
  pages     = {1442--1448},
  year      = {2018},
  month     = {7},
}

@article{shridhar2019comprehensive,
  title={A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1901.02731},
  year={2019}
}

@inproceedings{tuero2021bayes,
  title={Bayes DistNet -- A Robust Neural Network for Algorithm Runtime Distribution Predictions},
  author={Tuero, Jake and Buro, Michael},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021)},
  volume={35},
  number={14},
  pages={12418--12426},
  year={2021},
}

@misc{maerz2019xgboostlss,
  title         = {{XGBoostLSS} -- An extension of {XGBoost} to probabilistic forecasting},
  author        = {Alexander M\"arz},
  year          = {2019},
  eprint        = {1907.03178},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

@misc{maerz2022dgbm,
  title         = {Distributional Gradient Boosting Machines},
  author        = {Alexander M\"arz and Thomas Kneib},
  year          = {2022},
  eprint        = {2204.00778},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{Maerz2023,
  author = {Alexander M\"arz},
  title = {{XGBoostLSS: An Extension of XGBoost to Probabilistic Modelling}},
  year = {2023},
  note = {GitHub repository, Version 0.4.0},
  howpublished = {\url{https://github.com/StatMixedML/XGBoostLSS}}
}

@misc{dillon2017tfdistributions,
  title         = {{TensorFlow} Distributions},
  author        = {Joshua V. Dillon and Ian Langmore and Dustin Tran and Eugene Brevdo and Srinivas Vasudevan and Dave Moore and Brian Patton and Alex Alemi and Matt Hoffman and Rif A. Saurous},
  year          = {2017},
  eprint        = {1711.10604},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@inproceedings{zela2022surrogate,
  title     = {Surrogate {NAS} Benchmarks: Going Beyond the Limited Search Spaces of Tabular {NAS} Benchmarks},
  author    = {Arber Zela and Julien Niklas Siems and Lucas Zimmer and Jovita Lukasik and Margret Keuper and Frank Hutter},
  booktitle = {International Conference on Learning Representations (ICLR 2022)},
  year      = {2022}
}

@inproceedings{Kostovska_2023, 
   title={Comparing Algorithm Selection Approaches on Black-Box Optimization Problems},
   booktitle={Proceedings of the Companion Conference on Genetic and Evolutionary Computation (GECCO 2023)},
   author={Kostovska, Ana and Jankovic, Anja and Vermetten, Diederick and D{\v{z}}eroski, Sa{\v{s}}o and Eftimov, Tome and Doerr, Carola},
   publisher={ACM},
   year={2023},
   month={7},
   pages={495--498},
}
```
