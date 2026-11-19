# Predicting Algorithm Performance Distribution using XGBoost

Code for my bachelor thesis **"Predicting Algorithm Performance Distribution using XGBoost"**, Chair for AI Methodology (AIM),
RWTH Aachen University. The thesis is in [`thesis/Bachelor_Thesis_Final.pdf`](thesis/Bachelor_Thesis_Final.pdf).

The thesis compares models that predict the full **runtime distribution** of an algorithm on a
problem instance: DistNet, Bayes DistNet, random-forest baselines and distributional gradient
boosting (XGBoostLSS), on discrete (SAT/CSP/planning) and continuous black-box optimisation data.

> **Most of the model code is adapted from other papers' repositories.** See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and each folder's `ATTRIBUTION.md` for what was
> taken from where and what I changed. Please cite the original authors.

## Repository structure

| Folder | Content | Based on |
|---|---|---|
| [`distnet/`](distnet/) | DistNet + RF baselines on the discrete scenarios, extended with a Weibull distribution | [DistNet](https://github.com/KEggensperger/DistNet) (Eggensperger et al., IJCAI'18) |
| [`bayes_distnet/`](bayes_distnet/) | Bayes DistNet with fixes for current library versions | [Bayes DistNet](https://github.com/tuero/BayesianDistNet) (Tuero & Buro, AAAI'21) |
| [`xgboostlss_experiments/`](xgboostlss_experiments/) | XGBoostLSS runtime-distribution models, discrete + continuous | DistNet pipeline + XGBoostLSS |
| [`continuous_optimization/`](continuous_optimization/) | DistNet / forests on continuous black-box optimisation data | DistNet + Kostovska et al. (GECCO'23) |
| [`third_party/xgboostlss/`](third_party/xgboostlss/) | trimmed XGBoostLSS v0.4.0 with added Exponential & Inverse Gaussian distributions | [XGBoostLSS](https://github.com/StatMixedML/XGBoostLSS) (Apache-2.0) |
| [`plots/`](plots/) | distribution plots used in the thesis | mine |
| [`thesis/`](thesis/) | final thesis PDF | mine |

## Setup

The sub-projects use different deep-learning stacks (TensorFlow/Keras for DistNet, PyTorch for
Bayes DistNet and XGBoostLSS), so a separate virtual environment per sub-project is recommended.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Data is not included. Download the DistNet data
(<http://www.ml4aad.org/wp-content/uploads/2018/04/DistNetData.zip>) and extract it into `data/`
inside the sub-project you want to run (e.g. `distnet/data/`).

## Usage examples

```bash
# DistNet (run from distnet/scripts)
python eval_model.py --model lognormal_nn.floc --scenario clasp_factoring --fold 0 --seed 100 --save ./TEST_100 --num_train_samples 100

# Bayes DistNet (run from bayes_distnet/src)
python eval_model.py --fold 1 --scenario lpg-zeno --config_section bayesian_lognormal --seed 1 --num_train_samples 8 --lb 20 --net_type bayes_distnet

# XGBoostLSS, discrete scenarios (run from xgboostlss_experiments/discrete/scripts)
python eval_model.py --help
```

Results are analysed with the notebooks in each sub-project's `notebooks/` folder
(`CreateTable_*` for NLLH tables, `PlotSubsets_*` for plots). SLURM launchers
(`run_scripts_cluster*.py`) need your own account in `slurm_account`.

## Citation

If you use code from this repository, please cite the original papers listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and my thesis if relevant.

## License

- **Third-party code** keeps its original license / copyright — see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (XGBoostLSS: Apache-2.0; PyTorch-BayesianCNN
  layers: MIT; DistNet and Bayes DistNet: no license stated by their authors).
- **My own code and the thesis**: © Hoang Dung Dang, all rights reserved. No license is granted;
  the code is published for transparency and reproducibility of the thesis. Contact me if you want
  to reuse it.

## Acknowledgements

Thanks to Prof. Dr. rer. nat. Holger Hoos, Hadar Shavit M.Sc, and Dr. Anja Jankovic at the Chair for AI Methodology at RWTH Aachen University, and to the authors of DistNet,
Bayes DistNet, XGBoostLSS, PyTorch-BayesianCNN and Ana Kostovska et al. for making their code available.
