# Adapted from Bayes DistNet (https://github.com/tuero/BayesianDistNet),
# Tuero & Buro, AAAI 2021. Modified by Hoang Dung Dang (thesis student); see ../ATTRIBUTION.md.

import os
import argparse


SCENARIOS = [
    "clasp_factoring",
    "saps-CVVAR",
    "spear_qcp",
    "yalsat_qcp",
    "spear_swgcp",
    "yalsat_swgcp",
    "lpg-zeno",
]
MODELS = [
    "BAYESIAN_LOGNORMAL",
    "BAYESIAN_WEIBULL",
    "BAYESIAN_INVGAUSS",
    "BAYESIAN_EXPONENTIAL",
    "BAYESIAN_NORMAL",
]

seeds = [0, 1] # , 1, 2, 3, 4]
lbs = [0] # , 20, 40, 60, 80]
sample_counts = [16] # [1, 2, 4, 8, 16]
folds = list(range(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", required=True, choices=MODELS)
    parser.add_argument("--scenario", dest="scenario", required=True, choices=SCENARIOS)
    args = parser.parse_args()
    for fold in folds:
        for seed in seeds:
            for lb in lbs:
                for sample_count in sample_counts:
                    cmd = (
                            f"python eval_model.py "
                            f"--scenario {args.scenario} "
                            f"--num_train_samples {sample_count} "
                            f"--fold {fold} "
                            f"--seed {seed} "
                            f"--net_type bayes_distnet "
                            f"--config_section {args.model} "
                            f"--lb {lb} "
                            )
                    print(f"Running: {cmd}")
                    os.system(cmd)


if __name__ == "__main__":
    main()