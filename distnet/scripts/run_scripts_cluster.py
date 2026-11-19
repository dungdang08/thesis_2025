# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import os
import argparse

FOLDS = list(range(10))
SCENARIOS = [
    "clasp_factoring",
    "saps-CVVAR",
    "spear_qcp",
    "yalsat_qcp",
    "spear_swgcp",
    "yalsat_swgcp",
    "lpg-zeno",
]
MODELS=(
    "invgauss_distfit.floc",
    "lognormal_distfit.floc",
    "expon_distfit.floc",
    "weibull_distfit.floc", # added weibull
    "invgauss_nn.floc",
    "lognormal_nn.floc",
    "expon_nn.floc",
    "weibull_nn.floc",
) # added weibull

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", required=True, choices=MODELS)
    parser.add_argument("--scenario", dest="scenario", required=True, choices=SCENARIOS)
    args = parser.parse_args()
    for fold in FOLDS:
        cmd = (
                f"python eval_model.py "
                f"--model {args.model} "
                f"--scenario {args.scenario} "
                f"--fold {fold} "
                f"--seed 100 "
                f"--save ./TEST_100 "
                f"--num_train_samples 100 "
                )
        print(f"Running: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    main()