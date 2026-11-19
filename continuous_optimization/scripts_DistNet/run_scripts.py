# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import os

MODELS = [
    "invgauss_distfit.floc",
    "invgauss_lbfgs.floc",
    "lognormal_distfit.floc",
    "lognormal_lbfgs.floc",
    "expon_distfit.floc",
    "expon_lbfgs.floc",
    "weibull_distfit.floc", # added weibull
    "weibull_lbfgs.floc", # added weibull
    "invgauss_nn.floc",
    "lognormal_nn.floc",
    "expon_nn.floc",
    "weibull_nn.floc"
    ]
ALGOS = ["DE", "DiagonalCMA", "modcma", "modde", "NaiveIsoEMNA", "NGOpt14", "NGOpt38", "OnePlusOne", "PSO", "RandomSearch", "RCobyla"]
DIMS =  [5, 20]
BUDGETS = [100, 250, 500, 1000] # , 2500, 5000, 10000]
FOLDS = list(range(10))

for scenario in ALGOS:
    for dim in DIMS:
        for budget in BUDGETS:
            for fold in FOLDS:
                cmd = (
                    f"python eval_model.py "
                    f"--model invgauss_distfit.floc "
                    f"--scenario {scenario} "
                    f"--budget {budget} "
                    f"--dim {dim} "
                    f"--fold {fold} "
                    f"--seed 50 "
                    f"--save ./TEST_50_distfit "
                    f"--num_train_samples 50"
                )
                print(f"Running: {cmd}")
                os.system(cmd)

                # cmd = (
                #     f"python eval_model.py "
                #     f"--model weibull_distfit.floc "
                #     f"--scenario {scenario} "
                #     f"--budget {budget} "
                #     f"--dim {dim} "
                #     f"--fold {fold} "
                #     f"--seed 50 "
                #     f"--save ./TEST_50_distfit "
                #     f"--num_train_samples 50"
                # )
                # print(f"Running: {cmd}")
                # os.system(cmd)
