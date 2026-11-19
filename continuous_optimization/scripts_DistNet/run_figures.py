# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import sys

import numpy as np

sys.path.append("../")
from helper_DistNet import load_data_continuous


ALGOS = ["DE", "DiagonalCMA", "modcma", "modde", "NaiveIsoEMNA", "NGOpt14", "NGOpt38", "OnePlusOne", "PSO", "RandomSearch", "RCobyla"]
DIMS =  [5, 20]
BUDGETS = [100, 250, 500, 1000, 2500, 5000, 10000]
RUNS = 50
FOLDS = list(range(10))

# 3) Load data
with open("training_data_num.txt", "w") as f_out:
    for scenario in ALGOS:
        for budget in BUDGETS:
            for dim in DIMS:
                target_precs, features, _ = load_data_continuous.get_data(
                    algo=scenario, budget=budget, dim=dim,
                    runs=RUNS, 
                    agregate_runs=False, 
                    log_target=False
                )
                features = np.array(features)
                target_precs = np.array(target_precs)
                f_out.write(f"\n\nScenario: {scenario}, budget: {budget}, dim: {dim}\n")
                f_out.write(f"runtimes: {target_precs.shape}, features: {features.shape}\n")
