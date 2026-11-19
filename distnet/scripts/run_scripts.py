# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import os


INSTANCES_NUMS =  [20, 50, 100, 200, 500, 1000]
INSTANCE_PERCS =  [10, 20, 50, 70, 100]
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

for scenario in SCENARIOS:
    for fold in FOLDS:
        cmd = (
            f"python eval_model.py "
            f"--model weibull_distfit.floc "
            f"--scenario {scenario} "
            f"--fold {fold} "
            f"--seed 100 "
            f"--save ./TEST_100 "
            f"--num_train_samples 100 "
        )
        print(f"Running: {cmd}")
        os.system(cmd)
        cmd = (
            f"python eval_model.py "
            f"--model invgauss_distfit.floc "
            f"--scenario {scenario} "
            f"--fold {fold} "
            f"--seed 100 "
            f"--save ./TEST_100 "
            f"--num_train_samples 100 "
        )
        print(f"Running: {cmd}")
        os.system(cmd)


# for scenario in SCENARIOS:
#     for fold in FOLDS:
#         for instances_num in INSTANCES_NUMS:
#             cmd = (
#                 f"python eval_model_cutInst.py "
#                 f"--model lognormal_distfit.floc "
#                 f"--scenario {scenario} "
#                 f"--fold {fold} "
#                 f"--seed 100 "
#                 f"--save ./TEST_100_cutInstances/NUM_{instances_num} "
#                 f"--num_train_samples 100 "
#                 f"--instances_num {instances_num}"
#             )
#             print(f"Running: {cmd}")
#             os.system(cmd)


# for scenario in SCENARIOS:
#     for fold in FOLDS:
#         for instance_perc in INSTANCE_PERCS:
#             cmd = (
#                 f"python eval_model_cutPercInst.py "
#                 f"--model lognormal_distfit.floc "
#                 f"--scenario {scenario} "
#                 f"--fold {fold} "
#                 f"--seed 100 "
#                 f"--save ./TEST_100_cutPercInst/PERC_{instance_perc} "
#                 f"--num_train_samples 100 "
#                 f"--instance_perc {instance_perc}"
#             )
#             print(f"Running: {cmd}")
#             os.system(cmd)