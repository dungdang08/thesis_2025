import os

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

# Test Exponential Distribution
# for SC in SCENARIOS:
#     for fold in FOLDS:
#         cmd = (
#             f"python eval_model.py "
#             f"--model expon_xgb.floc "
#             f"--scenario {SC} "
#             f"--fold {fold} "
#             f"--seed 100 "
#             f"--save ./TEST_100 "
#             f"--num_train_samples 100 "
#         )
#         print(f"Running: {cmd}")
#         os.system(cmd)



# Test Inverse Gaussian Distribution
for SC in SCENARIOS:
    for fold in FOLDS:
        cmd = (
            f"python eval_model.py "
            f"--model invgauss_xgb.floc "
            f"--scenario {SC} "
            f"--fold {fold} "
            f"--seed 100 "
            f"--save ./TEST_100_noHPO/TEST_100_INV_50 "
            f"--num_train_samples 100 "
        )
        print(f"Running: {cmd}")
        os.system(cmd)



# for SC in SCENARIOS:
#     for fold in FOLDS:
#         cmd = (
#             f"python eval_model.py "
#             f"--model lognormal_xgb.floc "
#             f"--scenario {SC} "
#             f"--fold {fold} "
#             f"--seed 100 "
#             f"--save ./TEST_100_noHPO "
#             f"--num_train_samples 100 "
#         )
#         print(f"Running: {cmd}")
#         os.system(cmd)

# cmd = (
#     f"python eval_model.py "
#     f"--model lognormal_xgb.floc "
#     f"--scenario clasp_factoring "
#     f"--fold 0 "
#     f"--seed 100 "
#     f"--save ./TEST_100_TEST "
#     f"--num_train_samples 100 "
# )
# print(f"Running: {cmd}")
# os.system(cmd)