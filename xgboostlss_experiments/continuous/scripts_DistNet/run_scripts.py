import os

MODELS = [
    # "invgauss_xgb.floc", not yet supported
    # "lognormal_xgb.floc",
    # "expon_xgb.floc", not yet supported
    "weibull_xgb.floc",
    ]
ALGOS = ["DE", "DiagonalCMA", "modcma", "modde", "NaiveIsoEMNA", "NGOpt14", "NGOpt38", "OnePlusOne", "PSO", "RandomSearch", "RCobyla"]
# ALGOS = ["RandomSearch", "NGOpt14", "NGOpt38", "RCobyla"]
DIMS =  [5, 20]
BUDGETS = [100, 250, 500, 1000] # , 2500, 5000, 10000]
RUNS = 50
FOLDS = list(range(1))

for fold in FOLDS:
    cmd = (
        f"python eval_model.py "
        f"--model weibull_xgb.floc "
        f"--scenario DE "
        f"--budget 100 "
        f"--dim 5 "
        f"--fold {fold} "
        f"--seed 50 "
        f"--save ./TEST_50_SMAC_NAS_100 "
        f"--num_train_samples 50"
    )
    print(f"Running: {cmd}")
    os.system(cmd)

            
# for scenario in ALGOS:
#     for dim in DIMS:
#         for budget in BUDGETS:
#             for fold in FOLDS:
#                 cmd = (
#                     f"python eval_model.py "
#                     f"--model lognormal_xgb.floc "
#                     f"--scenario {scenario} "
#                     f"--budget {budget} "
#                     f"--dim {dim} "
#                     f"--fold {fold} "
#                     f"--seed 50 "
#                     f"--save ./TEST_50_noHPO "
#                     f"--num_train_samples 50"
#                 )
#                 print(f"Running: {cmd}")
#                 os.system(cmd)
                
#                 cmd = (
#                     f"python eval_model.py "
#                     f"--model weibull_xgb.floc "
#                     f"--scenario {scenario} "
#                     f"--budget {budget} "
#                     f"--dim {dim} "
#                     f"--fold {fold} "
#                     f"--seed 50 "
#                     f"--save ./TEST_50 "
#                     f"--num_train_samples 50"
#                 )
#                 print(f"Running: {cmd}")
#                 os.system(cmd)