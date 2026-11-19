import os
import submitit

SCENARIOS = []
# SCENARIOS += ["saps-CVVAR", ] * 10
SCENARIOS += ["spear_qcp", ] * 1
# SCENARIOS += ["spear_swgcp", ] * 3
# SCENARIOS += ["yalsat_qcp", ] * 6
# SCENARIOS += ["yalsat_swgcp", ] * 2
# SCENARIOS += ["lpg-zeno", ] * 10
# SCENARIOS += ["clasp_factoring", ] * 10

FOLDS = []
FOLDS += [8]


MODEL = "weibull"
# Available models:
    # "invgauss"
    # "lognormal"
    # "expon"
    # "weibull"


def main():
    def run_eval(scenario, fold):
        env_cmd = (
            "export CONDA_ROOT=$HOME/miniforge3 && "
            "source $CONDA_ROOT/etc/profile.d/conda.sh && "
            "export PATH='$CONDA_ROOT/bin:$PATH' &&"
            "conda activate xgb_venv"
        )
        cmd = (
            f"python eval_model.py "
            f"--model {MODEL}_xgb.floc "
            f"--scenario {scenario} "
            f"--fold {fold} "
            f"--seed 100 "
            f"--save ./TEST_100_SMAC_NAS_100 "
            f"--num_train_samples 100 "
        )
        full_cmd = f"{env_cmd} && {cmd}"
        print(f"Running: {full_cmd}")
        os.system(full_cmd)

    log_folder = "log_test_SMAC_NAS_WEI/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    num_cpus = 10
    executor.update_parameters(
        cpus_per_task=num_cpus,
        # mem_gb=5.5 * num_cpus,  # 5 cpus in c23mm (5210 MiB) partition
        timeout_min=60*24, # 100 hours
        slurm_job_name= f"xgb_wei",
        slurm_account="<your-slurm-account>",
        slurm_partition="c23mm",
        slurm_array_parallelism=len(SCENARIOS),
    )
    jobs = executor.map_array(
        run_eval,
        SCENARIOS,
        FOLDS
    )


if __name__ == "__main__":
    main()