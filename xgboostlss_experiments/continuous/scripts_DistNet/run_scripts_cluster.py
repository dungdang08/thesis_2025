import os
import submitit

MODELS=(
    "invgauss",
    # "lognormal",
    # "expon",
    # "weibull",
)

ALGOS = ["DE", "DiagonalCMA", "modcma", "modde", "NaiveIsoEMNA", "NGOpt14", "NGOpt38", "OnePlusOne", "PSO", "RandomSearch", "RCobyla"] * 2
DIMS =  [5] * 11 + [20] * 11
BUDGETS = [100, 250, 500, 1000] # , 2500, 5000, 10000]
FOLDS = list(range(10))

def main():
    for budget in BUDGETS:
        def run_eval(scenario, dim):
            env_cmd = (
                "export CONDA_ROOT=$HOME/miniforge3 && "
                "source $CONDA_ROOT/etc/profile.d/conda.sh && "
                "export PATH='$CONDA_ROOT/bin:$PATH' &&"
                "conda activate xgb_venv"
            )
            # for budget in BUDGETS:
            for fold in FOLDS:
                cmd = (
                    f"python eval_model.py "
                    f"--model invgauss_xgb.floc "
                    f"--scenario {scenario} "
                    f"--budget {budget} "
                    f"--dim {dim} "
                    f"--fold {fold} "
                    f"--seed 50 "
                    f"--save ./TEST_50_SMAC_NAS_100 "
                    f"--num_train_samples 50"
                )
                full_cmd = f"{env_cmd} && {cmd}"
                print(f"Running: {full_cmd}")
                os.system(full_cmd)

        log_folder = "log_test_SMAC_NAS_INV/%j"
        executor = submitit.AutoExecutor(folder=log_folder)
        num_cpus = 1
        executor.update_parameters(
            cpus_per_task=num_cpus,
            # mem_gb=5.5 * num_cpus,
            timeout_min=60*24, # 50 hours
            slurm_job_name=f"inv_{budget}",
            slurm_account="<your-slurm-account>",
            # slurm_partition="c23mm",
            slurm_array_parallelism=len(ALGOS),  # Adjust based on the number of jobs
        )
        jobs = executor.map_array(
            run_eval,
            ALGOS,
            DIMS
        )


if __name__ == "__main__":
    main()