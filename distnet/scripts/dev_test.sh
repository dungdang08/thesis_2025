#!/usr/bin/zsh 
############################################################
### Slurm flags
############################################################

#SBATCH --ntasks=1                  # request desired number of processes (or MPI tasks)
#SBATCH --time=00:05:00             # max. run time of the job
#SBATCH --job-name=log_lpg_DistNet  # set the job name
#SBATCH --output=stdout_%j.txt      # redirects stdout and stderr to stdout.txt
#SBATCH --account=<your-project-id>      # insert your project-id or delete this line

############################################################
### Load modules or software
############################################################

# load/activate your desired modules and virtual environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate thesis_venv

############################################################
### Parameters and Settings
############################################################

# print some information about current system
echo "Job nodes: ${SLURM_JOB_NODELIST}"
echo "Current machine: $(hostname)"
nvidia-smi

############################################################
### Execution (Model Training)
############################################################

# run the python script

python eval_model.py --model lognormal_nn.floc --scenario lpg-zeno --fold 0 --seed 100 --save ./TEST_100_nn --num_train_samples 100


