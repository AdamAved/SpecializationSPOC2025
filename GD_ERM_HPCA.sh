#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
#SBATCH --job-name=SE_ERM
#SBATCH --array=0-199
#SBATCH --output=GD_ERM_HPCA.out
#SBATCH --error=GD_ERM_HPCA.err
#SBATCH --ntasks=1
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --time=02:00:00

# Load all modules
module load gcc
module load python

# Activate the environment
source myenv/bin/activate
read alpha lambda0 lambda1 reps<<< $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Params_Alpha_Scan_GD.txt)
cd GD_ERM

srun python GD_ERM.py --dim 2000 --Nrep $reps --alpha $alpha --Lambda0 $lambda0 --Lambda1 $lambda1 --EpsConvergence 1e-10 --MaxIter 1000000