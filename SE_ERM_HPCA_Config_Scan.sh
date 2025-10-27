#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
#SBATCH --job-name=SE_ERM
#SBATCH --array=0-0
#SBATCH --output=SE_ERM_HPCA.out
#SBATCH --error=SE_ERM_HPCA.err
#SBATCH --ntasks=36
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=01:00:00

# Load all modules
module load gcc
module load python
module load openmpi

# Activate the environment
source myenv/bin/activate
read alpha lambda0 lambda1 <<< $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Params_Alpha_Scan.txt)
cd SE_ERM

srun python SE_ERM.py --alpha $alpha --Lambda0 $lambda0 --Lambda1 $lambda1 --Damping 0.6 --Nsample 180000 --EpsConvergence 1e-6 --MaxIter 50