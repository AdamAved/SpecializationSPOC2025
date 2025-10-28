#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
#SBATCH --job-name=SE_BO
#SBATCH --array=0-61
#SBATCH --output=SE_BO_HPCA.out
#SBATCH --error=SE_BO_HPCA.err
#SBATCH --ntasks=36
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=10:00:00

# Load all modules
module load gcc
module load python
module load openmpi

# Activate the environment
source myenv/bin/activate
read alpha lambda0 lambda1 <<< $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Params_Alpha_Scan.txt)
cd SE_BO

srun python SE_BO_MC_Integral.py --IntSamples 1000 --alpha $alpha --Damping 0.6 --Nsample 3600 --MaxIter 50 --EpsConvergence 1e-6