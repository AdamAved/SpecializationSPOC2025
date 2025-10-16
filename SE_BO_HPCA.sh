#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
#SBATCH --job-name=SE_BO
#SBATCH --array=0-0
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
cd SE_ERM

srun python SE_BO.py --IntBoundaries 10 --alpha 1 --Damping 0.6 --Nsample 360 --EpsConvergence 1e-3 --Verbose --Debug