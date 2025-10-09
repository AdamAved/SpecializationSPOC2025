#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/Specialization2025
#SBATCH --job-name=test
#SBATCH --array=0-127
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
cd SE_ERM

srun python test.py 
mpiexec -n 4 python SE_ERM_HPCA.py --BatchSize 100 --nBatch 0 --alphaMin -1 --alphaMax 2 --Damping 0.6 --Nsample 360000 --EpsConvergence 5e-4