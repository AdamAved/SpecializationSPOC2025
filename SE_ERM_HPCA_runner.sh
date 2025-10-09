#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
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

alphamin=-1
alphamax=2

srun python SE_ERM_HPCA.py --BatchSize 100 --nBatch ${SLURM_ARRAY_TASK_ID} --alphaMin $alphamin --alphaMax $alphamax --Damping 0.6 --Nsample 360000 --EpsConvergence 5e-4