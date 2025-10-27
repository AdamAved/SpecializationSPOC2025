#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
#SBATCH --job-name=SE_ERM
#SBATCH --array=0-19
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

alpha=20
lambdamin=-1
lambdamax=1

srun python SE_ERM_HPCA_lambdascan.py --BatchSize 20 --nBatch ${SLURM_ARRAY_TASK_ID} --alpha $alpha --LambdaMin $lambdamin --LambdaMax $lambdamax --Damping 0.6 --Nsample 180000 --EpsConvergence 1e-3