#!/bin/bash
#SBATCH --chdir /home/avedissi/PDM2025/SpecializationSPOC2025
#SBATCH --job-name=Zipping
#SBATCH --array=0-0
#SBATCH --output=Zipping.out
#SBATCH --error=Zipping.err
#SBATCH --ntasks=1
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --time=02:00:00

# Activate the environment
cd GD_ERM

srun zip GD_ERM_AlphaScan *.mat