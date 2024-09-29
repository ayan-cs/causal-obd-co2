#!/bin/bash
# Job name:
#SBATCH --job-name=real

# Partition:
#SBATCH --partition=phd
#SBATCH --nodes=1

#SBATCH --ntasks=1
## Processors per task:

#SBATCH --cpus-per-task=1
#
#SBATCH --gres=gpu:1
#
## Command(s) to run :
module load python/3.10.pytorch
mpirun python3 /csehome/p23iot002/causal-obd-co2/main_obd.py