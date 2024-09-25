#!/bin/bash
  
#SBATCH --job-name=test
#SBATCH --time=29:59
#SBATCH --qos=30min
#SBATCH --mem=3G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./SLURM/slurm-%A_%a.out
##SBATCH --error=./SLURM/error-%A_%a.out
#SBATCH --array=1-130051%3000

mkdir -p ./SLURM

$(head -$SLURM_ARRAY_TASK_ID SLURM_list_comands.txt | tail -1)