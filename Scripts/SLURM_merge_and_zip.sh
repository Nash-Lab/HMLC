#!/bin/bash
  
#SBATCH --job-name=merge_and_zip
#SBATCH --time=23:59:00
#SBATCH --qos=1day
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./slurm-%A.out

path=../Results/2D/
path_save_tar=../Results/

pattern=uni_2D

conda run --no-capture-output -n Process4DataAnalysis python3 -u ./04_ML_merge_data.py --path ${path} --pattern ${pattern}_*_*_*.npz

mv ${path}${pattern}_out.npz ${path}../

tar zcvf ${path_save_tar}${pattern}.tar.gz ${path}

