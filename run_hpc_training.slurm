#!/bin/bash

#SBATCH --job-name=latexocr
#SBATCH --account=adarsh
#SBATCH --output=%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=5gb
#SBATCH --gres=gpu:1
#SBATCH --time=02-12:00:00
#SBATCH --partition=standard
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abanerjee@arizona.edu

module load python/3.11
rm -rf checkpoints
./scripts/run_original_training