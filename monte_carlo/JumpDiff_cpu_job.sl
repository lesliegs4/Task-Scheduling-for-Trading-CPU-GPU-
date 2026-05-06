#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --partition=main

python jump_diffusion_synth.py \
  --input ../usdjpy-m1-bid-2013.csv \
  --n-paths 100000 \
  --no-plots

