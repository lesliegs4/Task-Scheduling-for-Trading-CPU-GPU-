#!/bin/bash
#SBATCH --job-name=jumpdiff_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=jumpdiff_cuda_%j.out
#SBATCH --error=jumpdiff_cuda_%j.err

set -euo pipefail

module purge
module load nvhpc
module load gcc/8.5.0
module avail cuda

# Load CUDA so `nvcc` exists in batch jobs.
# (On this cluster, `module avail cuda` lists e.g. cuda/11.3.0 as default.)
module load cuda/11.3.0

# Avoid swapping compilers after pinning gcc/8.5.0.
module load gcc/8.5.0

echo "Using nvcc at: $(which nvcc || true)"
nvcc --version

cd ~/Task-Scheduling-for-Trading-CPU-GPU-

# GCC 8.x needs -lstdc++fs for std::filesystem.
nvcc -O3 -std=c++17 monte_carlo/GPU/jump_diffusion_synth.cu -o jump_diffusion_cuda -lcurand -lstdc++fs

mkdir -p reports/jump_diffusion

./jump_diffusion_cuda --input usdjpy-m1-bid-2013.csv --n-paths 1000 --n-steps 200

