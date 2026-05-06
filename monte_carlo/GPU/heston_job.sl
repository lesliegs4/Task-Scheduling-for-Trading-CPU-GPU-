#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=heston_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=gpujob.out
#SBATCH --error=gpujob.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3

echo "[BUILD] heston_gpu"
nvcc -O3 -arch=sm_70 -o heston_gpu heston_gpu.cu -lcurand -std=c++17

# Fixed inputs for every run
INPUT_CSV="usdjpy-m1-bid-2013.csv"
SEED=42

# Change the number of paths by passing the first argument to sbatch.
# Optional second argument sets the CUDA block size.
N_PATHS="${1:-1000000}"
BLOCK_SIZE="${2:-256}"

OUT_BARS="heston_synthetic_bars.csv"
OUT_PARAMS="heston_params.json"

echo "[RUN] ./heston_gpu"
srun ./heston_gpu \
  --input "${INPUT_CSV}" \
  --n-paths "${N_PATHS}" \
  --seed "${SEED}" \
  --out-bars "${OUT_BARS}" \
  --out-params "${OUT_PARAMS}" \
  --block-size "${BLOCK_SIZE}"
