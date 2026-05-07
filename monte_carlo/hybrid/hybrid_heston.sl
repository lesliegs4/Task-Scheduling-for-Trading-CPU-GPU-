#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=hybrid_heston
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hybrid_heston_%j.out
#SBATCH --error=hybrid_heston_%j.err

# Hybrid Heston: CPU (NumPy) + GPU (heston_gpu) via hybrid_heston.py
#
# Portable: put this .sl next to hybrid_heston.py (flat bundle) or run from repo root.
# Default HYBRID_ROOT is the directory you were in when you ran sbatch (SLURM_SUBMIT_DIR).
#   cd /path/to/bundle && sbatch hybrid_heston.sl
#   cd /path/to/bundle && sbatch hybrid_heston.sl 500000
#   cd /path/to/bundle && sbatch hybrid_heston.sl 500000 252
# From another cwd, set the bundle explicitly:
#   HYBRID_ROOT=/path/to/bundle sbatch /path/to/bundle/hybrid_heston.sl
#
# Repo layout (this file under monte_carlo/hybrid/):
#   sbatch monte_carlo/hybrid/hybrid_heston.sl
#
# Flat bundle on CARC needs (besides this .sl): hybrid_heston.py, heston_synth.py,
# trading_model_utils.py, OHLC CSV, and either heston_gpu (binary) or heston_gpu.cu in
# the same directory (compiled automatically; override arch with HESTON_NVCC_ARCH, e.g. sm_80 for A100).
#
# Override defaults (examples):
#   N_PATHS=500000 N_STEPS=200 SCHEDULE=static sbatch hybrid_heston.sl
#   SCHEDULE=dynamic BATCH_CAP=8192 sbatch hybrid_heston.sl
# Save every path's merged final price in the job cwd (HYBRID_ROOT); file can be huge:
#   MERGED_FINALS_OUT=hybrid_merged_final_prices.csv sbatch hybrid_heston.sl
#
# If your partition requires a GPU type, set e.g. #SBATCH --gres=gpu:v100:1
# Match gcc/cuda modules to what `module avail` shows on CARC for the gpu partition.
#
# Python: 3.7+ (dataclasses). CARC Discovery: ``module avail python`` -> use python/3.11.9.
# Override: PYTHON_MODULE=python/3.12.8 sbatch hybrid_heston.sl
# Skip load (e.g. conda already activated): PYTHON_MODULE_SKIP=1 sbatch hybrid_heston.sl

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3

PYTHON_CMD="${PYTHON_CMD:-python3}"
if [[ -z "${PYTHON_MODULE_SKIP:-}" ]]; then
  module load "${PYTHON_MODULE:-python/3.11.9}"
fi

echo "Python: $($PYTHON_CMD -c 'import sys; print(sys.executable, sys.version.split()[0])')"

export HYBRID_ROOT="${HYBRID_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${HYBRID_ROOT}"

if [[ -f "${HYBRID_ROOT}/hybrid_heston.py" ]]; then
  HYBRID_PY="${HYBRID_ROOT}/hybrid_heston.py"
elif [[ -f "${HYBRID_ROOT}/monte_carlo/hybrid/hybrid_heston.py" ]]; then
  HYBRID_PY="${HYBRID_ROOT}/monte_carlo/hybrid/hybrid_heston.py"
else
  echo "Cannot find hybrid_heston.py under ${HYBRID_ROOT} (flat or repo layout)." >&2
  exit 1
fi

echo "HYBRID_ROOT=${HYBRID_ROOT}"
echo "HYBRID_PY=${HYBRID_PY}"
echo "nvcc: $(command -v nvcc || echo 'missing')"
nvcc --version 2>/dev/null || true

# Use existing heston_gpu if present; otherwise build from a known Makefile location.
if [[ -n "${HESTON_CUDA_BIN:-}" && -f "${HESTON_CUDA_BIN}" ]]; then
  :
elif [[ -f "${HYBRID_ROOT}/heston_gpu" ]]; then
  export HESTON_CUDA_BIN="${HYBRID_ROOT}/heston_gpu"
elif [[ -f "${HYBRID_ROOT}/GPU/build/heston_gpu" ]]; then
  export HESTON_CUDA_BIN="${HYBRID_ROOT}/GPU/build/heston_gpu"
elif [[ -f "${HYBRID_ROOT}/monte_carlo/GPU/build/heston_gpu" ]]; then
  export HESTON_CUDA_BIN="${HYBRID_ROOT}/monte_carlo/GPU/build/heston_gpu"
elif [[ -f "${HYBRID_ROOT}/heston_gpu.cu" ]]; then
  NVCC_CMD="${NVCC:-nvcc}"
  HESTON_NVCC_ARCH="${HESTON_NVCC_ARCH:-sm_70}"
  echo "[BUILD] ${NVCC_CMD} heston_gpu.cu -> ${HYBRID_ROOT}/heston_gpu (arch=${HESTON_NVCC_ARCH})"
  "${NVCC_CMD}" -O3 -arch="${HESTON_NVCC_ARCH}" -o "${HYBRID_ROOT}/heston_gpu" \
    "${HYBRID_ROOT}/heston_gpu.cu" -lcurand -std=c++17
  export HESTON_CUDA_BIN="${HYBRID_ROOT}/heston_gpu"
else
  if [[ -f "${HYBRID_ROOT}/monte_carlo/GPU/Makefile" ]]; then
    make -C "${HYBRID_ROOT}/monte_carlo/GPU" all
    export HESTON_CUDA_BIN="${HYBRID_ROOT}/monte_carlo/GPU/build/heston_gpu"
  elif [[ -f "${HYBRID_ROOT}/GPU/Makefile" ]]; then
    make -C "${HYBRID_ROOT}/GPU" all
    export HESTON_CUDA_BIN="${HYBRID_ROOT}/GPU/build/heston_gpu"
  else
    echo "No heston_gpu binary and no GPU/Makefile under ${HYBRID_ROOT}." >&2
    echo "Copy heston_gpu, or include monte_carlo/GPU (or GPU/) with Makefile, or set HESTON_CUDA_BIN." >&2
    exit 1
  fi
fi

if [[ ! -f "${HESTON_CUDA_BIN}" ]]; then
  echo "HESTON_CUDA_BIN is not a file: ${HESTON_CUDA_BIN}" >&2
  exit 1
fi

INPUT_CSV="${INPUT_CSV:-usdjpy-m1-bid-2013.csv}"
N_PATHS="${N_PATHS:-100000}"
N_STEPS="${N_STEPS:-200}"
SCHEDULE="${SCHEDULE:-static}"
CPU_FRACTION="${CPU_FRACTION:-0.35}"
BATCH_CAP="${BATCH_CAP:-4096}"
SEED="${SEED:-42}"

# Positional args from: sbatch .../hybrid_heston.sl [N_PATHS] [N_STEPS]
if [[ $# -ge 1 ]]; then N_PATHS="$1"; fi
if [[ $# -ge 2 ]]; then N_STEPS="$2"; fi

JSON_OUT="${JSON_OUT:-${HYBRID_ROOT}/hybrid_heston_${SLURM_JOB_ID:-local}.json}"

EXTRA=()
if [[ "${SCHEDULE}" == "dynamic" ]]; then
  EXTRA+=(--schedule dynamic --batch-cap "${BATCH_CAP}")
else
  EXTRA+=(--schedule static --cpu-fraction "${CPU_FRACTION}")
fi

MERGED_EXTRA=()
if [[ -n "${MERGED_FINALS_OUT:-}" ]]; then
  MERGED_EXTRA=(--merged-final-prices-out "${MERGED_FINALS_OUT}")
fi

"${PYTHON_CMD}" -u "${HYBRID_PY}" \
  --input "${INPUT_CSV}" \
  --n-paths "${N_PATHS}" \
  --n-steps "${N_STEPS}" \
  --seed "${SEED}" \
  "${EXTRA[@]}" \
  "${MERGED_EXTRA[@]}" \
  --json-out "${JSON_OUT}"

echo "Wrote ${JSON_OUT}"
