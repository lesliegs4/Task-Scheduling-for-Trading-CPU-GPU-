## Task Scheduling for Trading (CPU/GPU)

A Monte Carlo synthetic-data workload for trading backtests, with CPU and CUDA GPU implementations and CARC-friendly reproduction scripts. We use NautilusTrader as a reference backtest harness (stretch goal: plug synthetic bars into the backtest), but the core project focus is **parallelizing the Monte Carlo simulations** and comparing performance.

### TL;DR for graders

```bash
git clone <repo_url>
cd Task-Scheduling-for-Trading-CPU-GPU-
```

Jump to one of:

| Goal | Section |
| --- | --- |
| Run a 1-minute CPU smoke test (any host) | Quick smoke test (CPU) |
| Generate CPU synthetic bars (Heston / Jump Diffusion) | CPU synthetic generation |
| Run CUDA Jump Diffusion on CARC Discovery (GPU) | GPU reproduction on CARC |
| Run NautilusTrader tutorial backtest in Docker | NautilusTrader backtest (Docker) |

### Repository layout

```
Task-Scheduling-for-Trading-CPU-GPU-/
├── README.md
├── backtest/
│   ├── backtest_fx_bars.py                # NautilusTrader EMA-cross tutorial (FXCM bars)
│   └── backtest_synthetic_fx_bars.py      # Same tutorial + timing (not yet wired to synthetic CSV)
├── benchmarks/
│   ├── benchmark_cpu_gbm.py               # CPU GBM benchmark → CSV
│   └── benchmark_cuda_gbm.cu              # CUDA GBM benchmark → CSV (needs CUDA machine)
├── monte_carlo/
│   ├── heston_synth.py                    # CPU Heston fit + many-path simulation + synthetic OHLC CSV
│   ├── jump_diffusion_synth.py            # CPU Jump-Diffusion fit + many-path simulation + synthetic OHLC CSV
│   ├── trading_model_utils.py             # Shared helpers (load bars, OHLC from close path, etc.)
│   └── GPU/
│       ├── Makefile                       # Builds CUDA binary (CARC-friendly)
│       └── jump_diffusion_synth.cu        # CUDA Jump-Diffusion simulation (GPU)
├── reports/                               # Output folder for generated synthetic bars + params
├── run_jumpdiff_gpu.slurm                 # CARC SLURM: build + run CUDA Jump Diffusion
└── usdjpy-m1-{bid,ask}-2013.csv            # Input OHLC data (FXCM)
```

### Software requirements

| Tool | Version | Where needed |
| --- | --- | --- |
| Python | 3.9+ | CPU synthetic generation + CPU GBM benchmark |
| CUDA Toolkit (`nvcc`) + `curand` | CUDA 11+ | GPU Jump Diffusion + CUDA GBM benchmark |
| SLURM | any | CARC job submission |
| Docker | Desktop | NautilusTrader backtest (optional) |

---

## Quick smoke test (CPU)

Run a small Jump-Diffusion generation on CPU (no plots) to validate the pipeline:

```bash
python monte_carlo/jump_diffusion_synth.py \
  --input usdjpy-m1-bid-2013.csv \
  --n-paths 10 \
  --no-plots
```

Outputs:
- `reports/jump_diffusion/jump_diffusion_synthetic_bars.csv`
- `reports/jump_diffusion/jump_diffusion_params.json`

---

## CPU synthetic generation

### Jump Diffusion (CPU)

```bash
python monte_carlo/jump_diffusion_synth.py \
  --input usdjpy-m1-bid-2013.csv \
  --n-paths 10000
```

### Heston (CPU)

```bash
python monte_carlo/heston_synth.py \
  --input usdjpy-m1-bid-2013.csv \
  --n-paths 10000
```

Both scripts fit model parameters from the input OHLC bars, simulate many Monte Carlo paths (the heavy part), and save:
- one representative synthetic OHLC series (using the first simulated path)
- a JSON file with fitted parameters and summary stats

---

## GPU reproduction on CARC

### Step 1 — Submit the SLURM job

From the repo root on CARC:

```bash
sbatch run_jumpdiff_gpu.slurm
```

### Step 2 — Monitor

```bash
squeue -u $USER
```

### Step 3 — Inspect output logs

Your job writes:
- `jumpdiff_cuda_<JOBID>.out`
- `jumpdiff_cuda_<JOBID>.err`

Example:

```bash
cat jumpdiff_cuda_<JOBID>.out
cat jumpdiff_cuda_<JOBID>.err
```

### Step 4 — Inspect generated outputs

```bash
ls -lh reports/jump_diffusion/
```

Expected files:
- `jump_diffusion_synthetic_bars_cuda.csv`
- `jump_diffusion_params_cuda.json`

---

## NautilusTrader backtest (Docker)

This repo includes NautilusTrader tutorial scripts as a reference backtest harness. This is not the main parallelization workload, but it’s useful for validating that the dataset and backtest plumbing work end-to-end.

### Build and enter the container

```bash
docker compose up -d --build
docker compose exec backtest bash
```

### Run the tutorial backtest

Inside the container:

```bash
python backtest/backtest_fx_bars.py
```

### Stop containers

```bash
docker compose down
```


