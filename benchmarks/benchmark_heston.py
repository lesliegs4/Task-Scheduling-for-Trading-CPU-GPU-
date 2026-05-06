"""
Benchmark: CPU (NumPy) vs. GPU (CUDA) Heston Simulation
"""
// new
import time
import subprocess
import pandas as pd
import numpy as np
import sys
import os

# helps ensure the root directory is in the path so we can find 'monte_carlo'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# importing the specific CPU math function and the required Parameter class
from monte_carlo.heston_synth import simulate_heston
from monte_carlo.trading_model_utils import BarModelParams

# configuration & hyperparameters
TEST_SIZES = [1000, 10000, 100000]  # scaling paths to test GPU efficiency

N_STEPS = 252  # number of time steps (one trading year)

DT = 1.0 / 252.0  # time step increment


# package parameters into the dataclass format
params = BarModelParams(
    model="heston",

    s0=150.0,

    mu=0.05,

    sigma=0.2,

    extra={

        "kappa": 2.0,   # mean reversion speed

        "theta": 0.04,  # long-term variance

        "xi": 0.3,      # volatility of volatility

        "rho": -0.7,    # correlation

        "v0": 0.04      # initial variance
    }
)

results = []

print(f"{'Paths':>10} | {'CPU Time (s)':>12} | {'GPU Time (s)':>12} | {'Speedup':>10}")

print("-" * 55)

for n_paths in TEST_SIZES:

    # benchmark CPU (Python/NumPy)

    # we loop n_paths times because the CPU function simulates one path at a time

    start_cpu = time.perf_counter()

    for _ in range(n_paths):

        _ = simulate_heston(

            n_steps=N_STEPS,

            params=params,

            dt=DT
        )

    cpu_time = time.perf_counter() - start_cpu


    # benchmark GPU (CUDA)

    # executes the compiled binary which handles all n_paths in parallel

    start_gpu = time.perf_counter()

    try:

        subprocess.run(

            ["./heston_cuda_bin", str(n_paths)],

            check=True,

            capture_output=True
        )

        gpu_time = time.perf_counter() - start_gpu

    except (

        FileNotFoundError,

        subprocess.CalledProcessError

    ):

        gpu_time = np.nan


    # performance metrics

    speedup = (

        cpu_time / gpu_time

        if (gpu_time and gpu_time > 0)

        else 0
    )

    results.append({

        "paths": n_paths,

        "cpu_time": cpu_time,

        "gpu_time": gpu_time,

        "speedup": speedup
    })


    if np.isnan(gpu_time):

        print(

            f"{n_paths:10d} | "

            f"{cpu_time:12.4f} | "

            f"{'MISSING BIN':>12} | "

            f"{'N/A':>10}"
        )

    else:

        print(

            f"{n_paths:10d} | "

            f"{cpu_time:12.4f} | "

            f"{gpu_time:12.4f} | "

            f"{speedup:9.1f}x"
        )


# save results in csv

df = pd.DataFrame(results)

df.to_csv(

    "heston_bench_results.csv",

    index=False
)

print("\nResults saved to heston_bench_results.csv")