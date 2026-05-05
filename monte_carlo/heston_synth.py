"""
    run like 
    python heston_synth.py --input ../usdjpy-m1-bid-2013.csv --n-paths #
                                                                    put 10 for testing 
                                                                    --no plots to skip graphs
    Output = synthetic bars in heston_synthetic_bars.csv and parameters it
             used to run Heston Monte Carlo simulation in heston_params.json
    - to put into main nautilus backtest to test but our main task is to parallellize Heston + JumpDiffusion w/ CUDA



"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_model_utils import (
    BarModelParams,
    close_prices_to_log_returns,
    infer_step_seconds,
    load_bars,
    make_ohlc_from_close_path,
    save_bars,
)


def fit_heston_params(df: pd.DataFrame) -> BarModelParams:
    close = df["close"]
    r = close_prices_to_log_returns(close)
    s0 = float(close.iloc[0])
    mu = float(np.mean(r))

    # Treat the return variance as the initial/long-run variance scale.
    v0 = float(np.var(r, ddof=1)) if len(r) > 1 else 1e-8
    theta = float(v0)

    # Estimate a simple mean reversion speed from lag-1 autocorrelation of squared returns.
    sq = (r - np.mean(r)) ** 2
    if len(sq) > 2 and np.std(sq[:-1]) > 0 and np.std(sq[1:]) > 0:
        corr = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    corr = float(np.clip(corr, -0.99, 0.99))
    # Convert autocorrelation into a positive mean-reversion proxy.
    kappa = float(max(0.5, -np.log(max(1e-6, abs(corr))) if abs(corr) > 1e-6 else 2.0))

    # Vol-of-vol proxy from variance of squared returns.
    xi = float(max(1e-6, np.std(sq, ddof=1) if len(sq) > 1 else 1e-4))

    # Use a conservative default. For FX this may be close to zero in practice.
    rho = -0.2

    return BarModelParams(
        model="heston",
        s0=s0,
        mu=mu,
        sigma=float(np.sqrt(v0)),
        extra={
            "v0": v0,
            "theta": theta,
            "kappa": kappa,
            "xi": xi,
            "rho": rho,
        },
    )


def simulate_heston(
    n_steps: int,
    params: BarModelParams,
    dt: float,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.empty(n_steps, dtype=float)
    s[0] = params.s0
    v = float(params.extra["v0"])
    theta = float(params.extra["theta"])
    kappa = float(params.extra["kappa"])
    xi = float(params.extra["xi"])
    rho = float(params.extra["rho"])
    mu = float(params.mu)

    sqrt_dt = np.sqrt(dt)
    for t in range(1, n_steps):
        z1 = rng.normal()
        z2 = rng.normal()
        w1 = z1
        w2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z2

        v = max(0.0, v + kappa * (theta - v) * dt + xi * np.sqrt(max(v, 0.0)) * sqrt_dt * w2)
        s[t] = max(1e-8, s[t - 1] * np.exp((mu - 0.5 * v) * dt + np.sqrt(max(v, 0.0)) * sqrt_dt * w1))

    return s


def simulate_heston_paths(
    n_paths: int,
    n_steps: int,
    params: BarModelParams,
    dt: float,
    seed: int = 42,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    Simulate many Heston paths.

    Returns an array of shape (n_paths, n_steps).
    This is the heavy part: increasing n_paths makes the workload much larger.
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps), dtype=np.float64)

    v0 = float(params.extra["v0"])
    theta = float(params.extra["theta"])
    kappa = float(params.extra["kappa"])
    xi = float(params.extra["xi"])
    rho = float(params.extra["rho"])
    mu = float(params.mu)

    sqrt_dt = np.sqrt(dt)

    for start in range(0, n_paths, chunk_size):
        end = min(start + chunk_size, n_paths)
        batch = end - start

        s = np.empty((batch, n_steps), dtype=np.float64)
        v = np.full(batch, v0, dtype=np.float64)
        s[:, 0] = params.s0

        for t in range(1, n_steps):
            z1 = rng.normal(size=batch)
            z2 = rng.normal(size=batch)

            w1 = z1
            w2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z2

            v = np.maximum(
                0.0,
                v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0.0)) * sqrt_dt * w2,
            )
            s[:, t] = np.maximum(
                1e-8,
                s[:, t - 1] * np.exp((mu - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0.0)) * sqrt_dt * w1),
            )

        paths[start:end, :] = s

    return paths


def plot_heston_results(
    timestamps: pd.Series,
    real_close: np.ndarray,
    all_paths: np.ndarray,
    output_prefix: str,
    max_paths_to_plot: int = 20,
) -> list[str]:
    output_files: list[str] = []
    prefix = Path(output_prefix)

    # Plot a few sample paths.
    plt.figure(figsize=(12, 6))
    n_plot = min(max_paths_to_plot, all_paths.shape[0])
    for i in range(n_plot):
        plt.plot(timestamps, all_paths[i], alpha=0.55, linewidth=1)
    plt.plot(timestamps, real_close, linewidth=2.0, label="Historical close")
    plt.title("Heston Monte Carlo - Sample Paths")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    paths_file = str(prefix / (prefix.name + "_paths.png"))
    plt.savefig(paths_file, dpi=150)
    plt.close()
    output_files.append(paths_file)

    # Plot final-price distribution.
    plt.figure(figsize=(12, 6))
    plt.hist(all_paths[:, -1], bins=50, alpha=0.85)
    plt.axvline(real_close[-1], linestyle="--", linewidth=2, label="Historical final close")
    plt.title("Heston Monte Carlo - Final Price Distribution")
    plt.xlabel("Final Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    hist_file = str(prefix / (prefix.name + "_final_prices.png"))
    plt.savefig(hist_file, dpi=150)
    plt.close()
    output_files.append(hist_file)

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Heston to OHLC bars and generate heavy synthetic workloads.")
    parser.add_argument("--input", required=True, help="Input OHLC CSV with timestamp,open,high,low,close")
    parser.add_argument("--output-bars", default="../reports/heston/heston_synthetic_bars.csv", help="Output synthetic OHLC CSV for one representative path")
    parser.add_argument("--output-params", default="../reports/heston/heston_params.json", help="Output fitted parameter JSON")
    parser.add_argument("--plot-prefix", default="../reports/heston", help="Prefix for saved graph files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-paths", type=int, default=10000, help="Number of Monte Carlo paths to simulate. Larger = heavier workload.")
    parser.add_argument("--chunk-size", type=int, default=256, help="How many paths to process at once to control memory use.")
    parser.add_argument("--save-path-stats", action="store_true", help="Also save summary stats for all simulated paths.")
    parser.add_argument("--no-plots", action="store_true", help="Disable graph generation.")
    args = parser.parse_args()

    # Ensure the reports/heston output directories exist so outputs go outside
    # the current working directory as requested.
    try:
        Path(args.output_bars).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_params).parent.mkdir(parents=True, exist_ok=True)
        Path(args.plot_prefix).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    t0 = perf_counter()

    # Load CSV
    t_load_start = perf_counter()
    df = load_bars(args.input)
    t_load_end = perf_counter()
    
    # Fit parameters
    t_fit_start = perf_counter()
    params = fit_heston_params(df)
    t_fit_end = perf_counter()
    
    dt_seconds = infer_step_seconds(df["timestamp"])
    dt = 1.0  # one step per bar interval for the synthetic process

    # Heavy work: simulate many paths.
    t_heston_start = perf_counter()
    all_paths = simulate_heston_paths(
        n_paths=args.n_paths,
        n_steps=len(df),
        params=params,
        dt=dt,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    t_heston_end = perf_counter()

    # Use the first simulated path as the representative series for OHLC output.
    close_path = all_paths[0]
    synthetic = make_ohlc_from_close_path(
        df["timestamp"],
        close_path,
        rng=np.random.default_rng(args.seed),
    )
    
    # Save files
    t_save_start = perf_counter()
    save_bars(synthetic, args.output_bars)
    t_save_end = perf_counter()

    # Generate plots
    t_plot_start = perf_counter()
    plot_files: list[str] = []
    if not args.no_plots:
        plot_files = plot_heston_results(
            timestamps=df["timestamp"],
            real_close=df["close"].to_numpy(dtype=float),
            all_paths=all_paths,
            output_prefix=args.plot_prefix,
        )
    t_plot_end = perf_counter()

    t1 = perf_counter()

    # Optional summary stats across all paths.
    path_stats = {
        "final_price_mean": float(np.mean(all_paths[:, -1])),
        "final_price_std": float(np.std(all_paths[:, -1], ddof=1)) if args.n_paths > 1 else 0.0,
        "min_final_price": float(np.min(all_paths[:, -1])),
        "max_final_price": float(np.max(all_paths[:, -1])),
    }

    with open(args.output_params, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fit_from": args.input,
                "step_seconds": dt_seconds,
                "n_paths": args.n_paths,
                "chunk_size": args.chunk_size,
                "elapsed_seconds": t1 - t0,
                "timing_breakdown": {
                    "load_csv_seconds": t_load_end - t_load_start,
                    "fit_parameters_seconds": t_fit_end - t_fit_start,
                    "heston_simulation_seconds": t_heston_end - t_heston_start,
                    "save_files_seconds": t_save_end - t_save_start,
                    "generate_plots_seconds": t_plot_end - t_plot_start,
                },
                "params": asdict(params),
                "path_stats": path_stats,
                "plot_files": plot_files,
            },
            f,
            indent=2,
            sort_keys=True,
            default=float,
        )

    if args.save_path_stats:
        with open("heston_path_stats.json", "w", encoding="utf-8") as f:
            json.dump(path_stats, f, indent=2, sort_keys=True)

    print(f"Saved synthetic bars to {args.output_bars}")
    print(f"Saved fitted params to {args.output_params}")
    if plot_files:
        for file in plot_files:
            print(f"Saved plot to {file}")
    print("\n=== Timing Breakdown ===")
    print(f"  Load CSV:           {t_load_end - t_load_start:.3f}s")
    print(f"  Fit parameters:     {t_fit_end - t_fit_start:.3f}s")
    print(f"  Heston simulation:  {t_heston_end - t_heston_start:.3f}s")
    print(f"  Save files:         {t_save_end - t_save_start:.3f}s")
    print(f"  Generate plots:     {t_plot_end - t_plot_start:.3f}s")
    print(f"  Total elapsed:      {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
