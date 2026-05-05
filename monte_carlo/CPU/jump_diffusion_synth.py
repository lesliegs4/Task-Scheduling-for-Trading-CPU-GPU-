"""
    run like 
    python jump_diffusion_synth.py --input usdjpy-m1-bid-2013.csv --n-paths #
                                                                    put 10 for testing 
    Output = synthetic bars in jump_diffusion_synthetic_bars.csv and parameters it
             used to run Jump-Diffusion Monte Carlo simulation in jump_diffusion_params.json
    - to put into main nautilus backtest to test but our main task is to parallellize Jump-Diffusion w/ CUDA



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

from monte_carlo.CPU.trading_model_utils import (
    BarModelParams,
    close_prices_to_log_returns,
    infer_step_seconds,
    load_bars,
    make_ohlc_from_close_path,
    save_bars,
)


def fit_jump_diffusion_params(df: pd.DataFrame, jump_threshold_mult: float = 2.5) -> BarModelParams:
    close = df["close"]
    r = close_prices_to_log_returns(close)
    s0 = float(close.iloc[0])
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if len(r) > 1 else 1e-6

    residuals = r - mu
    threshold = jump_threshold_mult * sigma
    jump_mask = np.abs(residuals) > threshold

    jump_residuals = residuals[jump_mask]
    non_jump_residuals = residuals[~jump_mask]

    lam = float(len(jump_residuals) / max(1, len(r)))
    mu_j = float(np.mean(jump_residuals)) if len(jump_residuals) else 0.0
    sigma_j = float(np.std(jump_residuals, ddof=1)) if len(jump_residuals) > 1 else max(1e-6, sigma * 0.5)
    sigma_diffusion = float(np.std(non_jump_residuals, ddof=1)) if len(non_jump_residuals) > 1 else sigma

    return BarModelParams(
        model="jump_diffusion",
        s0=s0,
        mu=mu,
        sigma=sigma_diffusion,
        extra={
            "lambda": lam,
            "muJ": mu_j,
            "sigmaJ": sigma_j,
            "threshold": threshold,
        },
    )


def simulate_jump_diffusion(
    n_steps: int,
    params: BarModelParams,
    dt: float,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.empty(n_steps, dtype=float)
    s[0] = params.s0
    mu = float(params.mu)
    sigma = float(params.sigma)
    lam = float(params.extra["lambda"])
    mu_j = float(params.extra["muJ"])
    sigma_j = float(params.extra["sigmaJ"])

    sqrt_dt = np.sqrt(dt)
    for t in range(1, n_steps):
        jump_count = rng.poisson(lam * dt)
        jump_term = 0.0
        if jump_count > 0:
            jump_term = float(np.sum(rng.normal(mu_j, sigma_j, size=jump_count)))
        diffusion = (mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * rng.normal()
        s[t] = max(1e-8, s[t - 1] * np.exp(diffusion + jump_term))

    return s


def simulate_jump_diffusion_paths(
    n_paths: int,
    n_steps: int,
    params: BarModelParams,
    dt: float,
    seed: int = 42,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    Simulate many jump-diffusion paths.

    Returns array of shape (n_paths, n_steps).
    This is the heavy Monte Carlo workload.
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps), dtype=np.float64)

    mu = float(params.mu)
    sigma = float(params.sigma)
    lam = float(params.extra["lambda"])
    mu_j = float(params.extra["muJ"])
    sigma_j = float(params.extra["sigmaJ"])

    sqrt_dt = np.sqrt(dt)

    for start in range(0, n_paths, chunk_size):
        end = min(start + chunk_size, n_paths)
        batch = end - start

        s = np.empty((batch, n_steps), dtype=np.float64)
        s[:, 0] = params.s0

        for t in range(1, n_steps):
            jump_counts = rng.poisson(lam * dt, size=batch)
            diffusion = (mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * rng.normal(size=batch)

            jump_term = np.zeros(batch, dtype=np.float64)
            jump_idx = np.nonzero(jump_counts > 0)[0]

            for i in jump_idx:
                jump_term[i] = float(np.sum(rng.normal(mu_j, sigma_j, size=int(jump_counts[i]))))

            s[:, t] = np.maximum(1e-8, s[:, t - 1] * np.exp(diffusion + jump_term))

        paths[start:end, :] = s

    return paths


def plot_jump_diffusion_results(
    timestamps: pd.Series,
    real_close: np.ndarray,
    all_paths: np.ndarray,
    output_prefix: str,
    max_paths_to_plot: int = 20,
) -> list[str]:
    output_files: list[str] = []
    prefix = Path(output_prefix)

    plt.figure(figsize=(12, 6))
    n_plot = min(max_paths_to_plot, all_paths.shape[0])
    for i in range(n_plot):
        plt.plot(timestamps, all_paths[i], alpha=0.55, linewidth=1)
    plt.plot(timestamps, real_close, linewidth=2.0, label="Historical close")
    plt.title("Jump-Diffusion Monte Carlo - Sample Paths")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    paths_file = str(prefix / (prefix.name + "_paths.png"))
    plt.savefig(paths_file, dpi=150)
    plt.close()
    output_files.append(paths_file)

    plt.figure(figsize=(12, 6))
    plt.hist(all_paths[:, -1], bins=50, alpha=0.85)
    plt.axvline(real_close[-1], linestyle="--", linewidth=2, label="Historical final close")
    plt.title("Jump-Diffusion Monte Carlo - Final Price Distribution")
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
    parser = argparse.ArgumentParser(description="Fit Jump-Diffusion to OHLC bars and generate heavy synthetic workloads.")
    parser.add_argument("--input", required=True, help="Input OHLC CSV with timestamp,open,high,low,close")
    parser.add_argument("--output-bars", default="../reports/jump_diffusion/jump_diffusion_synthetic_bars.csv", help="Output synthetic OHLC CSV for one representative path")
    parser.add_argument("--output-params", default="../reports/jump_diffusion/jump_diffusion_params.json", help="Output fitted parameter JSON")
    parser.add_argument("--plot-prefix", default="../reports/jump_diffusion", help="Prefix for saved graph files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jump-threshold-mult", type=float, default=2.5)
    parser.add_argument("--n-paths", type=int, default=10000, help="Number of Monte Carlo paths to simulate. Larger = heavier workload.")
    parser.add_argument("--chunk-size", type=int, default=256, help="How many paths to process at once to control memory use.")
    parser.add_argument("--save-path-stats", action="store_true", help="Also save summary stats for all simulated paths.")
    parser.add_argument("--no-plots", action="store_true", help="Disable graph generation.")
    args = parser.parse_args()

    # Ensure reports/jump_diffusion directory exists so outputs are written there
    try:
        Path(args.output_bars).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_params).parent.mkdir(parents=True, exist_ok=True)
        Path(args.plot_prefix).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    t0 = perf_counter()

    df = load_bars(args.input)
    params = fit_jump_diffusion_params(df, jump_threshold_mult=args.jump_threshold_mult)
    dt_seconds = infer_step_seconds(df["timestamp"])
    dt = 1.0  # one step per bar interval for the synthetic process

    all_paths = simulate_jump_diffusion_paths(
        n_paths=args.n_paths,
        n_steps=len(df),
        params=params,
        dt=dt,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    # Use the first simulated path as the synthetic OHLC series for your backtest.
    close_path = all_paths[0]
    synthetic = make_ohlc_from_close_path(
        df["timestamp"],
        close_path,
        rng=np.random.default_rng(args.seed),
    )
    save_bars(synthetic, args.output_bars)

    plot_files: list[str] = []
    if not args.no_plots:
        plot_files = plot_jump_diffusion_results(
            timestamps=df["timestamp"],
            real_close=df["close"].to_numpy(dtype=float),
            all_paths=all_paths,
            output_prefix=args.plot_prefix,
        )

    t1 = perf_counter()

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
        with open("jump_diffusion_path_stats.json", "w", encoding="utf-8") as f:
            json.dump(path_stats, f, indent=2, sort_keys=True)

    print(f"Saved synthetic bars to {args.output_bars}")
    print(f"Saved fitted params to {args.output_params}")
    if plot_files:
        for file in plot_files:
            print(f"Saved plot to {file}")
    print(f"Simulated {args.n_paths} paths in {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
