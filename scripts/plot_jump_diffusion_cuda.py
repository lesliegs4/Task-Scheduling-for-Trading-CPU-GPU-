from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    # Try relative to CWD first (matches how jobs are usually run).
    if Path(maybe_path).exists():
        return Path(maybe_path)
    # Then relative to the JSON’s directory.
    return (base_dir / maybe_path).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots for CUDA Jump Diffusion outputs and update plot_files in JSON.")
    ap.add_argument("--json", required=True, help="Path to jump_diffusion_params_cuda.json")
    ap.add_argument("--max-paths", type=int, default=20, help="Max sample paths to plot (default 20)")
    args = ap.parse_args()

    json_path = Path(args.json).resolve()
    base_dir = json_path.parent

    t0 = perf_counter()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    fit_from = data.get("fit_from")
    if not fit_from:
        raise SystemExit("JSON missing fit_from")

    artifacts = data.get("artifacts") or {}
    sample_paths_csv = artifacts.get("sample_paths_csv") or ""
    final_prices_csv = artifacts.get("final_prices_csv") or ""

    # Load historical closes for overlay.
    input_csv_path = _resolve_path(base_dir, fit_from)
    df = pd.read_csv(input_csv_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise SystemExit("Input CSV must have columns: timestamp, close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)

    n_steps = int(data.get("n_steps") or 0)
    if n_steps > 0:
        df = df.iloc[:n_steps].copy()

    plot_files: list[str] = []

    # 1) Paths plot (requires sample_paths_csv).
    if sample_paths_csv:
        sp_path = _resolve_path(base_dir, sample_paths_csv)
        sp = pd.read_csv(sp_path)
        ts = pd.to_datetime(sp["timestamp"], utc=True, errors="coerce")
        path_cols = [c for c in sp.columns if c.startswith("path_")]
        if path_cols:
            plt.figure(figsize=(12, 6))
            for c in path_cols[: args.max_paths]:
                plt.plot(ts, sp[c].to_numpy(dtype=float), alpha=0.55, linewidth=1)
            plt.plot(df["timestamp"], df["close"].to_numpy(dtype=float), linewidth=2.0, label="Historical close")
            plt.title("Jump-Diffusion Monte Carlo (CUDA) - Sample Paths")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            out_paths_png = base_dir / "jump_diffusion_paths_cuda.png"
            plt.savefig(out_paths_png, dpi=150)
            plt.close()
            plot_files.append(str(out_paths_png))

    # 2) Final-price histogram (requires final_prices_csv).
    if final_prices_csv:
        fp_path = _resolve_path(base_dir, final_prices_csv)
        fp = pd.read_csv(fp_path)
        if "final_price" in fp.columns:
            final = fp["final_price"].to_numpy(dtype=float)
        else:
            final = fp.iloc[:, 0].to_numpy(dtype=float)

        plt.figure(figsize=(12, 6))
        plt.hist(final, bins=50, alpha=0.85)
        plt.axvline(float(df["close"].iloc[-1]), linestyle="--", linewidth=2, label="Historical final close")
        plt.title("Jump-Diffusion Monte Carlo (CUDA) - Final Price Distribution")
        plt.xlabel("Final Price")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        out_hist_png = base_dir / "jump_diffusion_final_prices_cuda.png"
        plt.savefig(out_hist_png, dpi=150)
        plt.close()
        plot_files.append(str(out_hist_png))

    t1 = perf_counter()

    # Update JSON: plot_files + generate_plots_seconds
    data["plot_files"] = plot_files
    tb = data.get("timing_breakdown") or {}
    tb["generate_plots_seconds"] = float(t1 - t0)
    data["timing_breakdown"] = tb

    json_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")

    print(f"Updated {json_path} with {len(plot_files)} plot file(s).")
    for p in plot_files:
        print(f"Saved plot to {p}")


if __name__ == "__main__":
    main()

