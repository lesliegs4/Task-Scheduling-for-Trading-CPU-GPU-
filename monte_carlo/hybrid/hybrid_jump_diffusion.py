"""
Hybrid CPU + GPU Jump-Diffusion Monte Carlo benchmark.

Runs a static split of paths across CPU (NumPy, chunked) and GPU (CUDA binary)
in parallel threads, then merges final prices for combined statistics.

This is *work partitioning* / *heterogeneous task parallelism*: you choose how
many paths each device owns. It is not an OS scheduler.

Usage (from repo root, after building the CUDA binary):

  make -C monte_carlo/GPU
  python monte_carlo/hybrid/hybrid_jump_diffusion.py --input usdjpy-m1-bid-2013.csv --n-paths 10000

Override binary path if needed:

  set JUMP_DIFFUSION_CUDA_BIN=C:\\path\\to\\jump_diffusion_cuda.exe   # Windows
  export JUMP_DIFFUSION_CUDA_BIN=/path/to/jump_diffusion_cuda         # Linux
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from typing import Any, Dict, Optional, Tuple

# Repo root: .../Task-Scheduling-for-Trading-CPU-GPU-
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monte_carlo.jump_diffusion_synth import (  # noqa: E402
    fit_jump_diffusion_params,
    simulate_jump_diffusion_paths,
)
from monte_carlo.trading_model_utils import load_bars  # noqa: E402


def _default_cuda_binary() -> Path:
    env = os.environ.get("JUMP_DIFFUSION_CUDA_BIN")
    if env:
        return Path(env)
    base = REPO_ROOT / "monte_carlo" / "GPU" / "build" / "jump_diffusion_cuda"
    if sys.platform == "win32":
        win = base.with_suffix(".exe")
        if win.is_file():
            return win
    return base


def _split_counts(n_paths: int, n_cpu: Optional[int], n_gpu: Optional[int], cpu_fraction: float) -> Tuple[int, int]:
    if n_cpu is not None and n_gpu is not None:
        if n_cpu + n_gpu != n_paths:
            raise ValueError(f"n_cpu ({n_cpu}) + n_gpu ({n_gpu}) must equal n_paths ({n_paths}).")
        return max(0, n_cpu), max(0, n_gpu)
    if n_cpu is not None:
        nc = max(0, min(n_paths, n_cpu))
        return nc, n_paths - nc
    if n_gpu is not None:
        ng = max(0, min(n_paths, n_gpu))
        return n_paths - ng, ng
    frac = float(np.clip(cpu_fraction, 0.0, 1.0))
    n_cpu_auto = int(round(n_paths * frac))
    n_cpu_auto = max(0, min(n_paths, n_cpu_auto))
    return n_cpu_auto, n_paths - n_cpu_auto


def _run_cpu_paths(
    *,
    n_cpu: int,
    n_steps: int,
    params,
    dt: float,
    seed: int,
    chunk_size: int,
) -> Tuple[np.ndarray, float]:
    if n_cpu <= 0:
        return np.zeros((0, n_steps), dtype=np.float64), 0.0
    t0 = time.perf_counter()
    paths = simulate_jump_diffusion_paths(
        n_paths=n_cpu,
        n_steps=n_steps,
        params=params,
        dt=dt,
        seed=seed,
        chunk_size=chunk_size,
    )
    t1 = time.perf_counter()
    return paths, t1 - t0


def _run_gpu_subprocess(
    *,
    cuda_bin: Path,
    input_csv: Path,
    n_gpu: int,
    n_steps: int,
    seed: int,
    jump_threshold_mult: float,
    dt: float,
    tmpdir: Path,
    artifact_prefix: str = "hybrid_gpu",
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    if n_gpu <= 0:
        return np.zeros(0, dtype=np.float64), 0.0, {}
    if not cuda_bin.is_file():
        raise FileNotFoundError(
            f"CUDA binary not found at {cuda_bin}. Build with: make -C monte_carlo/GPU"
        )

    bars_out = tmpdir / f"{artifact_prefix}_bars.csv"
    params_out = tmpdir / f"{artifact_prefix}_params.json"
    finals_out = tmpdir / f"{artifact_prefix}_final_prices.csv"

    cmd = [
        str(cuda_bin),
        "--input",
        str(input_csv.resolve()),
        "--n-paths",
        str(n_gpu),
        "--n-steps",
        str(n_steps),
        "--seed",
        str(seed),
        "--jump-threshold-mult",
        str(jump_threshold_mult),
        "--dt",
        str(dt),
        "--output-bars",
        str(bars_out),
        "--output-params",
        str(params_out),
        "--output-final-prices",
        str(finals_out),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"GPU subprocess failed (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"{err}"
        )
    if not finals_out.is_file():
        raise RuntimeError(f"GPU run did not write final prices: {finals_out}")

    finals = np.loadtxt(finals_out, dtype=np.float64, skiprows=1)
    if finals.ndim == 0:
        finals = np.array([float(finals)], dtype=np.float64)
    if finals.shape[0] != n_gpu:
        raise RuntimeError(
            f"Expected {n_gpu} GPU final prices, got {finals.shape[0]} from {finals_out}"
        )
    gpu_meta: Dict[str, Any] = {}
    if params_out.is_file():
        try:
            gpu_meta = json.loads(params_out.read_text(encoding="utf-8"))
        except Exception:
            gpu_meta = {}
    return finals, t1 - t0, gpu_meta


def _run_dynamic_hybrid(
    *,
    cuda_bin: Path,
    trimmed_csv: Path,
    tmpdir: Path,
    n_paths: int,
    n_steps: int,
    params,
    dt: float,
    jump_threshold_mult: float,
    base_seed_cpu: int,
    base_seed_gpu: int,
    batch_cap: int,
    cpu_chunk_size: int,
) -> tuple[list[np.ndarray], np.ndarray, float, float, float, int, int]:
    """
    Dynamic batch queue: CPU and GPU workers share a path budget (thread-safe claims).
    GPU uses the stock jump_diffusion_cuda binary once per claimed batch.
    """
    if not cuda_bin.is_file():
        raise FileNotFoundError(
            f"CUDA binary not found at {cuda_bin}. Build with: make -C monte_carlo/GPU"
        )

    remaining = n_paths
    lock = threading.Lock()
    cpu_parts: list[np.ndarray] = []
    gpu_batches: list[np.ndarray] = []
    cpu_batch_time = 0.0
    gpu_batch_time = 0.0
    cpu_paths_done = 0
    gpu_paths_done = 0
    cpu_batch_idx = 0
    gpu_batch_idx = 0

    def claim() -> int:
        nonlocal remaining
        with lock:
            if remaining <= 0:
                return 0
            take = min(batch_cap, remaining)
            remaining -= take
            return take

    def cpu_worker() -> None:
        nonlocal cpu_batch_time, cpu_paths_done, cpu_batch_idx
        while True:
            take = claim()
            if take == 0:
                break
            seed_b = int(base_seed_cpu) + cpu_batch_idx * 1_000_003
            t0 = time.perf_counter()
            paths = simulate_jump_diffusion_paths(
                n_paths=take,
                n_steps=n_steps,
                params=params,
                dt=dt,
                seed=seed_b,
                chunk_size=min(cpu_chunk_size, take),
            )
            t1 = time.perf_counter()
            cpu_batch_time += t1 - t0
            cpu_parts.append(paths[:, -1].copy())
            cpu_paths_done += take
            cpu_batch_idx += 1

    def gpu_worker() -> None:
        nonlocal gpu_batch_time, gpu_paths_done, gpu_batch_idx
        while True:
            take = claim()
            if take == 0:
                break
            seed_b = int(base_seed_gpu) + gpu_batch_idx * 1_000_003
            prefix = f"gpu_dyn_{gpu_batch_idx:05d}"
            finals, elapsed, _gpu_meta = _run_gpu_subprocess(
                cuda_bin=cuda_bin,
                input_csv=trimmed_csv,
                n_gpu=take,
                n_steps=n_steps,
                seed=seed_b,
                jump_threshold_mult=jump_threshold_mult,
                dt=dt,
                tmpdir=tmpdir,
                artifact_prefix=prefix,
            )
            gpu_batch_time += elapsed
            gpu_batches.append(finals)
            gpu_paths_done += take
            gpu_batch_idx += 1

    wall0 = time.perf_counter()
    t_cpu = threading.Thread(target=cpu_worker, name="cpu_batches")
    t_gpu = threading.Thread(target=gpu_worker, name="gpu_batches")
    t_cpu.start()
    t_gpu.start()
    t_cpu.join()
    t_gpu.join()
    wall1 = time.perf_counter()

    if cpu_paths_done + gpu_paths_done != n_paths:
        raise RuntimeError(
            f"Path accounting error: cpu={cpu_paths_done} gpu={gpu_paths_done} total={n_paths}"
        )

    gpu_finals = np.concatenate(gpu_batches) if gpu_batches else np.zeros(0, dtype=np.float64)
    return (
        cpu_parts,
        gpu_finals,
        wall1 - wall0,
        cpu_batch_time,
        gpu_batch_time,
        cpu_paths_done,
        gpu_paths_done,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid CPU+GPU jump-diffusion benchmark (static split or dynamic batch queue)."
    )
    ap.add_argument("--input", required=True, help="OHLC CSV (timestamp, open, high, low, close).")
    ap.add_argument("--n-paths", type=int, default=10_000, help="Total Monte Carlo paths.")
    ap.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help="Time steps (bars). 0 = use min(len(csv), cap from --max-steps).",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="When --n-steps is 0, cap steps at this. 0 means use all rows.",
    )
    ap.add_argument(
        "--schedule",
        choices=("static", "dynamic"),
        default="static",
        help="static = fixed CPU/GPU counts; dynamic = shared batch queue (multiple GPU runs).",
    )
    ap.add_argument("--batch-cap", type=int, default=4096, help="Max paths claimed per batch (dynamic schedule).")
    ap.add_argument("--cpu-fraction", type=float, default=0.35, help="(static) CPU path fraction.")
    ap.add_argument("--n-cpu", type=int, default=None, help="(static) Exact CPU path count.")
    ap.add_argument("--n-gpu", type=int, default=None, help="(static) Exact GPU path count.")
    ap.add_argument("--seed", type=int, default=42, help="CPU seed; GPU uses seed + offset.")
    ap.add_argument("--gpu-seed-offset", type=int, default=1_000_003, help="Added to --seed for GPU RNG.")
    ap.add_argument("--chunk-size", type=int, default=256, help="CPU batch size (paths per chunk).")
    ap.add_argument("--jump-threshold-mult", type=float, default=2.5, help="Must match CUDA default for comparable fit.")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step per bar (matches synth scripts).")
    ap.add_argument("--cuda-bin", default="", help="Path to jump_diffusion_cuda; default build path or env.")
    ap.add_argument("--json-out", default="", help="Optional path to write timing + split metadata JSON.")
    ap.add_argument(
        "--cpu-runtime-seconds",
        type=float,
        default=0.0,
        help="Optional: baseline CPU runtime (seconds) to compute speedup for the hybrid table row.",
    )
    args = ap.parse_args()

    input_csv = Path(args.input)
    if not input_csv.is_file():
        raise SystemExit(f"Input not found: {input_csv}")

    # Measure end-to-end time including load+fit so it's comparable to CPU/GPU scripts.
    t_total0 = time.perf_counter()

    t_load0 = time.perf_counter()
    df_full = load_bars(input_csv)
    t_load1 = time.perf_counter()
    n_available = len(df_full)
    if args.n_steps and args.n_steps > 0:
        n_steps = min(n_available, args.n_steps)
    else:
        n_steps = n_available if int(args.max_steps) <= 0 else min(n_available, int(args.max_steps))
    if n_steps < 3:
        raise SystemExit("Need at least 3 steps after applying n_steps/max_steps.")

    df = df_full.iloc[:n_steps].copy()
    t_fit0 = time.perf_counter()
    params = fit_jump_diffusion_params(df, jump_threshold_mult=args.jump_threshold_mult)
    t_fit1 = time.perf_counter()

    n_paths = int(args.n_paths)
    if n_paths < 1:
        raise SystemExit("--n-paths must be >= 1")

    cuda_bin = Path(args.cuda_bin) if args.cuda_bin else _default_cuda_binary()
    gpu_base_seed = int(args.seed) + int(args.gpu_seed_offset)

    reports_dir = REPO_ROOT / "reports" / "jump_diffusion"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hybrid_jd_", dir=str(reports_dir)) as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        # Persist a truncated CSV so the CUDA fit sees the same rows as Python (first n_steps bars).
        trimmed_csv = tmpdir / "hybrid_input_trimmed.csv"
        t_save0 = time.perf_counter()
        df.to_csv(trimmed_csv, index=False)
        t_save1 = time.perf_counter()

        if args.schedule == "dynamic":
            if int(args.batch_cap) < 1:
                raise SystemExit("--batch-cap must be >= 1")
            cpu_parts, gpu_finals, hybrid_wall_s, cpu_sim_s, gpu_wall_s, n_cpu, n_gpu = _run_dynamic_hybrid(
                cuda_bin=cuda_bin,
                trimmed_csv=trimmed_csv,
                tmpdir=tmpdir,
                n_paths=n_paths,
                n_steps=n_steps,
                params=params,
                dt=float(args.dt),
                jump_threshold_mult=float(args.jump_threshold_mult),
                base_seed_cpu=int(args.seed),
                base_seed_gpu=gpu_base_seed,
                batch_cap=int(args.batch_cap),
                cpu_chunk_size=int(args.chunk_size),
            )
            cpu_finals = np.concatenate(cpu_parts) if cpu_parts else np.zeros(0, dtype=np.float64)
            merged = np.concatenate([cpu_finals, gpu_finals]) if (cpu_finals.size or gpu_finals.size) else np.zeros(0)
            ideal_parallel = max(cpu_sim_s, gpu_wall_s)
            serial_sum = cpu_sim_s + gpu_wall_s
            gpu_meta: Dict[str, Any] = {}
        else:
            n_cpu, n_gpu = _split_counts(n_paths, args.n_cpu, args.n_gpu, args.cpu_fraction)
            if n_cpu + n_gpu != n_paths:
                raise SystemExit("Internal split error: n_cpu + n_gpu != n_paths")

            wall0 = time.perf_counter()
            cpu_paths: Optional[np.ndarray] = None
            gpu_finals_arr: Optional[np.ndarray] = None
            cpu_sim_s = 0.0
            gpu_wall_s = 0.0
            gpu_meta = {}

            with ThreadPoolExecutor(max_workers=2) as ex:
                f_cpu = ex.submit(
                    _run_cpu_paths,
                    n_cpu=n_cpu,
                    n_steps=n_steps,
                    params=params,
                    dt=float(args.dt),
                    seed=int(args.seed),
                    chunk_size=int(args.chunk_size),
                )
                f_gpu = ex.submit(
                    _run_gpu_subprocess,
                    cuda_bin=cuda_bin,
                    input_csv=trimmed_csv,
                    n_gpu=n_gpu,
                    n_steps=n_steps,
                    seed=gpu_base_seed,
                    jump_threshold_mult=float(args.jump_threshold_mult),
                    dt=float(args.dt),
                    tmpdir=tmpdir,
                    artifact_prefix="hybrid_gpu",
                )

                for fut in as_completed([f_cpu, f_gpu]):
                    if fut is f_cpu:
                        cpu_paths, cpu_sim_s = fut.result()
                    else:
                        gpu_finals_arr, gpu_wall_s, gpu_meta = fut.result()

            wall1 = time.perf_counter()
            hybrid_wall_s = wall1 - wall0

            if cpu_paths is None or gpu_finals_arr is None:
                raise RuntimeError("CPU or GPU task did not complete.")

            cpu_finals = cpu_paths[:, -1] if cpu_paths.shape[0] else np.zeros(0, dtype=np.float64)
            gpu_finals = gpu_finals_arr
            merged = np.concatenate([cpu_finals, gpu_finals]) if (cpu_finals.size or gpu_finals.size) else np.zeros(0)
            ideal_parallel = max(cpu_sim_s, gpu_wall_s)
            serial_sum = cpu_sim_s + gpu_wall_s

        mean_f = float(np.mean(merged)) if merged.size else 0.0
        std_f = float(np.std(merged, ddof=1)) if merged.size > 1 else 0.0

        ideal_parallel = max(cpu_sim_s, gpu_wall_s)
        serial_sum = cpu_sim_s + gpu_wall_s

        t_total1 = time.perf_counter()
        total_elapsed_s = t_total1 - t_total0

        load_sv_seconds = t_load1 - t_load0
        fit_parameters_seconds = t_fit1 - t_fit0
        save_files_seconds = t_save1 - t_save0
        # Hybrid "simulation time" for the table: critical-path time while CPU+GPU compute overlap.
        # If you want pure wall during the concurrent section, that's `hybrid_wall_s`.
        gpu_sim_s = 0.0
        try:
            tb = (gpu_meta.get("timing_breakdown") or {}) if isinstance(gpu_meta, dict) else {}
            gpu_sim_s = float(tb.get("jump_diffusion_simulation_seconds") or 0.0)
        except Exception:
            gpu_sim_s = 0.0
        jump_diffusion_simulation_seconds = max(cpu_sim_s, gpu_sim_s if gpu_sim_s > 0 else gpu_wall_s)
        generate_plots_seconds = 0.0

        # Table metrics
        runtime_s = total_elapsed_s
        sim_s = jump_diffusion_simulation_seconds
        throughput = (n_paths / runtime_s) if runtime_s > 0 else 0.0
        compute_pct = (sim_s / runtime_s * 100.0) if runtime_s > 0 else 0.0
        comm_comp_ratio = ((runtime_s - sim_s) / sim_s) if sim_s > 0 else 0.0
        speedup = (float(args.cpu_runtime_seconds) / runtime_s) if args.cpu_runtime_seconds and runtime_s > 0 else None

        print(f"{'Schedule':>14} | {args.schedule}")
        print(f"{'Paths (total)':>14} | {n_paths}")
        print(f"{'n_steps':>14} | {n_steps}")
        print(f"{'CPU paths':>14} | {n_cpu}")
        print(f"{'GPU paths':>14} | {n_gpu}")
        if args.schedule == "dynamic":
            print(f"{'batch_cap':>14} | {args.batch_cap}")
        print(f"{'CPU sim (s)':>14} | {cpu_sim_s:.6f}")
        print(f"{'GPU wall (s)':>14} | {gpu_wall_s:.6f}")
        print(f"{'Hybrid wall (s)':>14} | {hybrid_wall_s:.6f}  (concurrent section only)")
        print(f"{'Total elapsed':>14} | {total_elapsed_s:.6f}  (load+fit+concurrent+overhead)")
        print(f"{'max(CPU,GPU)':>14} | {ideal_parallel:.6f}  (ideal overlap lower bound)")
        print(f"{'CPU+GPU sum':>14} | {serial_sum:.6f}  (if run back-to-back)")
        print(f"{'Throughput':>14} | {throughput:.1f} paths/s  (uses total elapsed)")
        print(f"{'Merged mean':>14} | {mean_f:.6g}")
        print(f"{'Merged std':>14} | {std_f:.6g}")
        print("\n--- Table metrics (hybrid) ---")
        print(f"{'Runtime (s)':>14} | {runtime_s:.6f}")
        print(f"{'Sim time (s)':>14} | {sim_s:.6f}")
        if speedup is not None:
            print(f"{'Speedup':>14} | {speedup:.3f}x  (vs --cpu-runtime-seconds)")
        print(f"{'Compute %':>14} | {compute_pct:.2f}%")
        print(f"{'Comm/Comp':>14} | {comm_comp_ratio:.6f}")

        if args.json_out:
            out = {
                "mode": "hybrid_jump_diffusion",
                "schedule": args.schedule,
                "input": str(input_csv),
                "trimmed_rows": n_steps,
                "n_paths": n_paths,
                "n_cpu": n_cpu,
                "n_gpu": n_gpu,
                "elapsed_seconds": total_elapsed_s,
                "timing_breakdown": {
                    "fit_parameters_seconds": fit_parameters_seconds,
                    "generate_plots_seconds": generate_plots_seconds,
                    "jump_diffusion_simulation_seconds": jump_diffusion_simulation_seconds,
                    "load_sv_seconds": load_sv_seconds,
                    "save_files_seconds": save_files_seconds,
                },
                "hybrid_concurrent_wall_seconds": hybrid_wall_s,
                "cpu_sim_seconds": cpu_sim_s,
                "gpu_wall_seconds": gpu_wall_s,
                "gpu_reported_sim_seconds": gpu_sim_s,
                "ideal_parallel_lower_bound_seconds": ideal_parallel,
                "serial_sum_seconds": serial_sum,
                "table_metrics": {
                    "runtime_seconds": runtime_s,
                    "sim_time_seconds": sim_s,
                    "throughput_paths_per_sec": throughput,
                    "compute_percent": compute_pct,
                    "comm_comp_ratio": comm_comp_ratio,
                    "speedup_vs_cpu": speedup,
                },
                "merged_final_mean": mean_f,
                "merged_final_std": std_f,
                "seed_cpu": int(args.seed),
                "seed_gpu": gpu_base_seed,
                "cuda_binary": str(cuda_bin),
            }
            if args.schedule == "dynamic":
                out["batch_cap"] = int(args.batch_cap)
            Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
