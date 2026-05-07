"""
Hybrid CPU + GPU Heston Monte Carlo benchmark.

Two schedules:
  * static  — fixed n_cpu / n_gpu (or cpu-fraction); one standard heston_gpu run.
  * dynamic — shared pool of paths; CPU and GPU workers claim batches until
              exhausted. Each GPU batch is a normal heston_gpu subprocess (same
              CLI as GPU-only); repeats load+fit per batch (see report for cost).

Portable layout (e.g. copy to CARC in one folder):
  hybrid_heston.py, hybrid_heston.sl, heston_synth.py, trading_model_utils.py,
  heston_gpu (executable) or heston_gpu.cu (built by hybrid_heston.sl), your OHLC CSV.

Set HYBRID_ROOT to that folder if the job cwd is not the bundle (optional).

Repo layout (still supported): this file under monte_carlo/hybrid/ with sibling monte_carlo/heston_synth.py.

Override binary: HESTON_CUDA_BIN=/path/to/heston_gpu

Requires Python 3.7+ (``trading_model_utils`` uses dataclasses). On CARC/Discovery the default
``python3`` after ``module purge`` may be 3.6 — load a CARC Python module (e.g. ``python/3.11.9``; see hybrid_heston.sl).
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
from typing import List, Optional, Tuple

import numpy as np

if sys.version_info < (3, 7):
    raise SystemExit(
        "hybrid_heston.py requires Python 3.7+ (dataclasses). "
        "On CARC: module load python/3.11.9   (or: module spider python). "
        "Current interpreter: {}".format(sys.executable)
    )

_HERE = Path(__file__).resolve().parent
_env_hybrid = os.environ.get("HYBRID_ROOT", "").strip()
HYBRID_ROOT = Path(_env_hybrid).resolve() if _env_hybrid else _HERE

# Repo checkout: monte_carlo/hybrid/hybrid_heston.py
if (
    _HERE.name == "hybrid"
    and _HERE.parent.name == "monte_carlo"
    and (_HERE.parent / "heston_synth.py").is_file()
):
    MONTE_CARLO = _HERE.parent
    WORK_ROOT = MONTE_CARLO.parent
    for p in (str(WORK_ROOT), str(MONTE_CARLO)):
        if p not in sys.path:
            sys.path.insert(0, p)
else:
    WORK_ROOT = HYBRID_ROOT
    if str(HYBRID_ROOT) not in sys.path:
        sys.path.insert(0, str(HYBRID_ROOT))

import heston_synth  # noqa: E402
from trading_model_utils import BarModelParams, load_bars  # noqa: E402


def _heston_params_dict(params: BarModelParams) -> dict:
    ex = params.extra or {}
    return {
        "model": params.model,
        "s0": params.s0,
        "mu": params.mu,
        "sigma": params.sigma,
        "v0": ex.get("v0"),
        "theta": ex.get("theta"),
        "kappa": ex.get("kappa"),
        "xi": ex.get("xi"),
        "rho": ex.get("rho"),
    }


def _default_cuda_binary() -> Path:
    if os.environ.get("HESTON_CUDA_BIN"):
        return Path(os.environ["HESTON_CUDA_BIN"])
    candidates = [
        HYBRID_ROOT / "heston_gpu",
        WORK_ROOT / "heston_gpu",
        WORK_ROOT / "GPU" / "build" / "heston_gpu",
        WORK_ROOT / "monte_carlo" / "GPU" / "build" / "heston_gpu",
    ]
    for base in candidates:
        if base.is_file():
            return base
        if sys.platform == "win32":
            win = base.with_suffix(".exe")
            if win.is_file():
                return win
    return candidates[0]


def _gpu_binary_supports_final_prices_csv(cuda_bin: Path) -> bool:
    """True if this heston_gpu was built with --output-final-prices (required by the hybrid driver)."""
    if not cuda_bin.is_file():
        return True
    proc = subprocess.run(
        [str(cuda_bin)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return "output-final-prices" in combined


def _split_counts(
    n_paths: int, n_cpu: Optional[int], n_gpu: Optional[int], cpu_fraction: float
) -> Tuple[int, int]:
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
    paths = heston_synth.simulate_heston_paths(
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
    seed: int,
    block_size: int,
    tmpdir: Path,
    artifact_prefix: str = "hybrid_gpu",
) -> Tuple[np.ndarray, float]:
    if n_gpu <= 0:
        return np.zeros(0, dtype=np.float64), 0.0
    if not cuda_bin.is_file():
        raise FileNotFoundError(
            f"CUDA binary not found at {cuda_bin}. "
            f"Set HESTON_CUDA_BIN or place heston_gpu under HYBRID_ROOT ({HYBRID_ROOT})."
        )

    bars_out = tmpdir / f"{artifact_prefix}_bars.csv"
    params_out = tmpdir / f"{artifact_prefix}_params.json"
    # Write per-GPU-run finals next to the job / cwd (not under tempfile), so outputs stay in
    # the directory you cd to before sbatch. Names stay unique per batch (dynamic prefixes).
    _jid = os.environ.get("SLURM_JOB_ID", "").strip()
    _stem = f"{artifact_prefix}_{_jid}" if _jid else artifact_prefix
    finals_out = (Path.cwd() / f"{_stem}_final_prices.csv").resolve()

    cmd = [
        str(cuda_bin),
        "--input",
        str(input_csv.resolve()),
        "--n-paths",
        str(n_gpu),
        "--seed",
        str(seed),
        "--block-size",
        str(block_size),
        "--out-bars",
        str(bars_out.resolve()),
        "--out-params",
        str(params_out.resolve()),
        "--output-final-prices",
        str(finals_out),
        "--no-bars",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(WORK_ROOT),
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
        tail = (proc.stderr or proc.stdout or "").strip()
        more = f"\n--- heston_gpu output ---\n{tail}\n" if tail else ""
        raise RuntimeError(
            f"GPU run did not write final prices: {finals_out}.{more}\n"
            f"Most often: {cuda_bin} is an old build without --output-final-prices. "
            f"Rebuild from the project heston_gpu.cu (see hybrid_heston.sl / nvcc line) and replace the binary."
        )

    finals = np.loadtxt(finals_out, dtype=np.float64, skiprows=1)
    if finals.ndim == 0:
        finals = np.array([float(finals)], dtype=np.float64)
    if finals.shape[0] != n_gpu:
        raise RuntimeError(
            f"Expected {n_gpu} GPU final prices, got {finals.shape[0]} from {finals_out}"
        )
    return finals, t1 - t0


def _run_dynamic_hybrid(
    *,
    cuda_bin: Path,
    trimmed_csv: Path,
    tmpdir: Path,
    n_paths: int,
    n_steps: int,
    params,
    dt: float,
    base_seed_cpu: int,
    base_seed_gpu: int,
    batch_cap: int,
    block_size: int,
    cpu_chunk_size: int,
) -> Tuple[List[np.ndarray], np.ndarray, float, float, float, int, int]:
    """
    Dynamic batch queue: CPU and GPU threads share a path budget (thread-safe claims).
    GPU runs the stock ``heston_gpu`` binary once per claimed batch (full load+fit+kernel each time).
    """
    if not cuda_bin.is_file():
        raise FileNotFoundError(
            f"CUDA binary not found at {cuda_bin}. "
            f"Set HESTON_CUDA_BIN or place heston_gpu under HYBRID_ROOT ({HYBRID_ROOT})."
        )

    remaining = n_paths
    lock = threading.Lock()
    cpu_parts: List[np.ndarray] = []
    gpu_batches: List[np.ndarray] = []
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
            paths = heston_synth.simulate_heston_paths(
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
            finals, elapsed = _run_gpu_subprocess(
                cuda_bin=cuda_bin,
                input_csv=trimmed_csv,
                n_gpu=take,
                seed=seed_b,
                block_size=block_size,
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
        description="Hybrid CPU+GPU Heston benchmark (static split or dynamic batch queue)."
    )
    ap.add_argument("--input", required=True, help="OHLC CSV (timestamp, open, high, low, close).")
    ap.add_argument("--n-paths", type=int, default=10_000, help="Total Monte Carlo paths.")
    ap.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help="Number of bars (steps). 0 = use min(len(csv), --max-steps).",
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
        help="static = fixed CPU/GPU counts; dynamic = shared batch queue (multiple heston_gpu runs).",
    )
    ap.add_argument(
        "--batch-cap",
        type=int,
        default=4096,
        help="Max paths claimed per batch (dynamic schedule).",
    )
    ap.add_argument("--cpu-fraction", type=float, default=0.35, help="(static) CPU path fraction.")
    ap.add_argument("--n-cpu", type=int, default=None, help="(static) Exact CPU path count.")
    ap.add_argument("--n-gpu", type=int, default=None, help="(static) Exact GPU path count.")
    ap.add_argument("--seed", type=int, default=42, help="CPU base seed.")
    ap.add_argument("--gpu-seed-offset", type=int, default=1_000_003, help="Added to --seed for GPU base.")
    ap.add_argument("--chunk-size", type=int, default=256, help="CPU chunk size (static or per-batch cap).")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step per bar.")
    ap.add_argument("--block-size", type=int, default=256, help="CUDA threads per block.")
    ap.add_argument("--cuda-bin", default="", help="Path to heston_gpu.")
    ap.add_argument("--json-out", default="", help="Optional timing JSON path.")
    ap.add_argument(
        "--merged-final-prices-out",
        default="",
        help="Optional CSV (header final_price, one column). Static: CPU paths then GPU paths. "
        "Dynamic: all CPU batch finals concatenated, then all GPU batch finals (not global path index). "
        "Relative paths use the process cwd (submit dir if you cd there before sbatch).",
    )
    args = ap.parse_args()

    input_csv = Path(args.input)
    if not input_csv.is_file():
        raise SystemExit(f"Input not found: {input_csv}")

    df_full = load_bars(input_csv)
    n_available = len(df_full)
    if args.n_steps and args.n_steps > 0:
        n_steps = min(n_available, args.n_steps)
    else:
        n_steps = n_available if int(args.max_steps) <= 0 else min(n_available, int(args.max_steps))
    if n_steps < 3:
        raise SystemExit("Need at least 3 steps after applying n_steps/max_steps.")

    df = df_full.iloc[:n_steps].copy()
    params = heston_synth.fit_heston_params(df)

    n_paths = int(args.n_paths)
    if n_paths < 1:
        raise SystemExit("--n-paths must be >= 1")

    cuda_bin = Path(args.cuda_bin) if args.cuda_bin else _default_cuda_binary()
    gpu_base_seed = int(args.seed) + int(args.gpu_seed_offset)

    if not _gpu_binary_supports_final_prices_csv(cuda_bin):
        raise SystemExit(
            f"GPU binary does not advertise --output-final-prices (required for hybrid):\n  {cuda_bin}\n"
            f"Rebuild heston_gpu from the repo's heston_gpu.cu on the GPU node, e.g.\n"
            f"  nvcc -O3 -arch=sm_70 -o heston_gpu heston_gpu.cu -lcurand -std=c++17\n"
            f"Then place the binary in {HYBRID_ROOT} or set HESTON_CUDA_BIN."
        )

    # Scratch for trimmed CSV + per-batch GPU artifacts; removed when the run finishes.
    with tempfile.TemporaryDirectory(prefix="hybrid_heston_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        trimmed_csv = tmpdir / "hybrid_input_trimmed.csv"
        df.to_csv(trimmed_csv, index=False)

        if args.schedule == "dynamic":
            if int(args.batch_cap) < 1:
                raise SystemExit("--batch-cap must be >= 1")
            cpu_parts, gpu_finals, hybrid_wall_s, cpu_sim_s, gpu_phase_s, n_cpu, n_gpu = _run_dynamic_hybrid(
                cuda_bin=cuda_bin,
                trimmed_csv=trimmed_csv,
                tmpdir=tmpdir,
                n_paths=n_paths,
                n_steps=n_steps,
                params=params,
                dt=float(args.dt),
                base_seed_cpu=int(args.seed),
                base_seed_gpu=gpu_base_seed,
                batch_cap=int(args.batch_cap),
                block_size=int(args.block_size),
                cpu_chunk_size=int(args.chunk_size),
            )
            cpu_finals = np.concatenate(cpu_parts) if cpu_parts else np.zeros(0, dtype=np.float64)
            merged = np.concatenate([cpu_finals, gpu_finals]) if (cpu_finals.size or gpu_finals.size) else np.zeros(0)
            ideal_parallel = max(cpu_sim_s, gpu_phase_s)
            serial_sum = cpu_sim_s + gpu_phase_s
        else:
            n_cpu, n_gpu = _split_counts(n_paths, args.n_cpu, args.n_gpu, args.cpu_fraction)
            if n_cpu + n_gpu != n_paths:
                raise SystemExit("Internal split error: n_cpu + n_gpu != n_paths")

            wall0 = time.perf_counter()
            cpu_paths: Optional[np.ndarray] = None
            gpu_finals_arr: Optional[np.ndarray] = None
            cpu_sim_s = 0.0
            gpu_phase_s = 0.0

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
                    seed=gpu_base_seed,
                    block_size=int(args.block_size),
                    tmpdir=tmpdir,
                )

                for fut in as_completed([f_cpu, f_gpu]):
                    if fut is f_cpu:
                        cpu_paths, cpu_sim_s = fut.result()
                    else:
                        gpu_finals_arr, gpu_phase_s = fut.result()

            wall1 = time.perf_counter()
            hybrid_wall_s = wall1 - wall0

            if cpu_paths is None or gpu_finals_arr is None:
                raise RuntimeError("CPU or GPU task did not complete.")

            cpu_finals = cpu_paths[:, -1] if cpu_paths.shape[0] else np.zeros(0, dtype=np.float64)
            gpu_finals = gpu_finals_arr
            merged = np.concatenate([cpu_finals, gpu_finals]) if (cpu_finals.size or gpu_finals.size) else np.zeros(0)
            ideal_parallel = max(cpu_sim_s, gpu_phase_s)
            serial_sum = cpu_sim_s + gpu_phase_s

        mean_f = float(np.mean(merged)) if merged.size else 0.0
        std_f = float(np.std(merged, ddof=1)) if merged.size > 1 else 0.0

        hp = _heston_params_dict(params)
        print("\n--- Fitted Heston parameters (shared CPU + GPU) ---")
        for k in ("s0", "mu", "sigma", "v0", "theta", "kappa", "xi", "rho"):
            v = hp.get(k)
            if v is not None:
                print(f"  {k:8} = {v:.8g}")

        print(f"\n{'Schedule':>14} | {args.schedule}")
        print(f"{'Paths (total)':>14} | {n_paths}")
        print(f"{'n_steps':>14} | {n_steps}")
        print(f"{'CPU paths':>14} | {n_cpu}")
        print(f"{'GPU paths':>14} | {n_gpu}")
        if args.schedule == "dynamic":
            print(f"{'batch_cap':>14} | {args.batch_cap}")
        print(f"{'CPU sim sum':>14} | {cpu_sim_s:.6f}  (sum of batch times; overlaps GPU)")
        print(f"{'GPU phase sum':>14} | {gpu_phase_s:.6f}  (sum of batch waits; overlaps CPU)")
        print(f"{'Hybrid wall (s)':>14} | {hybrid_wall_s:.6f}")
        print(f"{'max(sum sums)':>14} | {ideal_parallel:.6f}  (loose lower bound)")
        print(f"{'CPU+GPU sums':>14} | {serial_sum:.6f}  (ignores overlap)")
        print(f"{'Throughput':>14} | {n_paths / hybrid_wall_s:.1f} paths/s")
        print(f"{'Merged mean':>14} | {mean_f:.6g}")
        print(f"{'Merged std':>14} | {std_f:.6g}")

        print("\n--- Timing breakdown (seconds) ---")
        print(f"  cpu_sim_sum_seconds       : {cpu_sim_s:.6f}  (CPU path simulation; may overlap GPU)")
        print(f"  gpu_batch_sum_seconds     : {gpu_phase_s:.6f}  (GPU subprocess wall; may overlap CPU)")
        print(f"  hybrid_wall_seconds       : {hybrid_wall_s:.6f}  (end-to-end hybrid)")
        print(f"  ideal_parallel_lower_bound: {ideal_parallel:.6f}  (max of the two sums)")
        print(f"  serial_sum_seconds        : {serial_sum:.6f}  (cpu + gpu sums, ignores overlap)")
        tp = n_paths / hybrid_wall_s if hybrid_wall_s > 0 else None
        if tp is not None:
            print(f"  throughput_paths_per_sec  : {tp:.1f}")

        merged_final_csv = ""
        if args.merged_final_prices_out:
            mp = Path(args.merged_final_prices_out).expanduser()
            if not mp.is_absolute():
                mp = Path.cwd() / mp
            mp.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(str(mp), merged, fmt="%.17g", header="final_price", comments="")
            merged_final_csv = str(mp.resolve())
            print(f"\nWrote merged final prices ({merged.size} rows) -> {merged_final_csv}")

        if args.json_out:
            out: dict = {
                "mode": "hybrid_heston",
                "schedule": args.schedule,
                "input": str(input_csv),
                "trimmed_rows": n_steps,
                "n_paths": n_paths,
                "n_cpu_paths": n_cpu,
                "n_gpu_paths": n_gpu,
                "heston_params": hp,
                "timing": {
                    "cpu_sim_sum_seconds": cpu_sim_s,
                    "gpu_batch_sum_seconds": gpu_phase_s,
                    "hybrid_wall_seconds": hybrid_wall_s,
                    "ideal_parallel_lower_bound_seconds": ideal_parallel,
                    "serial_sum_seconds": serial_sum,
                    "throughput_paths_per_sec": n_paths / hybrid_wall_s if hybrid_wall_s > 0 else None,
                },
                "cpu_sim_sum_seconds": cpu_sim_s,
                "gpu_batch_sum_seconds": gpu_phase_s,
                "hybrid_wall_seconds": hybrid_wall_s,
                "ideal_parallel_lower_bound_seconds": ideal_parallel,
                "serial_sum_seconds": serial_sum,
                "throughput_paths_per_sec": n_paths / hybrid_wall_s if hybrid_wall_s > 0 else None,
                "merged_final_mean": mean_f,
                "merged_final_std": std_f,
                "seed_cpu": int(args.seed),
                "seed_gpu_base": gpu_base_seed,
                "cuda_binary": str(cuda_bin),
                "dt": float(args.dt),
                "block_size": int(args.block_size),
                "chunk_size": int(args.chunk_size),
            }
            if merged_final_csv:
                out["merged_final_prices_csv"] = merged_final_csv
            if args.schedule == "dynamic":
                out["batch_cap"] = int(args.batch_cap)
            Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
