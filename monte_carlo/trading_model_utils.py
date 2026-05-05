from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]


@dataclass
class BarModelParams:
    model: str
    s0: float
    mu: float
    sigma: float
    extra: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def load_bars(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        bad = df[df["timestamp"].isna()].index.tolist()[:5]
        raise ValueError(f"Invalid timestamps in rows: {bad}")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Found non-numeric OHLC values.")
    return df.sort_values("timestamp").reset_index(drop=True)


def close_prices_to_log_returns(close: pd.Series) -> np.ndarray:
    prices = np.asarray(close, dtype=float)
    if prices.ndim != 1 or len(prices) < 2:
        raise ValueError("Need at least two close prices.")
    if np.any(prices <= 0):
        raise ValueError("Close prices must be positive.")
    return np.diff(np.log(prices))


def estimate_basic_return_stats(close: pd.Series) -> Tuple[float, float, np.ndarray]:
    r = close_prices_to_log_returns(close)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return mu, sigma, r


def detect_jump_residuals(r: np.ndarray, threshold_mult: float = 2.5) -> tuple[np.ndarray, np.ndarray, float]:
    if r.size < 3:
        raise ValueError("Need at least 3 returns to estimate jumps.")
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    threshold = threshold_mult * sigma
    residuals = r - mu
    jump_mask = np.abs(residuals) > threshold
    return residuals[~jump_mask], residuals[jump_mask], threshold


def make_ohlc_from_close_path(
    timestamps: pd.DatetimeIndex | pd.Series,
    close_path: np.ndarray,
    base_spread_fraction: float = 0.15,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    close_path = np.asarray(close_path, dtype=float)
    if close_path.ndim != 1 or len(close_path) < 2:
        raise ValueError("close_path must be a 1D array with at least 2 points.")
    if np.any(close_path <= 0):
        raise ValueError("Synthetic prices must remain positive.")

    ts = pd.DatetimeIndex(timestamps)
    if len(ts) != len(close_path):
        raise ValueError("timestamps and close_path must have the same length.")

    opens = np.empty_like(close_path)
    highs = np.empty_like(close_path)
    lows = np.empty_like(close_path)

    opens[0] = close_path[0]
    highs[0] = close_path[0]
    lows[0] = close_path[0]

    for i in range(1, len(close_path)):
        o = close_path[i - 1]
        c = close_path[i]
        move = abs(c - o)
        noise = float(abs(rng.normal(0.0, 1.0)))
        wick = max(move * base_spread_fraction, 1e-8) * (1.0 + 0.75 * noise)
        highs[i] = max(o, c) + wick
        lows[i] = max(min(o, c) - wick, 1e-8)
        opens[i] = o

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_path,
        }
    )
    return out


def save_bars(df: pd.DataFrame, out_path: str | Path) -> None:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_csv(out_path, index=False)


def infer_step_seconds(timestamps: pd.Series) -> float:
    ts = pd.to_datetime(timestamps, utc=True)
    deltas = ts.diff().dropna().dt.total_seconds().to_numpy()
    if len(deltas) == 0:
        raise ValueError("Need at least two timestamps to infer step size.")
    return float(np.median(deltas))