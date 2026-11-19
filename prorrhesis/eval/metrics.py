"""Evaluation metrics (spec section 10.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def brier(P: np.ndarray, y: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    Y = np.eye(3)[y]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def rps(P: np.ndarray, y: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    Y = np.eye(3)[y]
    cp = np.cumsum(P, axis=1)[:, :2]
    cy = np.cumsum(Y, axis=1)[:, :2]
    return float(np.mean(0.5 * np.sum((cp - cy) ** 2, axis=1)))


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(P.argmax(1) == y))


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    # Handle 1D input (single outcome probability)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.shape[1] == 1:
        pk = P.ravel()
        obs = (y == 0).astype(float)  # single outcome: check if y matches outcome 0
        order = np.argsort(pk)
        chunks = np.array_split(order, bins)
        total = 0.0
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            conf = float(pk[chunk].mean())
            acc = float(obs[chunk].mean())
            total += (len(chunk) / len(y)) * abs(acc - conf)
        return float(total)
    # 2D input with 3 columns (H, D, A)
    total = 0.0
    for k in range(3):
        pk = P[:, k]
        obs = (y == k).astype(float)
        order = np.argsort(pk)
        chunks = np.array_split(order, bins)
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            conf = float(pk[chunk].mean())
            acc = float(obs[chunk].mean())
            total += (len(chunk) / len(y)) * abs(acc - conf)
    return float(total)


def reliability_bins(P: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Per-outcome reliability table with Wilson intervals."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    out = []
    for k in range(3):
        pk = P[:, k]
        obs = (y == k).astype(float)
        order = np.argsort(pk)
        for chunk in np.array_split(order, bins):
            if len(chunk) == 0:
                continue
            conf = float(pk[chunk].mean())
            acc = float(obs[chunk].mean())
            n = len(chunk)
            out.append({
                "outcome": k,
                "bin": len(out),
                "n": n,
                "predicted": conf,
                "observed": acc,
                "wilson_low": _wilson_low(acc, n),
                "wilson_high": _wilson_high(acc, n),
            })
    return out


def _wilson_low(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, (centre - margin) / denom))


def _wilson_high(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + margin) / denom))


def skill_score(ll_model: float, ll_baseline: float) -> float:
    """1 - LL_model / LL_baseline."""
    if not np.isfinite(ll_baseline) or ll_baseline == 0:
        return float("nan")
    return float(1.0 - ll_model / ll_baseline)


def per_group_metrics(P: np.ndarray, y: np.ndarray, groups: pd.Series) -> dict:
    out = {}
    for key, idx in groups.groupby(groups).groups.items():
        positions = groups.index.get_indexer(idx)
        out[key] = {
            "n": len(positions),
            "log_loss": log_loss(P[positions], y[positions]),
            "brier": brier(P[positions], y[positions]),
            "rps": rps(P[positions], y[positions]),
            "accuracy": accuracy(P[positions], y[positions]),
        }
    return out
