"""Block bootstrap for model comparison (spec section 10.4).

Whole 7-day blocks are resampled with replacement, so the correlation between
matches of the same week is kept. "A is better" is only claimed when the
interval of the mean loss difference excludes zero.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 5000
DEFAULT_SEED = 42


def _block_sums(diff: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    sums = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    np.add.at(sums, inverse, diff)
    np.add.at(counts, inverse, 1.0)
    return sums, counts


def block_bootstrap_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    blocks: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """Confidence interval of the mean per-match loss difference (a - b)."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("loss vectors must have the same length")
    if len(a) == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": 0, "resamples": 0, "significant": False}

    diff = a - b
    sums, counts = _block_sums(diff, blocks)
    n_blocks = len(sums)
    if n_blocks == 0:
        return {"mean_diff": float(np.mean(diff)), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": len(diff), "resamples": 0,
                "significant": False}

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    sample_sums = sums[picks].sum(axis=1)
    sample_counts = counts[picks].sum(axis=1)
    means = np.divide(sample_sums, sample_counts, out=np.full(n_resamples, np.nan),
                      where=sample_counts > 0)
    means = means[np.isfinite(means)]

    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(means, lo_q)) if len(means) else float("nan")
    ci_high = float(np.percentile(means, hi_q)) if len(means) else float("nan")
    mean_diff = float(np.mean(diff))
    significant = bool(np.isfinite(ci_low) and np.isfinite(ci_high)
                       and (ci_low > 0.0 or ci_high < 0.0))
    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(diff)),
        "blocks": int(n_blocks),
        "resamples": int(len(means)),
        "significant": significant,
        "a_better": bool(ci_high < 0.0),
        "b_better": bool(ci_low > 0.0),
    }


def block_bootstrap_mean_ci(
    loss: np.ndarray,
    blocks: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """Confidence interval of the mean loss of one model."""
    x = np.asarray(loss, dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": 0, "blocks": 0, "resamples": 0}
    sums, counts = _block_sums(x, blocks)
    n_blocks = len(sums)
    if n_blocks == 0:
        return {"mean": float(np.mean(x)), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": len(x), "blocks": 0, "resamples": 0}

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    sample_sums = sums[picks].sum(axis=1)
    sample_counts = counts[picks].sum(axis=1)
    means = np.divide(sample_sums, sample_counts,
                      out=np.full(n_resamples, np.nan), where=sample_counts > 0)
    means = means[np.isfinite(means)]

    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.percentile(means, lo_q)) if len(means) else float("nan"),
        "ci_high": float(np.percentile(means, hi_q)) if len(means) else float("nan"),
        "n": int(len(x)),
        "blocks": int(n_blocks),
        "resamples": int(len(means)),
    }


def per_match_loss(P: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Negative log probability of the realised outcome, per match."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return -np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))
