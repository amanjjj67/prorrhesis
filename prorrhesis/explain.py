"""Explanations (spec section 11).

Every sentence here is produced by code from the model's own numbers: a
feature name is mapped through a phrase table, a Dixon-Coles lambda is compared
with the league mean, and disagreement is measured as the spread of `pH`
across the models. No language model is involved.
"""

from __future__ import annotations

DISAGREEMENT_THRESHOLD = 0.06

# feature -> (phrase when the value is high, phrase when the value is low)
FEATURE_PHRASES: dict[str, tuple[str, str]] = {
    "elo_diff": ("Home side is rated clearly higher", "Away side is rated clearly higher"),
    "clubelo_diff": ("ClubElo also rates the home side higher",
                     "ClubElo also rates the away side higher"),
    "days_since_h": ("Home side has had a long break", "Home side has had little rest"),
    "days_since_a": ("Away side has had a long break", "Away side has had little rest"),
    "season_progress": ("The season is well advanced", "The season has barely started"),
    "new_promoted_h": ("The home side is newly promoted", "The home side is not newly promoted"),
    "new_promoted_a": ("The away side is newly promoted", "The away side is not newly promoted"),
    "mkt_open_h": ("The market prices the home side as the favourite",
                   "The market prices the home side as the outsider"),
    "mkt_open_a": ("The market prices the away side as the favourite",
                   "The market prices the away side as the outsider"),
}

# feature-group phrases keyed by the statistic they belong to
_STAT_PHRASES: dict[str, tuple[str, str]] = {
    "gf": ("{side} side has scored freely lately", "{side} side has been sterile lately"),
    "ga": ("{side} side leaks goals", "{side} side's defence holds up"),
    "xgf": ("{side} side's chances have been worth a lot", "{side} side has not been creating much"),
    "xga": ("{side} side is conceding high-quality chances", "{side} side is not conceding much"),
    "pts": ("{side} side is in good form", "{side} side is out of form"),
    "shots": ("{side} side is shooting a lot", "{side} side is barely shooting"),
    "sot": ("{side} side is getting shots on target", "{side} side is not testing the keeper"),
}


def _feature_parts(name: str) -> tuple[str, str | None] | None:
    """Split a feature name into its statistic and side, when it has one."""
    side = None
    if name.endswith("_h"):
        name, side = name[:-2], "Home"
    elif name.endswith("_a"):
        name, side = name[:-2], "Away"
    if name.startswith("f5_") or name.startswith("f12_"):
        name = name.split("_", 1)[1]
    if name.startswith("v_"):
        name = name[2:]
    return name, side


def feature_phrase(name: str, value: float, reference: float | None = None) -> str:
    """One sentence about a single feature value."""
    if value is None or value != value:  # NaN
        return ""
    direct = FEATURE_PHRASES.get(name)
    if direct is not None:
        high, low = direct
        if reference is None:
            reference = 0.0
        return high if value >= reference else low

    parts = _feature_parts(name)
    if parts is None:
        return ""
    stat, side = parts
    phrases = _STAT_PHRASES.get(stat)
    if phrases is None or side is None:
        return ""
    high, low = phrases
    if reference is None:
        reference = 0.0
    return (high if value >= reference else low).format(side=side)


def gbm_reasons(contributions: dict[str, float], limit: int = 3,
                reference: dict[str, float] | None = None) -> list[str]:
    """Sentences for the three features that push the prediction hardest."""
    reference = reference or {}
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:limit]
    out = []
    for name, value in ranked:
        sentence = feature_phrase(name, abs(value), reference.get(name))
        if sentence:
            out.append(sentence)
    return out


def dc_reason(lambdas: tuple[float, float, float],
               league_mean: tuple[float, float] | None = None) -> str:
    """Compare the expected goals with the league mean."""
    lh, la = float(lambdas[0]), float(lambdas[1])
    if league_mean is None:
        return f"Expected goals {lh:.1f} vs {la:.1f}"
    mh, ma = float(league_mean[0]), float(league_mean[1])
    return (f"Expected goals {lh:.1f} vs {la:.1f} "
            f"(league average {mh:.1f} vs {ma:.1f})")


def _triple_of(entry) -> tuple[float, float, float] | None:
    """The (pH, pD, pA) of one model entry, or None when it abstained."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        if str(entry.get("status", "ok")) != "ok":
            return None
        values = (entry.get("pH"), entry.get("pD"), entry.get("pA"))
    else:
        values = tuple(entry)
    if len(values) != 3:
        return None
    try:
        out = (float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError):
        return None
    if not all(x == x for x in out):
        return None
    return out


def disagreement(triples: dict) -> dict | None:
    """Flag a match the models read differently."""
    usable = {}
    for name, entry in triples.items():
        values = _triple_of(entry)
        if values is not None:
            usable[name] = values
    if len(usable) < 2:
        return None
    spread = max(v[0] for v in usable.values()) - min(v[0] for v in usable.values())
    if spread <= DISAGREEMENT_THRESHOLD:
        return None
    highest = max(usable.items(), key=lambda kv: kv[1][0])
    lowest = min(usable.items(), key=lambda kv: kv[1][0])
    return {
        "spread": float(spread),
        "highest": {"model": highest[0], "pH": float(highest[1][0])},
        "lowest": {"model": lowest[0], "pH": float(lowest[1][0])},
        "text": (f"Models disagree on this match: {highest[0]} rates the home side at "
                 f"{highest[1][0]:.1%}, {lowest[0]} at {lowest[1][0]:.1%}."),
    }


def reliability_bins(predicted: list[float], observed: list[float],
                     bins: int = 10) -> list[dict]:
    """Reliability diagram bins (spec section 14.2, Calibration screen)."""
    import numpy as np

    p = np.asarray(predicted, dtype=float)
    o = np.asarray(observed, dtype=float)
    keep = np.isfinite(p) & np.isfinite(o)
    p, o = p[keep], o[keep]
    out = []
    if len(p) == 0:
        return out
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not sel.any():
            continue
        out.append({
            "low": float(lo),
            "high": float(hi),
            "count": int(sel.sum()),
            "mean_predicted": float(p[sel].mean()),
            "observed_rate": float(o[sel].mean()),
        })
    return out
