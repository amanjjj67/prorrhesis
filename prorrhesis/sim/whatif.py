"""Manual what-if (spec section 12.2).

The user moves sliders for one team and the Dixon-Coles lambdas of that team
are scaled before the grid is rebuilt. The result is a manual judgement, not
something learned from data, and it is never written to the ledger.
"""

from __future__ import annotations

import numpy as np

from ..models.dixoncoles import _grid_from_lambdas

ATTACK_OUT_PRESET = -0.08
DEFENCE_OUT_PRESET = -0.08
SLIDER_LIMIT = 0.25


def clamp(value: float) -> float:
    """Keep a slider inside the allowed range."""
    return float(min(max(float(value), -SLIDER_LIMIT), SLIDER_LIMIT))


def adjusted_grid(lh: float, la: float, rho: float, max_goals: int,
                  attack: float = 0.0, defence: float = 0.0,
                  side: str = "home") -> np.ndarray:
    """Grid with the attack and defence of one side scaled."""
    attack = clamp(attack)
    defence = clamp(defence)
    if side == "home":
        lh = float(lh) * (1.0 + attack)
        la = float(la) * (1.0 - defence)
    else:
        la = float(la) * (1.0 + attack)
        lh = float(lh) * (1.0 - defence)
    return _grid_from_lambdas(lh, la, rho, max_goals)


def grid_to_probs(grid: np.ndarray) -> tuple[float, float, float]:
    """H/D/A probabilities of a scoreline grid."""
    d = grid.shape[0] - 1
    ii, jj = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing="ij")
    pH = float(grid[ii > jj].sum())
    pD = float(np.trace(grid))
    pA = float(grid[ii < jj].sum())
    total = pH + pD + pA
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (pH / total, pD / total, pA / total)


def apply(lambdas: tuple[float, float, float], max_goals: int = 10,
          attack: float = 0.0, defence: float = 0.0,
          side: str = "home") -> dict:
    """Adjusted grid and probabilities for one match."""
    lh, la, rho = lambdas
    grid = adjusted_grid(lh, la, rho, max_goals, attack, defence, side)
    pH, pD, pA = grid_to_probs(grid)
    return {
        "pH": pH,
        "pD": pD,
        "pA": pA,
        "lambdas": [float(lh), float(la)],
        "adjustment": {
            "attack": clamp(attack),
            "defence": clamp(defence),
            "side": side,
        },
        "grid": grid.tolist(),
        "note": "Manual adjustment, not learned from data.",
    }
