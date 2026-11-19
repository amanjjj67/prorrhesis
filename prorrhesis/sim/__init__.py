"""Season simulation and manual what-if (spec section 12)."""

from .season import (
    SimulationResult,
    current_season,
    league_clubs,
    played_pairs,
    remaining_pairs,
    season_teams,
    simulate,
)
from .whatif import apply, clamp, grid_to_probs

__all__ = [
    "SimulationResult",
    "apply",
    "clamp",
    "current_season",
    "grid_to_probs",
    "league_clubs",
    "played_pairs",
    "remaining_pairs",
    "season_teams",
    "simulate",
]
