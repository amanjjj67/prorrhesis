"""The canonical feature schema (spec section 8.2).

Kept in its own module with no imports so both the feature store and the
models can use it without a circular dependency.
"""

FORM_STATS = ("gf", "ga", "shots", "sot", "xgf", "xga", "pts")
VENUE_STATS = ("gf", "ga", "xgf", "xga")
FORM_HALF_LIVES = (5, 12)
VENUE_HALF_LIFE = 8
PRIOR_N = 4

FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"f{hl}_{stat}_{side}" for hl in FORM_HALF_LIVES for stat in FORM_STATS for side in ("h", "a")),
    *(f"v_{stat}_{side}" for stat in VENUE_STATS for side in ("h", "a")),
    "days_since_h", "days_since_a",
    "m14_h", "m14_a", "m30_h", "m30_a",
    "season_progress", "league", "month", "new_promoted_h", "new_promoted_a",
    "elo_diff", "clubelo_diff",
    "mkt_open_h", "mkt_open_d", "mkt_open_a",
)
MARKET_FEATURE_COLUMNS: tuple[str, ...] = ("mkt_open_h", "mkt_open_d", "mkt_open_a")
