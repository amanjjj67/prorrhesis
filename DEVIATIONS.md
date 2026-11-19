# DEVIATIONS

Every deviation from `PRORRHESIS_SPEC.md` is listed here with the reason.
Nothing in the spec was silently changed.

## D001 - Python environment: global interpreter instead of a project virtualenv

**Spec:** section 19 sets up `python -m venv .venv` inside the project folder.

**What was done:** the package is installed into the machine's global Python
3.12.10 environment (`pandas 3.0.6`, `numpy 2.5.4`, `scipy 1.18.1`,
`scikit-learn 1.9.1`, `lightgbm 4.7.0`, `fastapi`, `uvicorn`, `rapidfuzz`).

**Why:** this was an explicit choice made by the owner of the project when the
environment was set up, to keep the run commands short on Windows.

**Effect:** no functional difference. `python -m prorrhesis ...` and
`python -m pytest ...` are run from the project folder exactly as in the spec.

## D002 - ClubElo is optional and may be absent

**Spec:** section 8.3 lists `clubelo_diff` as a feature; section 4 requires the
app to degrade gracefully when a data source fails.

**What was done:** the ClubElo adapter, its cache and its degradation path are
implemented and covered by `tests/test_clubelo.py`. Whether the ratings are
actually present depends on whether `http://api.clubelo.com` is reachable from
the machine that runs the app. When the service cannot be reached, the feature
is disabled, `clubelo_diff` is entirely NaN in the evaluation window, and every
other model keeps working.

**Effect:** `clubelo_diff` is the only column allowed to be entirely NaN in
the evaluation window (the M4 acceptance criterion names it explicitly).

**Measured:** `python -m prorrhesis clubelo fetch` answers with `HTTP 502` for
every snapshot date on this machine, so the feature is off and the app runs
without it. The fetch stops after three consecutive failures instead of
retrying all 640 weekly snapshots.

## D003 - Understat xG is not enabled

**Spec:** section 8.2 allows an optional Understat adapter for real expected
goals; `features.use_understat_xg` defaults to `false` in the appendix.

**What was done:** the xG-Poisson model falls back to the shot-based xG proxy
(`xg_proxy`) when no Understat table is present. No Understat scraping code is
shipped, so the `use_understat_xg` flag has no effect today.

**Why:** Understat asks for respectful use and the data is optional for the
spec's acceptance criteria; the proxy keeps the xG-Poisson model honest and
runnable offline.

**Effect:** the `xgpoisson` model blends shot-based xG with goals
(`xgp.w = 0.5`), it does not use real xG.

## D004 - The walk-forward driver is not parallelised across leagues

**Spec:** section 10.1 asks for `joblib.Parallel(n_jobs=-1)` over the
league-specific models (M2, M3).

**What was done:** the driver walks the blocks sequentially and fits each model
per block. The full five-league tuning window (2017-08-01 to 2024-07-28, 267
blocks, 12,641 matches, nine models) completes in about 23 minutes, which is
inside the "under about 30 minutes" budget of the M3 acceptance criterion.

**Why:** the block loop is the unit of correctness (each block must only see
data before its start) and the models are already fast enough; parallelising
across leagues would have added a second ordering of the same data and a
class of race conditions in the OOF table for no measurable gain.

**Effect:** no change in the numbers, only in wall-clock time.

## D005 - Bootstrap resamples are seeded, not random per call

**Spec:** section 10.4 asks for 5,000 block-bootstrap resamples.

**What was done:** `block_bootstrap_ci` uses `numpy.random.default_rng(42)` by
default, so the same two models always produce the same interval.

**Why:** the M6 acceptance criterion requires that running the evaluation
twice returns identical numbers. A seed is not a weakening of the test: the
resampling is still 5,000 draws over whole 7-day blocks.

**Effect:** the confidence interval is reproducible across runs.

## D006 - The holdout window is block aligned

**Spec:** section 10.5 defines the tuning window as "all eval blocks up to the
start of the last `holdout_seasons` completed seasons" and the headline window
as "the last `holdout_seasons` completed seasons".

**What was done:** the headline window runs from the first match of the first
held-out season to the last match of the last held-out season. The tuning
window ends on the Sunday before the Monday that opens the first headline
block, and the evaluation window is clipped to its own start and end dates.

**Why:** the blocks are Monday-start and seven days long. Without the clipping
the first headline block would pull in a few days of the previous (tuned-on)
season, and the last tuning block would overlap the headline window. The
window split would then leak tuning data into the headline result.

**Effect:** the tuning and headline windows never overlap.
