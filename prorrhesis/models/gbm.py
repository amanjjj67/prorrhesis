"""LightGBM multiclass model (spec section 7.5).

Trees stay small on purpose: with about 380 matches per league-season a
bigger model overfits within a season. Early stopping uses the most recent
10% of the training window by time, never a random split, and the model is
refitted on a cadence (every 4 weeks by default) and reused in between.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.schema import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
from .base import assert_as_of

try:  # LightGBM is optional: the app must run without it.
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover - depends on the environment
    lgb = None
    HAVE_LIGHTGBM = False

GBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    learning_rate=0.03,
    num_leaves=8,
    min_data_in_leaf=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    max_depth=4,
    verbose=-1,
    seed=42,
)

LABELS = {"H": 0, "D": 1, "A": 2}


class GbmModel:
    """M5: LightGBM over the feature set of spec section 8.2."""

    name = "gbm"
    needs = {"results", "features"}
    needs_features = True

    def __init__(
        self,
        refit_weeks: int = 4,
        early_stop_frac: float = 0.10,
        params: dict | None = None,
        use_market: bool = False,
        feature_columns: tuple[str, ...] | None = None,
        num_boost_round: int = 500,
        stopping_rounds: int = 30,
        name: str | None = None,
    ):
        self.refit_weeks = int(refit_weeks)
        self.early_stop_frac = float(early_stop_frac)
        self.params = dict(params or GBM_PARAMS)
        self.use_market = bool(use_market)
        self.num_boost_round = int(num_boost_round)
        self.stopping_rounds = int(stopping_rounds)
        if name:
            self.name = name
        base = list(feature_columns or FEATURE_COLUMNS)
        if not self.use_market:
            base = [c for c in base if c not in MARKET_FEATURE_COLUMNS]
        self.feature_names: list[str] = base

        self.feature_table: pd.DataFrame | None = None
        self.booster = None
        self.last_refit: str | None = None
        self.importances: dict[str, float] = {}
        self.best_iteration: int | None = None
        self.refits: int = 0

    # -- features ----------------------------------------------------------
    def attach_features(self, table: pd.DataFrame | None) -> None:
        """Receive the feature table built by the walk-forward driver."""
        if table is None or len(table) == 0:
            self.feature_table = None
            return
        self.feature_table = table.set_index("match_id", drop=False)

    def _rows_for(self, ids: list[int]) -> pd.DataFrame:
        if self.feature_table is None:
            return pd.DataFrame(columns=self.feature_names)
        return self.feature_table.reindex(ids)

    # -- fitting -----------------------------------------------------------
    def fit(self, history: pd.DataFrame, as_of: str) -> None:
        assert_as_of(history, as_of)
        if not HAVE_LIGHTGBM:
            raise RuntimeError("lightgbm is not installed")
        if self.feature_table is None:
            raise ValueError("features are not attached")
        if history is None or len(history) == 0:
            return

        if self.booster is not None and self.last_refit is not None:
            elapsed = (pd.Timestamp(as_of) - pd.Timestamp(self.last_refit)).days
            if elapsed < 7 * self.refit_weeks:
                return

        ids = [int(x) for x in history["id"].tolist()]
        rows = self._rows_for(ids)
        labels = history["res"].astype(str).map(LABELS)
        rows = rows.assign(_y=labels.to_numpy())
        rows = rows[rows["_y"].notna()].sort_values("date")
        if len(rows) < 200:
            return

        n_valid = max(1, int(round(len(rows) * self.early_stop_frac)))
        valid = rows.iloc[-n_valid:]
        train = rows.iloc[:-n_valid]
        if len(train) < 100:
            return
        if train["_y"].nunique() < 2:
            return

        X_train = train[self.feature_names]
        y_train = train["_y"].astype(int).to_numpy()
        X_valid = valid[self.feature_names]
        y_valid = valid["_y"].astype(int).to_numpy()

        params = dict(self.params)
        params["feature_fraction_seed"] = int(params.get("seed", 42))
        params["bagging_seed"] = int(params.get("seed", 42))
        params["data_random_seed"] = int(params.get("seed", 42))

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)
        booster = lgb.train(
            params, dtrain, num_boost_round=self.num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.stopping_rounds, verbose=False)],
        )
        self.booster = booster
        self.last_refit = str(as_of)
        self.refits += 1
        self.best_iteration = int(booster.best_iteration or self.num_boost_round)
        gains = booster.feature_importance(importance_type="gain")
        total = float(np.sum(gains)) or 1.0
        self.importances = {name: float(g) / total
                             for name, g in zip(self.feature_names, gains)}

    # -- prediction --------------------------------------------------------
    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        n = len(fixtures)
        P = np.full((n, 3), np.nan)
        status = np.empty(n, dtype=object)
        if self.booster is None:
            status[:] = "abstain"
            out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
            out["status"] = status
            return out

        ids = [int(x) for x in fixtures["id"].tolist()] if "id" in fixtures.columns else []
        if not ids:
            status[:] = "abstain"
            out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
            out["status"] = status
            return out

        rows = self._rows_for(ids)
        X = rows[self.feature_names]
        preds = np.asarray(self.booster.predict(X), dtype=float)
        if preds.ndim == 1:
            preds = np.column_stack([1 - preds, preds, np.zeros(len(preds))])

        for i in range(n):
            if pd.isna(rows.iloc[i]["match_id"]):
                P[i] = (1 / 3, 1 / 3, 1 / 3)
                status[i] = "abstain"
                continue
            row = preds[i]
            total = float(row.sum())
            if not np.isfinite(total) or total <= 0:
                P[i] = (1 / 3, 1 / 3, 1 / 3)
                status[i] = "abstain"
                continue
            P[i] = (row[0] / total, row[1] / total, row[2] / total)
            status[i] = "ok"

        out = pd.DataFrame({"pH": P[:, 0], "pD": P[:, 1], "pA": P[:, 2]}, index=fixtures.index)
        out["status"] = status
        return out

    # -- explanations ------------------------------------------------------
    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP-style per-feature contributions for the rows of `X`."""
        if self.booster is None:
            return np.zeros((len(X), len(self.feature_names) + 1))
        contrib = self.booster.predict(X[self.feature_names], pred_contrib=True)
        return np.asarray(contrib, dtype=float)

    def top_features(self, n: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n]
