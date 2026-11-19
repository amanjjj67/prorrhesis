"""Model factory: builds the model list used by the backtest and live prediction."""

from __future__ import annotations

from ..config import Config
from .baseline import FreqModel, HomeModel
from .dixoncoles import DixonColesModel
from .elo import DEFAULT_HOME_ADV, EloModel
from .gbm import GbmModel
from .market import MarketCloseModel, MarketOpenModel
from .xgpoisson import XgPoissonModel

ALL_MODELS = ["freq", "home", "elo", "dc", "xgpoisson", "market_close", "market_open",
              "gbm", "gbm_market_aug"]


def build_models(cfg: Config, names: list[str] | None = None, tuning: dict | None = None,
                 dc_tuning: dict | None = None) -> list:
    wanted = list(names) if names else ["freq", "elo", "market_close"]
    tuning = tuning or {}
    dc_tuning = dc_tuning or {}
    elo_K = float(tuning.get("K", cfg.elo.get("K", 20)))
    elo_H = tuning.get("H") or dict(cfg.elo.get("H", DEFAULT_HOME_ADV))
    elo_rho = float(tuning.get("rho", cfg.elo.get("rho_season", 0.25)))
    elo_penalty = float(cfg.elo.get("promoted_penalty", 50))
    dc_cfg = cfg.dc or {}

    models = []
    for name in wanted:
        if name == "freq":
            models.append(FreqModel(window_seasons=3))
        elif name == "home":
            models.append(HomeModel())
        elif name == "elo":
            models.append(EloModel(K=elo_K, home_adv=elo_H, rho_season=elo_rho,
                                   promoted_penalty=elo_penalty))
        elif name == "dc":
            models.append(DixonColesModel(
                window_seasons=int(dc_tuning.get("window_seasons", dc_cfg.get("window_seasons", 3))),
                half_life_days=float(dc_tuning.get("half_life_days", dc_cfg.get("half_life_days", 365.0))),
                gamma=float(dc_tuning.get("gamma", dc_cfg.get("gamma", 0.5))),
            ))
        elif name == "market_close":
            models.append(MarketCloseModel())
        elif name == "market_open":
            models.append(MarketOpenModel())
        elif name == "xgpoisson":
            xgp_cfg = cfg.xgp or {}
            models.append(XgPoissonModel(
                w=float(xgp_cfg.get("w", 0.5)),
                window_seasons=int(dc_tuning.get("window_seasons", dc_cfg.get("window_seasons", 3))),
                half_life_days=float(dc_tuning.get("half_life_days", dc_cfg.get("half_life_days", 365.0))),
                gamma=float(dc_tuning.get("gamma", dc_cfg.get("gamma", 0.5))),
            ))
        elif name in ("gbm", "gbm_market_aug"):
            gbm_cfg = cfg.gbm or {}
            models.append(GbmModel(
                refit_weeks=int(gbm_cfg.get("refit_weeks", 4)),
                early_stop_frac=float(gbm_cfg.get("early_stop_frac", 0.10)),
                use_market=(name == "gbm_market_aug"),
                name=name,
            ))
        else:
            raise ValueError(f"unknown model: {name}")
    return models
