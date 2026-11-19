from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


@dataclass(frozen=True)
class Config:
    leagues: Mapping[str, str]
    first_season: str
    data: Mapping[str, Any]
    features: Mapping[str, Any]
    elo: Mapping[str, Any]
    dc: Mapping[str, Any]
    xgp: Mapping[str, Any]
    gbm: Mapping[str, Any]
    stack: Mapping[str, Any]
    calibrate: Mapping[str, Any]
    eval: Mapping[str, Any]
    sim: Mapping[str, Any]
    server: Mapping[str, Any]
    raw: Mapping[str, Any]
    config_hash: str
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "prorrhesis.db"

    @property
    def alias_overrides_path(self) -> Path:
        return self.data_dir / "alias_overrides.csv"

    @property
    def discovered_names_path(self) -> Path:
        return self.data_dir / "discovered_names.json"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.raw_dir, self.cache_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


def _config_hash(raw: Mapping[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> Config:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    root = path.parent
    return Config(
        leagues=_freeze(raw.get("leagues", {})),
        first_season=str(raw.get("first_season", "1415")),
        data=_freeze(raw.get("data", {})),
        features=_freeze(raw.get("features", {})),
        elo=_freeze(raw.get("elo", {})),
        dc=_freeze(raw.get("dc", {})),
        xgp=_freeze(raw.get("xgp", {})),
        gbm=_freeze(raw.get("gbm", {})),
        stack=_freeze(raw.get("stack", {})),
        calibrate=_freeze(raw.get("calibrate", {})),
        eval=_freeze(raw.get("eval", {})),
        sim=_freeze(raw.get("sim", {})),
        server=_freeze(raw.get("server", {})),
        raw=_freeze(raw),
        config_hash=_config_hash(raw),
        root=root,
    )


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return _load_cached(str(p))


def season_codes(first_season: str, current_season: str) -> list[str]:
    """Season folder codes from `first_season` (e.g. '1415') to `current_season` inclusive."""
    codes = []
    y0 = int(first_season[:2])
    y1 = int(current_season[:2])
    for y in range(y0, y1 + 1):
        codes.append(f"{y:02d}{(y + 1) % 100:02d}")
    return codes
