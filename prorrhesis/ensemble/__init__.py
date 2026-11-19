"""Ensemble: stacking and calibration (spec section 9)."""

from .calibrate import (
    CalibrationOutcome,
    DirichletCalibrator,
    EnsembleCalibrator,
    IsotonicCalibrator,
    apply_temperature,
    fit_temperature,
)
from .stack import (
    EnsembleStacker,
    LogisticStacker,
    StackOutcome,
    available_mask,
    equal_pool,
    fit_log_pool,
    log_pool,
    stack_features,
    subset_key,
)

__all__ = [
    "CalibrationOutcome",
    "DirichletCalibrator",
    "EnsembleCalibrator",
    "EnsembleStacker",
    "IsotonicCalibrator",
    "LogisticStacker",
    "StackOutcome",
    "apply_temperature",
    "available_mask",
    "equal_pool",
    "fit_log_pool",
    "fit_temperature",
    "log_pool",
    "stack_features",
    "subset_key",
]
