"""Regularized level-tangent logdet corrections."""
from .exact import LogDetResult, LogDetStatus, exact_logdet_correction
from .stochastic import stochastic_logdet_correction

__all__ = ["LogDetResult", "LogDetStatus", "exact_logdet_correction",
           "stochastic_logdet_correction"]
