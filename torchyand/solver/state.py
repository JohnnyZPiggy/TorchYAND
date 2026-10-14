"""Explicit per-item iteration state, with no retained face geometry."""
from dataclasses import dataclass
from enum import IntEnum
import torch
from ..oracle import MVSKCache
from .options import YANDOptions


class SolverStatus(IntEnum):
    RUNNING=0
    CONVERGED_KKT=1
    MAX_ITER=2
    STALLED=3
    DIRECTION_FAILED=4
    LINE_SEARCH_FAILED=5
    INVALID_STATE=6


class SolverReason(IntEnum):
    NONE=0
    FACE_UPDATE_LIMIT=1
    ZERO_PROGRESS=2
    ACTIVE_SET_FAILURE=3
    DIRECTION_NUMERICAL_FAILURE=4
    LINE_SEARCH_NUMERICAL_FAILURE=5
    INVALID_FEASIBILITY=6
    NAN_DETECTED=7
    DRIVER_GUARD_EXCEEDED=8


class IterationEvent(IntEnum):
    NONE=0
    RELEASE=1
    DIRECTION=2


@dataclass(frozen=True)
class YANDState:
    """Do not mutate tensors/oracle/options during use. Cache binds exact x.

    Generator is a shared service, not a per-item state tensor. Shared streams
    guarantee reproducibility for identical shapes/order/configuration only.
    Counters named *_refreshes count outer logical work, not hidden kernel
    evaluations inside frozen direction/search. Unknown totals remain None.
    """
    x: torch.Tensor
    free_mask: torch.Tensor
    value: torch.Tensor
    grad: torch.Tensor
    cache: MVSKCache
    status: torch.Tensor
    reason: torch.Tensor
    active: torch.Tensor
    iterations: torch.Tensor
    accepted_steps: torch.Tensor
    face_updates: torch.Tensor
    fallback_count: torch.Tensor
    no_progress_count: torch.Tensor
    lambda_reg: torch.Tensor
    kkt_residual: torch.Tensor
    last_alpha: torch.Tensor
    last_step_norm: torch.Tensor
    last_objective_change: torch.Tensor
    face_revision: torch.Tensor
    x_revision: torch.Tensor
    direction_attempts: torch.Tensor
    probe_draws: torch.Tensor
    value_refreshes: torch.Tensor
    gradient_refreshes: torch.Tensor
    cache_rebuilds: torch.Tensor
    options: YANDOptions
    generator: torch.Generator
    owner: object
    total_hvp: None = None
    total_third: None = None
    total_pcg_iterations: None = None
    total_value_evaluations: None = None
    total_gradient_evaluations: None = None


@dataclass(frozen=True)
class IterationResult:
    state: YANDState
    event: torch.Tensor
    measurable_progress: torch.Tensor
    current_kkt: object | None = None
    final_kkt: object | None = None
    release_result: object | None = None
    direction_result: object | None = None
    line_search_result: object | None = None
    activation_result: object | None = None
