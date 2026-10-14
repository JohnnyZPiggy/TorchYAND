"""Validated simplex step caps; no active-set mutation."""
from dataclasses import dataclass
from enum import IntEnum
import torch
from ..geometry.projectors import _vector, _mask
from . import kernels


class StepCapStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT_FACE = 1
    NO_RELIABLE_NEGATIVE = 2
    NONFINITE_RATIO = 3


@dataclass(frozen=True)
class StepCapResult:
    alpha_max: torch.Tensor
    direction: torch.Tensor
    hit_mask: torch.Tensor
    has_boundary_hit: torch.Tensor
    zero_direction: torch.Tensor
    success: torch.Tensor
    status: torch.Tensor


def _scalar(value, name, x, batch):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if value.dtype != x.dtype:
        raise TypeError(f'{name} must match input dtype')
    if value.device != x.device:
        raise ValueError(f'{name} must match input device')
    if not (torch.isfinite(value).all() & (value >= 0).all()).item():
        raise ValueError(f'{name} must be finite and nonnegative')
    return torch.broadcast_to(value, batch)


def _inputs(x, vector, free_mask, tau):
    _vector(x, 'x')
    _vector(vector, 'vector')
    if vector.dtype != x.dtype:
        raise TypeError('vector must match x dtype')
    if vector.device != x.device or vector.shape[-1] != x.shape[-1]:
        raise ValueError('vector must match x device and asset dimension')
    if free_mask is not None:
        _mask(free_mask, x)
    batch = torch.broadcast_shapes(x.shape[:-1], vector.shape[:-1], () if free_mask is None else free_mask.shape[:-1])
    x, vector = x.expand(batch+(x.shape[-1],)), vector.expand(batch+(x.shape[-1],))
    tau = _scalar(tau, 'tau', x, batch)
    if free_mask is None:
        free_mask = x-tau.unsqueeze(-1) > 64*torch.finfo(x.dtype).eps
    else:
        free_mask = free_mask.expand(x.shape)
    return x, vector, free_mask, tau, batch


def simplex_step_cap(x, d, *, tau=0.0, free_mask=None, direction_tol=None, hit_tol=None):
    """Return cap for q=P_F(d), not the unprojected input direction.

    Every exact negative q limits the cap; direction_tol only diagnoses
    reliability. Explicit tolerances are absolute. Free coordinates at the
    bound are allowed (an outward direction has cap zero). Without a mask,
    infer support for convenience only; outer solvers must supply face state.
    No artificial cap is imposed. Infinite cap is successful only for q=0.
    """
    x, d, mask, tau, batch = _inputs(x, d, free_mask, tau)
    dt = None if direction_tol is None else _scalar(direction_tol, 'direction_tol', x, batch)
    ht = None if hit_tol is None else _scalar(hit_tol, 'hit_tol', x, batch)
    return StepCapResult(*kernels.step_cap(x, d, tau, mask, dt, ht))
