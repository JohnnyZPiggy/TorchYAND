"""Validated mask-only active-set updates; no outer-loop orchestration."""
from dataclasses import dataclass
from enum import IntEnum
import torch
from ..geometry.projectors import _mask, _vector
from ..diagnostics import simplex_kkt_residual
from .feasibility import _inputs, _scalar
from .active_set_kernels import update_active_set_unchecked


class ActiveSetStatus(IntEnum):
    NO_CHANGE=0
    ACTIVATED=1
    RELEASED=2
    LAST_FREE_PRESERVED=3
    INVALID_FACE=4
    INVALID_REQUEST=5


@dataclass(frozen=True)
class ActiveSetResult:
    free_mask: torch.Tensor
    activated_mask: torch.Tensor
    released_mask: torch.Tensor
    changed: torch.Tensor
    valid: torch.Tensor
    status: torch.Tensor


def update_active_set(x, grad, free_mask, *, phase, activation_mask=None,
                      release_mask=None, tau=0.0, active_tol=None,
                      kkt_rtol=None, kkt_atol=0.0):
    """Update only masks; x/grad/input masks are never mutated.

    phase is activation or release, never both. Callers must gate activation
    requests on successful positive boundary steps; this function verifies
    requested coordinates against current free status and active_tol.
    Release rechecks frozen KKT validity/stationarity and releases only the
    largest requested violation, breaking exact ties by smallest index.
    Free coordinates at the bound are legitimate. No repair/snapping occurs.
    Invalid items retain their input mask and report no actual updates.
    All vectors share float32/64 dtype/device; only leading dimensions
    broadcast. Explicit tolerances are finite nonnegative (active_tol absolute).
    Import from constraints.active_set; the frozen package __init__ is unchanged.
    """
    if phase not in ('activation','release'):
        raise ValueError('phase must be activation or release')
    if (phase=='activation' and release_mask is not None) or (phase=='release' and activation_mask is not None):
        raise ValueError('one phase per call; opposite-phase request is not allowed')
    _vector(x,'x')
    _mask(free_mask,x)
    x,grad,mask,tau,batch=_inputs(x,grad,free_mask,tau)
    request=activation_mask if phase=='activation' else release_mask
    if request is not None:
        _mask(request,x)
        batch=torch.broadcast_shapes(batch,request.shape[:-1])
        x=x.expand(batch+(x.shape[-1],)); grad=grad.expand(x.shape)
        mask=mask.expand(x.shape); tau=tau.expand(batch); request=request.expand(x.shape)
    elif phase=='activation':
        request=torch.zeros_like(mask)
    else:
        # All active coordinates form the default pool; KKT filters them.
        request=~mask
    tol=_scalar(64*torch.finfo(x.dtype).eps if active_tol is None else active_tol,'active_tol',x,batch)
    kkt=simplex_kkt_residual(x,grad,free_mask=mask,tau=tau,rtol=kkt_rtol,atol=kkt_atol)
    return ActiveSetResult(*update_active_set_unchecked(x,grad,mask,request,tau,tol,kkt,phase=phase))
