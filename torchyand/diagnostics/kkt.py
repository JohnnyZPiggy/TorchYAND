"""Simplex/face optimality diagnostics, not active-set updates."""
from dataclasses import dataclass
import torch
from ..constraints.feasibility import _inputs, _scalar
from . import kernels


@dataclass(frozen=True)
class KKTResult:
    face_residual: torch.Tensor
    active_violation: torch.Tensor
    kkt_residual: torch.Tensor
    simplex_multiplier: torch.Tensor
    stopping_threshold: torch.Tensor
    face_stationary: torch.Tensor
    kkt_stationary: torch.Tensor
    release_mask: torch.Tensor
    free_count: torch.Tensor
    feasible: torch.Tensor
    valid: torch.Tensor


def simplex_kkt_residual(x, grad, *, tau=0.0, free_mask=None, rtol=None, atol=0.0, feasibility_tol=None):
    """Use grad=gradient of the minimization objective at the current x.

    simplex_multiplier is gamma, the common-gradient convention. For
    L=f+lambda*(sum(x)-1)-eta^T*x, lambda=-gamma and eta_i=grad_i-gamma.
    Face stationarity alone is not full simplex convergence. m=0 is invalid
    and has NaN multiplier/residuals. All coordinates must satisfy
    x >= tau-bound_tol; active coordinates additionally satisfy
    abs(x-tau) <= bound_tol. Free coordinates may be exactly at the bound,
    including immediately after release. sum(x) must remain within sum_tol.
    Without free_mask infer support for convenience diagnostics only; outer
    active-set solvers must explicitly provide their face state.
    Explicit feasibility_tol is absolute for both sum and bound checks.
    """
    x, grad, mask, tau, batch = _inputs(x, grad, free_mask, tau)
    rt = _scalar((1e-5 if x.dtype == torch.float32 else 1e-10) if rtol is None else rtol, 'rtol', x, batch)
    at = _scalar(atol, 'atol', x, batch)
    ft = None if feasibility_tol is None else _scalar(feasibility_tol, 'feasibility_tol', x, batch)
    return KKTResult(*kernels.kkt(x, grad, tau, mask, rt, at, ft))
