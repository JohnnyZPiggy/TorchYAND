"""Safe exact feasible MVSK line search; no repair, backtracking or mutation."""
from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace
import torch
from ..oracle import MVSKOracle, MVSKCache
from ..constraints import simplex_step_cap, StepCapResult
from ..constraints.feasibility import _inputs, _scalar
from ..polynomial.quartic_line_search import QuarticMinResult
from . import kernels


class LineSearchStatus(IntEnum):
    SUCCESS=0
    ZERO_DIRECTION=1
    ZERO_CAP=2
    ZERO_STEP=3
    STEP_CAP_FAILED=4
    NONFINITE_CAP=5
    ASSEMBLY_FAILED=6
    QUARTIC_FAILED=7
    TRIAL_INFEASIBLE=8
    VERIFICATION_FAILED=9


@dataclass(frozen=True)
class LineSearchResult:
    alpha: torch.Tensor
    x_trial: torch.Tensor
    current_value: torch.Tensor
    trial_value: torch.Tensor
    predicted_value: torch.Tensor
    direction: torch.Tensor
    alpha_max: torch.Tensor
    hit_mask: torch.Tensor
    hits_boundary: torch.Tensor
    success: torch.Tensor
    status: torch.Tensor
    step_cap_result: StepCapResult
    quartic_result: QuarticMinResult | None


def exact_mvsk_line_search(oracle, x, d, *, free_mask=None, tau=0.0, cache=None,
                           direction_tol=None, hit_tol=None, interval_tol=None,
                           value_tol=None, acceptance_tol=None):
    """Search along the cap's projected direction, then verify the true value.

    All vectors/data share float32/64 dtype/device; only leading dimensions
    broadcast. Cache is validated against original x before expansion.
    Per-item numerical failure returns alpha=0 and x_trial=x, success=False;
    predicted_value and nested diagnostics describe the attempted computation.
    Returned trial_value always corresponds to returned x_trial (current on
    failure, possibly nonfinite if even the current objective overflowed).
    Zero statuses are successful diagnostics, never convergence certificates.
    No clamp, repair, precision promotion or artificial step cap is applied.
    In mixed batches quartic diagnostics contain benign dummy rows for items
    that did not search; their success must not override line-search status.
    """
    if not isinstance(oracle,MVSKOracle): raise TypeError('oracle must be MVSKOracle')
    oracle._validate_tensor(x,'x'); oracle._validate_tensor(d,'d')
    oracle._validate_tensor(oracle.coefficients,'coefficients',(4,))
    if not (torch.isfinite(oracle.A).all() & torch.isfinite(oracle.mu).all()).item():
        raise ValueError('oracle data must be finite')
    if cache is not None:
        if not isinstance(cache,MVSKCache) or cache.owner is not oracle._owner or cache.x is not x:
            raise ValueError('cache must belong to this oracle and original x object')
        if cache.shape!=x.shape or cache.dtype!=x.dtype or cache.device!=x.device:
            raise ValueError('cache metadata mismatch')
        if cache.z.shape!=x.shape[:-1]+(oracle.T,) or cache.z.dtype!=x.dtype or cache.z.device!=x.device:
            raise ValueError('cache projection metadata mismatch')
        if not torch.isfinite(cache.z).all().item(): raise ValueError('cache must be finite')
        z=cache.z
    else:
        z=x@oracle.A.T
    x,d,mask,tau,batch=_inputs(x,d,free_mask,tau)
    z=z.expand(batch+(oracle.T,))
    cap=simplex_step_cap(x,d,free_mask=mask,tau=tau,direction_tol=direction_tol,hit_tol=hit_tol)
    tolerances=[]
    for name,v in (('interval_tol',interval_tol),('value_tol',value_tol)):
        if v is not None:
            t=_scalar(v,name,x,())
            v=float(t.item())
        tolerances.append(v)
    acceptance=None if acceptance_tol is None else _scalar(acceptance_tol,'acceptance_tol',x,batch)
    # A fixed metadata container keeps frozen StepCapResult untouched.
    internal=SimpleNamespace(**vars(cap),_free_mask=mask)
    run=bool((cap.success & ~cap.zero_direction & torch.isfinite(cap.alpha_max) & (cap.alpha_max>0)).any().item())
    values=kernels.exact_mvsk_line_search_unchecked(x,z,internal,oracle.A,oracle.mu,oracle.coefficients,tau,
        interval_tol=tolerances[0],value_tol=tolerances[1],acceptance_tol=acceptance,run_quartic=run)
    return LineSearchResult(*values[:-1],cap,values[-1])
