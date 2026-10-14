"""Masking/commit helpers only; reuse frozen objective/gradient kernels."""
import torch
from ..oracle import kernels as oracle_kernels
from .state import SolverStatus as S, SolverReason as R


def choose(mask,new,old):
    return torch.where(mask.reshape(mask.shape+(1,)*(new.ndim-mask.ndim)),new,old)


def evaluate(x,A,mu,c):
    z=x@A.T
    value=oracle_kernels.value(x,z,mu,c)
    grad=oracle_kernels.gradient(z,A,mu,c)
    finite=torch.isfinite(z).all(-1) & torch.isfinite(value) & torch.isfinite(grad).all(-1)
    return z,value,grad,finite


def refresh_gradient(x,A,mu,c):
    z=x@A.T
    grad=oracle_kernels.gradient(z,A,mu,c)
    finite=torch.isfinite(z).all(-1) & torch.isfinite(grad).all(-1)
    return z,grad,finite


def anchor_rows(x,eligible):
    # Fixed-shape gather of a known eligible point; no ragged slicing.
    flat=x.reshape(-1,x.shape[-1]); idx=eligible.reshape(-1).to(torch.int64).argmax().reshape(1,1)
    anchor=flat.gather(0,idx.expand(1,x.shape[-1])).reshape((1,)*(x.ndim-1)+(x.shape[-1],))
    return choose(eligible,x,anchor.expand_as(x))


def finish(status,reason,eligible,kkt,iterations,face_updates,no_progress,options):
    next_status=torch.full_like(status,int(S.RUNNING)); next_reason=torch.zeros_like(reason)
    next_status=torch.where(iterations>=options.max_iter,int(S.MAX_ITER),next_status)
    stalled=no_progress>=options.stall_patience
    next_status=torch.where(stalled,int(S.STALLED),next_status)
    next_reason=torch.where(stalled,int(R.ZERO_PROGRESS),next_reason)
    limit=face_updates>=options.max_face_updates
    next_status=torch.where(limit,int(S.STALLED),next_status)
    next_reason=torch.where(limit,int(R.FACE_UPDATE_LIMIT),next_reason)
    next_status=torch.where(~kkt.valid,int(S.INVALID_STATE),next_status)
    next_reason=torch.where(~kkt.valid,int(R.INVALID_FEASIBILITY),next_reason)
    converged=kkt.valid & kkt.kkt_stationary
    next_status=torch.where(converged,int(S.CONVERGED_KKT),next_status)
    next_reason=torch.where(converged,0,next_reason)
    return choose(eligible,next_status,status),choose(eligible,next_reason,reason)
