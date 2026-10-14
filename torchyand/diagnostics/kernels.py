"""Pure tensor KKT diagnostics, using raw objective gradients."""
import torch
from ..constraints.kernels import face_validity


def kkt(x, grad, tau, mask, rtol, atol, feasibility_tol=None):
    count = mask.sum(-1)
    gamma = torch.where(mask, grad, 0).sum(-1)/count.clamp_min(1)
    delta = grad-gamma.unsqueeze(-1)
    face = torch.where(mask, delta.abs(), 0).amax(-1)
    active = torch.where(mask, 0, (-delta).clamp_min(0)).amax(-1)
    residual = torch.maximum(face, active)
    threshold = atol+rtol*grad.abs().amax(-1).clamp_min(1)
    # Reuse the existing simplex feasibility policy, but not its stricter
    # face consistency: a released/free coordinate may remain at the bound.
    feasible, _ = face_validity(x, tau, mask, feasibility_tol)
    bound_tol = 64*torch.finfo(x.dtype).eps if feasibility_tol is None else feasibility_tol.unsqueeze(-1)
    active_consistent = torch.where(mask, True, (x-tau.unsqueeze(-1)).abs() <= bound_tol).all(-1)
    valid = feasible & active_consistent & (count > 0)
    finite = torch.isfinite(gamma) & torch.isfinite(residual) & torch.isfinite(threshold)
    valid = valid & finite
    release = ~mask & (delta < -threshold.unsqueeze(-1)) & (count > 0).unsqueeze(-1)
    gamma = torch.where(count > 0, gamma, float('nan'))
    face = torch.where(count > 0, face, float('nan'))
    active = torch.where(count > 0, active, float('nan'))
    residual = torch.where(count > 0, residual, float('nan'))
    return face, active, residual, gamma, threshold, valid & (face <= threshold), valid & (residual <= threshold), release, count, feasible, valid
