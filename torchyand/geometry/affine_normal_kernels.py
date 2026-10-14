"""Pure tensor affine-normal algebra; no retry control or validation here."""
import torch
from .kernels import face_tangent, level_tangent


def assemble_rhs(h, correction, grad_norm, free_mask, nu):
    # n is the full ambient asset dimension, never the number of free assets.
    rhs = h-(grad_norm/h.shape[-1]).unsqueeze(-1)*correction
    return level_tangent(rhs, free_mask, nu)


def assemble_direction(u, free_mask, nu):
    u = level_tangent(u, free_mask, nu)
    return u, face_tangent(u-nu, free_mask)


def assess_direction(direction, gradient, grad_norm, free_mask, descent_factor):
    eps = torch.finfo(direction.dtype).eps
    length = torch.linalg.vector_norm(direction, dim=-1)
    dot = (gradient*direction).sum(-1)
    projected = face_tangent(direction, free_mask)
    drift = torch.linalg.vector_norm(direction-projected, dim=-1)
    floor = 128*eps*free_mask.sum(-1).clamp_min(1).to(direction.dtype).sqrt()*length.clamp_min(1)
    finite = (torch.isfinite(direction).all(-1) & torch.isfinite(length)
              & torch.isfinite(dot) & torch.isfinite(drift))
    active_zero = (torch.where(free_mask, 0, direction) == 0).all(-1)
    valid = finite & active_zero & (drift <= floor) & (dot < -descent_factor*eps*grad_norm*length)
    return dot, valid


def retry_seed(z, A, c):
    """128*eps*max(1, ||diag(H)||_inf), per batch item.

    This is the minimum configured roundoff-scale seed, not finfo.tiny.
    It creates a positive first retry from lambda=0 without a dense Hessian.
    Nonfinite curvature scales cannot rescue construction; use the absolute
    floor for the retry schedule and let solve/finite guards report failure.
    """
    _, c2, c3, c4 = c.unbind()
    weight = 2*c2-6*c3*z+12*c4*z.square()
    diagonal = (weight@A.square())/A.shape[0]
    scale = diagonal.abs().amax(-1).clamp_min(1)
    scale = torch.where(torch.isfinite(scale), scale, 1)
    return 128*torch.finfo(z.dtype).eps*scale


def next_damping(damping, seed, growth):
    return torch.where(damping == 0, seed, damping*growth)
