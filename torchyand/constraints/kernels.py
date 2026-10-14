"""Fixed-shape simplex feasibility and step-cap tensor algebra."""
import torch
from ..geometry.kernels import face_tangent


def face_validity(x, tau, mask, feasibility_tol=None):
    eps = torch.finfo(x.dtype).eps
    bound_tol = torch.full_like(tau, 64*eps) if feasibility_tol is None else feasibility_tol
    sum_tol = 64*eps*x.abs().sum(-1).clamp_min(1) if feasibility_tol is None else feasibility_tol
    slack = x-tau.unsqueeze(-1)
    feasible = (slack >= -bound_tol.unsqueeze(-1)).all(-1) & ((x.sum(-1)-1).abs() <= sum_tol)
    consistent = torch.where(mask, slack > bound_tol.unsqueeze(-1), slack.abs() <= bound_tol.unsqueeze(-1)).all(-1)
    return feasible, feasible & consistent & mask.any(-1)


def step_cap(x, d, tau, mask, direction_tol=None, hit_tol=None):
    q = face_tangent(d, mask)
    negative = q < 0
    denominator = torch.where(negative, -q, 1)
    slack = x-tau.unsqueeze(-1)
    ratio = torch.where(negative, slack/denominator, float('inf'))
    alpha = ratio.amin(-1)
    zero = (q == 0).all(-1)
    eps = torch.finfo(x.dtype).eps
    threshold = 64*eps*q.abs().amax(-1).clamp_min(1) if direction_tol is None else direction_tol
    reliable = (q < -threshold.unsqueeze(-1)).any(-1)
    finite_ratio = torch.where(negative, torch.isfinite(ratio), True).all(-1)
    finite = torch.isfinite(q).all(-1) & finite_ratio & (torch.isfinite(alpha) | zero)
    # Avoid inf*0 even for an exactly zero direction.
    safe_alpha = torch.where(torch.isfinite(alpha), alpha, 0)
    movement = safe_alpha.unsqueeze(-1)*q
    endpoint = slack+movement
    scale = torch.maximum(slack.abs(), movement.abs()).clamp_min(1)
    hit_threshold = 64*eps*scale if hit_tol is None else hit_tol.unsqueeze(-1)
    hit = negative & torch.isfinite(alpha).unsqueeze(-1) & (endpoint.abs() <= hit_threshold)
    # Step cap permits a free coordinate exactly at the bound: cap=0 is useful.
    feasible, _ = face_validity(x, tau, mask)
    bound = 64*eps
    consistent = torch.where(mask, slack >= 0, slack.abs() <= bound).all(-1)
    valid_face = feasible & consistent & mask.any(-1)
    bad_input = ~valid_face | (alpha < 0)
    status = torch.where(bad_input, 1, torch.where(~finite, 3, torch.where(~zero & ~reliable, 2, 0)))
    return alpha, q, hit, hit.any(-1), zero, status == 0, status
