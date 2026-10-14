"""Tensor PCG state transitions and fixed loop, no scalar tensor reads.

Callbacks must preserve [...,n], dtype and device. Inactive directions are
zero; all updates use masks, never dynamic batch slicing. Optional callback
branches and max_iter are Python configuration, not tensor-dependent choices.
"""
import torch


def finite(v):
    return torch.isfinite(v).all(dim=-1)


def norm(v):
    return torch.linalg.vector_norm(v, dim=-1)


def masked(v, mask):
    return torch.where(mask.unsqueeze(-1), v, 0)


def project(v, projector):
    return v if projector is None else projector(v)


def precondition(r, M_inv, projector):
    raw = r if M_inv is None else M_inv(r)
    z = project(raw, projector)
    return z, finite(raw) & finite(z)


def initialize(matvec, b, x0, M_inv, projector, tolerance):
    xp = project(x0, projector)
    bad = ~finite(xp)
    x = torch.where(bad.unsqueeze(-1), x0, xp)
    ax = matvec(x)
    raw = b-ax
    r = project(raw, projector)
    rn = norm(r)
    bad = bad | ~finite(ax) | ~finite(raw) | ~finite(r) | ~torch.isfinite(rn) | ~torch.isfinite(tolerance)
    converged = ~bad & (rn <= tolerance)
    active = ~bad & ~converged
    r = masked(r, ~bad)
    z, z_finite = precondition(masked(r, active), M_inv, projector)
    rho = (r*z).sum(-1)
    bad = bad | (active & (~z_finite | ~torch.isfinite(rho) | (rho <= 0)))
    active = ~bad & ~converged
    p = masked(z, active)
    rho = torch.where(active, rho, 1)
    iterations = torch.zeros_like(rho, dtype=torch.int64)
    curvature = torch.zeros_like(converged)
    return x, r, p, rho, converged, iterations, bad, curvature


def step(matvec, state, M_inv, projector, tolerance):
    x, r, p, rho, converged, iterations, bad, curvature = state
    active = ~converged & ~bad
    p = masked(p, active)
    ap = matvec(p)
    denominator = (p*ap).sum(-1)
    pn, an = norm(p), norm(ap)
    floor = 128*torch.finfo(x.dtype).eps*pn*an
    numerical = ~finite(ap) | ~torch.isfinite(denominator) | ~torch.isfinite(floor)
    unsafe = ~numerical & (denominator <= floor)
    curvature = curvature | (active & unsafe)
    bad = bad | (active & (numerical | unsafe))
    work = active & ~bad
    # Replace BOTH numerator and denominator before division, not after it.
    alpha = torch.where(work, rho, 0)/torch.where(work, denominator, 1)
    bad = bad | (work & ~torch.isfinite(alpha))
    work = work & ~bad
    alpha = torch.where(work, alpha, 0)
    x_raw = x+alpha.unsqueeze(-1)*masked(p, work)
    r_raw = r-alpha.unsqueeze(-1)*masked(ap, work)
    xp = project(x_raw, projector)
    rp = project(r_raw, projector)
    rn = norm(rp)
    bad = bad | (work & (~finite(x_raw) | ~finite(r_raw) | ~finite(xp) | ~finite(rp) | ~torch.isfinite(rn)))
    work = work & ~bad
    x = torch.where(work.unsqueeze(-1), xp, x)
    r = torch.where(work.unsqueeze(-1), rp, r)
    iterations = iterations+work.to(iterations.dtype)
    converged = converged | (work & (rn <= tolerance))
    active = ~converged & ~bad
    z, z_finite = precondition(masked(r, active), M_inv, projector)
    new_rho = (r*z).sum(-1)
    bad = bad | (active & (~z_finite | ~torch.isfinite(new_rho) | (new_rho <= 0)))
    work = active & ~bad
    beta = torch.where(work, new_rho, 0)/torch.where(work, rho, 1)
    bad = bad | (work & ~torch.isfinite(beta))
    work = work & ~bad
    beta = torch.where(work, beta, 0)
    p_raw = masked(z, work)+beta.unsqueeze(-1)*masked(p, work)
    new_p = project(p_raw, projector)
    bad = bad | (work & (~finite(p_raw) | ~finite(new_p)))
    active = ~converged & ~bad
    p = masked(new_p, active)
    rho = torch.where(active, new_rho, 1)
    return x, r, p, rho, converged, iterations, bad, curvature


def fixed(matvec, b, x0, M_inv, projector, tolerance, max_iter):
    state = initialize(matvec, b, x0, M_inv, projector, tolerance)
    for _ in range(max_iter):
        state = step(matvec, state, M_inv, projector, tolerance)
    return state


def finish(matvec, b, state, b_norm, tolerance, max_iter):
    x, _, _, _, _, iterations, bad, curvature = state
    ax = matvec(x)
    r_true = b-ax  # Intentionally NOT projected: report the actual equation.
    residual_norm = norm(r_true)
    bad = bad | ~finite(ax) | ~finite(r_true) | ~torch.isfinite(residual_norm)
    residual_norm = torch.where(torch.isfinite(residual_norm), residual_norm, float("inf"))
    relative = residual_norm/torch.where(b_norm > 0, b_norm, 1)
    relative = torch.where(b_norm > 0, relative,
                           torch.where(residual_norm == 0, 0, float("inf")))
    relative = torch.where(torch.isfinite(b_norm), relative, float("inf"))
    converged = ~bad & (residual_norm <= tolerance)
    reached = ~converged & ~bad & (iterations >= max_iter)
    return x, residual_norm, relative, converged, iterations, bad, curvature, reached
