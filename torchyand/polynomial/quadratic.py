"""Stable batched quadratic real roots with linear fallback."""

import torch
from .common import (
    validate_coefficients, normalize_coefficients, tolerance, make_result, polish_roots,
)
from .linear import _linear_roots_unchecked


def quadratic_real_roots(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, *,
    degree_tol: float | None = 0.0,
    discriminant_tol: float | None = None,
    polish: bool = True,
):
    """Solve ax²+bx+c=0; preserve multiplicity and return sorted [...,3] slots.

    degree_tol is applied after coefficient normalization; None also means 0.
    degree_tol=0 preserves algebraic degree unless the leading coefficient is
    exactly zero. degree_tol>0 intentionally treats sufficiently small
    normalized leading coefficients as numerical degree degeneration.
    discriminant_tol, if
    supplied, replaces the relative factor 128*eps multiplying the local
    discriminant scale b²+|4ac| (clamped to dtype tiny).
    """
    a, b, c = validate_coefficients(a, b, c)
    eps = torch.finfo(a.dtype).eps
    dtol = tolerance(degree_tol, 0.0)
    relative_dtol = tolerance(discriminant_tol, 128 * eps)
    return _quadratic_real_roots_unchecked(
        a, b, c, degree_tol=dtol, discriminant_tol=relative_dtol, polish=polish)


def _quadratic_real_roots_unchecked(a, b, c, *, degree_tol=0.0,
                                   discriminant_tol=None, polish=True):
    """Tensor kernel; scalar settings must be resolved/validated by caller."""
    a, b, c = normalize_coefficients(a, b, c)
    dtol = degree_tol
    relative_dtol = 128 * torch.finfo(a.dtype).eps if discriminant_tol is None else discriminant_tol
    quadratic = a.abs() > dtol
    # Guard inactive divisions before masked selection.
    safe_a = torch.where(quadratic, a, torch.ones_like(a))
    delta = b * b - 4 * a * c
    scale = (b * b + (4 * a * c).abs()).clamp_min(torch.finfo(a.dtype).tiny)
    threshold = relative_dtol * scale
    double = delta.abs() <= threshold
    real = delta >= -threshold
    sqrt_delta = torch.sqrt(delta.clamp_min(0))
    sign_b = torch.where(b >= 0, torch.ones_like(b), -torch.ones_like(b))
    q = -0.5 * (b + sign_b * sqrt_delta)
    safe_q = torch.where(q != 0, q, torch.ones_like(q))
    midpoint = -b / (2 * safe_a)
    r1 = torch.where(double | (q == 0), midpoint, q / safe_a)
    r2 = torch.where(double | (q == 0), midpoint, c / safe_q)
    roots = torch.stack((r1, r2, torch.zeros_like(r1)), -1)
    mask = quadratic & real
    valid = torch.stack((mask, mask, torch.zeros_like(mask)), -1)
    if polish:
        roots = polish_roots((a, b, c), roots, valid)
    # Preserve the original normalization when deciding the lower degree.
    linear = b.abs() > dtol
    low_degree = torch.where(linear, 1, torch.where(c.abs() > dtol, 0, -1))
    lower = _linear_roots_unchecked(b, c, degree_tol=0.0)
    lower_valid = lower.valid & linear.unsqueeze(-1)
    roots = torch.where(quadratic.unsqueeze(-1), roots, lower.roots)
    valid = torch.where(quadratic.unsqueeze(-1), valid, lower_valid)
    degree = torch.where(quadratic, 2, low_degree).to(torch.int64)
    return make_result(roots, valid, degree, (a, b, c))
