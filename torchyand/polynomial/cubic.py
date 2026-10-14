"""Fixed-shape analytic roots and companion reference roots."""

import math
import operator
from typing import Literal
import torch
from .common import (
    RootResult, validate_coefficients, normalize_coefficients, tolerance,
    polish_roots, make_result,
)
from .quadratic import quadratic_real_roots, _quadratic_real_roots_unchecked


def _cbrt_real(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * x.abs().pow(1.0 / 3.0)


def cubic_real_roots(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, *,
    method: Literal["analytic", "companion"] = "analytic",
    degree_tol: float | None = 0.0,
    discriminant_tol: float | None = None,
    polish_steps: int = 1,
    imag_tol: float | None = None,
) -> RootResult:
    """Solve ax³+bx²+cx+d=0 for broadcastable float32/64 tensors.

    Roots will be sorted, retain algebraic multiplicity, and occupy [...,3]
    slots with NaN padding and boolean valid masks. Degree -1 denotes infinite
    roots, degree 0 a nonzero constant; neither returns concrete roots.
    Inputs must be finite and share a device. Torch dtype promotion applies;
    computation stays in that dtype/device. No public promote parameter.
    degree_tol is applied after coefficient normalization; None also means 0.
    degree_tol=0 preserves algebraic degree unless the leading coefficient is
    exactly zero. degree_tol>0 intentionally treats sufficiently small
    normalized leading coefficients as numerical degree degeneration.
    Discriminant classification
    uses the design's scale-aware rule. Newton steps require a safe derivative
    and non-increasing normalized residual. Companion is reference/testing only.

    discriminant_tol overrides the relative multiplier (default 128*eps) of
    max((q/2)²+|p/3|³, tiny), matching the quadratic API convention.
    Companion uses eigvals and accepts |imag| <= imag_tol*max(1, |real|).
    Its default imag_tol is 4*eps**(1/3), allowing eigenvalue splitting at
    triple roots. This heuristic cannot distinguish sufficiently close complex
    roots from numerical splitting; override it for strict reference checks.
    imag_tol affects only companion. multiplicity_hint remains None.
    """
    if method not in ("analytic", "companion"):
        raise ValueError("method must be 'analytic' or 'companion'")
    try:
        polish_steps = operator.index(polish_steps)
    except TypeError as exc:
        raise ValueError("polish_steps must be a nonnegative integer") from exc
    if polish_steps < 0:
        raise ValueError("polish_steps must be a nonnegative integer")

    a, b, c, d = validate_coefficients(a, b, c, d)
    finfo = torch.finfo(a.dtype)
    degree_threshold = tolerance(degree_tol, 0.0)
    discriminant_multiplier = tolerance(discriminant_tol, 128 * finfo.eps)
    if method == "companion":
        coefficients = normalize_coefficients(a, b, c, d)
        cubic = coefficients[0].abs() > degree_threshold
        return _companion_roots(coefficients, cubic, degree_tol,
                                discriminant_tol, polish_steps, imag_tol)
    return _cubic_real_roots_unchecked(
        a, b, c, d, degree_tol=degree_threshold,
        discriminant_tol=discriminant_multiplier, polish_steps=polish_steps)


def _cubic_real_roots_unchecked(a, b, c, d, *, degree_tol=0.0,
                                discriminant_tol=None, polish_steps=1):
    """Analytic tensor kernel for validated, broadcast, same-dtype/device inputs.

    Scalar settings are validated by the public caller. No .item(), input
    validation, or tensor-dependent Python control flow occurs here or in the
    lower-degree kernels. Normalization and all numerical formulas are retained.
    """
    coefficients = normalize_coefficients(a, b, c, d)
    a, b, c, d = coefficients
    finfo = torch.finfo(a.dtype)
    degree_threshold = degree_tol
    discriminant_multiplier = 128 * finfo.eps if discriminant_tol is None else discriminant_tol
    cubic = a.abs() > degree_threshold
    # Inactive polynomials are replaced by x³ before monic conversion.
    safe_a = torch.where(cubic, a, torch.ones_like(a))
    A = torch.where(cubic, b, torch.zeros_like(b)) / safe_a
    B = torch.where(cubic, c, torch.zeros_like(c)) / safe_a
    C = torch.where(cubic, d, torch.zeros_like(d)) / safe_a
    p = B - A * A / 3
    q = 2 * A * A * A / 27 - A * B / 3 + C
    # At a shifted triple root both expressions cancel. Their rounding error
    # must be assessed against the terms being subtracted, not p and q alone:
    # otherwise uniform coefficient scaling can change the root count.
    p_roundoff = 8 * finfo.eps * (B.abs() + A * A / 3)
    q_roundoff = 8 * finfo.eps * (2 * A.abs().pow(3) / 27 + (A * B).abs() / 3 + C.abs())
    triple_cancellation = cubic & (p.abs() <= p_roundoff) & (q.abs() <= q_roundoff)
    p = torch.where(triple_cancellation, torch.zeros_like(p), p)
    q = torch.where(triple_cancellation, torch.zeros_like(q), q)
    half_q = q / 2
    third_p = p / 3
    delta = half_q * half_q + third_p * third_p * third_p
    scale_delta = half_q * half_q + third_p.abs().pow(3)
    delta_threshold = discriminant_multiplier * scale_delta.clamp_min(finfo.tiny)
    one_real = cubic & (delta > delta_threshold)
    repeated = cubic & (delta.abs() <= delta_threshold)
    three_real = cubic & (delta < -delta_threshold)

    # Each branch receives benign inputs outside its own mask. torch.where
    # selects values only; it does not suppress evaluation of either argument.
    one_q = torch.where(one_real, half_q, torch.zeros_like(q))
    one_p = torch.where(one_real, p, torch.zeros_like(p))
    sqrt_delta = torch.sqrt(torch.where(one_real, delta, torch.zeros_like(delta)).clamp_min(0))
    sign_q = torch.where(one_q >= 0, torch.ones_like(q), -torch.ones_like(q))
    u_arg = -one_q - sign_q * sqrt_delta
    u = _cbrt_real(u_arg)
    u_scale = _cbrt_real(one_q.abs() + sqrt_delta)
    safe_u = u.abs() > 64 * finfo.eps * u_scale.clamp_min(finfo.tiny)
    denominator = torch.where(safe_u, u, torch.ones_like(u))
    stabilized = u - (one_p / 3) / denominator
    fallback = _cbrt_real(-one_q + sqrt_delta) + _cbrt_real(-one_q - sqrt_delta)
    one_root = torch.where(safe_u, stabilized, fallback) - A / 3

    repeated_u = _cbrt_real(torch.where(repeated, -half_q, torch.zeros_like(q)))
    repeated_roots = torch.stack((2 * repeated_u, -repeated_u, -repeated_u), -1) - A.unsqueeze(-1) / 3

    negative_third_p = torch.where(three_real, -third_p, torch.ones_like(p)).clamp_min(0)
    radius = 2 * torch.sqrt(negative_third_p)
    cos_denominator = 2 * torch.sqrt(negative_third_p.pow(3))
    cos_denominator = torch.where(cos_denominator > 0, cos_denominator, torch.ones_like(p))
    trig_q = torch.where(three_real, q, torch.zeros_like(q))
    cos_arg = (-trig_q / cos_denominator).clamp(-1, 1)
    theta = torch.acos(cos_arg) / 3
    phases = torch.arange(3, dtype=a.dtype, device=a.device) * (2 * math.pi / 3)
    three_roots = radius.unsqueeze(-1) * torch.cos(theta.unsqueeze(-1) - phases) - A.unsqueeze(-1) / 3

    roots = torch.where(repeated.unsqueeze(-1), repeated_roots, three_roots)
    roots = torch.where(one_real.unsqueeze(-1), one_root.unsqueeze(-1).expand_as(roots), roots)
    first_slot = torch.arange(3, device=a.device) == 0
    valid = (repeated | three_real).unsqueeze(-1).expand_as(roots) | (one_real.unsqueeze(-1) & first_slot)
    roots = polish_roots(coefficients, roots, valid, steps=polish_steps)

    # Reuse the lower-degree implementation, with benign coefficients in
    # cubic slots. Its output is selected unchanged for degenerate items.
    lower = _quadratic_real_roots_unchecked(
        torch.where(cubic, torch.zeros_like(b), b),
        torch.where(cubic, torch.zeros_like(c), c),
        torch.where(cubic, torch.ones_like(d), d),
        degree_tol=degree_threshold, discriminant_tol=discriminant_multiplier,
        polish=polish_steps > 0,
    )
    roots = torch.where(cubic.unsqueeze(-1), roots, lower.roots)
    valid = torch.where(cubic.unsqueeze(-1), valid, lower.valid)
    degree = torch.where(cubic, torch.full_like(lower.degree, 3), lower.degree)
    return make_result(roots, valid, degree, coefficients)


def _companion_roots(coefficients, cubic, degree_tol, discriminant_tol,
                     polish_steps, imag_tol):
    """Reference only: eigensolve the true-cubic subset, never degenerate items."""
    a, b, c, d = coefficients
    relative_imag_tol = tolerance(imag_tol, 4 * torch.finfo(a.dtype).eps ** (1 / 3))
    lower = quadratic_real_roots(
        torch.where(cubic, torch.zeros_like(b), b),
        torch.where(cubic, torch.zeros_like(c), c),
        torch.where(cubic, torch.ones_like(d), d),
        degree_tol=degree_tol, discriminant_tol=discriminant_tol,
        polish=polish_steps > 0,
    )
    # Boolean indexing intentionally compacts only true cubics for reference.
    aa, bb, cc, dd = (v[cubic] for v in coefficients)
    roots = lower.roots.clone()
    valid = lower.valid.clone()
    if aa.numel():
        matrix = torch.zeros((aa.numel(), 3, 3), dtype=a.dtype, device=a.device)
        matrix[:, 1, 0] = 1
        matrix[:, 2, 1] = 1
        matrix[:, 0, 2] = -dd / aa
        matrix[:, 1, 2] = -cc / aa
        matrix[:, 2, 2] = -bb / aa
        eigenvalues = torch.linalg.eigvals(matrix)
        real = eigenvalues.real
        is_real = eigenvalues.imag.abs() <= relative_imag_tol * real.abs().clamp_min(1)
        real = polish_roots((aa, bb, cc, dd), real, is_real, steps=polish_steps)
        roots[cubic] = real
        valid[cubic] = is_real
    degree = torch.where(cubic, torch.full_like(lower.degree, 3), lower.degree)
    return make_result(roots, valid, degree, coefficients)
