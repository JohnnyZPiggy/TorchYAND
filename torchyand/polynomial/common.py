"""Shared input contracts, numerical guards, and result formatting."""

from dataclasses import dataclass
import math

import torch


@dataclass
class RootResult:
    """Sorted roots with algebraic multiplicity; all root arrays are [..., 3].

    Invalid roots and residuals are NaN. ``degree`` is [...] int64;
    -1 denotes the identically zero polynomial, 0 a nonzero constant.
    multiplicity_hint is optional; repeated roots are retained regardless.
    """

    roots: torch.Tensor
    valid: torch.Tensor
    multiplicity_hint: torch.Tensor | None
    degree: torch.Tensor
    residual: torch.Tensor


def validate_coefficients(*coefficients: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Validate finite float32/64 tensors, promote dtype, and broadcast.

    Device transfers are forbidden. Scalars must be zero-dimensional tensors.
    """
    if not coefficients:
        raise ValueError("At least one coefficient is required")
    if any(not isinstance(c, torch.Tensor) for c in coefficients):
        raise TypeError("Coefficients must be torch.Tensor objects")
    device = coefficients[0].device
    dtype = coefficients[0].dtype
    for c in coefficients:
        if c.dtype not in (torch.float32, torch.float64):
            raise TypeError("Only float32 and float64 are supported")
        if c.device != device:
            raise ValueError("All coefficients must be on the same device")
        if not torch.isfinite(c).all().item():
            raise ValueError("Coefficients must not contain NaN or Inf")
        dtype = torch.promote_types(dtype, c.dtype)
    return torch.broadcast_tensors(*(c.to(dtype=dtype) for c in coefficients))


def tolerance(value: float | None, default: float) -> float:
    """Resolve a finite, nonnegative absolute threshold."""
    if value is None:
        return default
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    return value


def normalize_coefficients(*coefficients: torch.Tensor) -> tuple[torch.Tensor, ...]:
    scale = torch.stack([c.abs() for c in coefficients], dim=-1).amax(-1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return tuple(c / safe_scale for c in coefficients)


def evaluate(coefficients: tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
    """Horner evaluation; coefficients are in descending degree order."""
    value = torch.zeros_like(x)
    for c in coefficients:
        value = value * x + c.unsqueeze(-1)
    return value


def normalized_residual(coefficients, x):
    numerator = evaluate(coefficients, x).abs()
    denominator = evaluate(tuple(c.abs() for c in coefficients), x.abs())
    return numerator / (denominator + torch.finfo(x.dtype).tiny)


def polish_roots(coefficients, roots, valid, steps=1):
    degree = len(coefficients) - 1
    derivative = tuple(c * (degree - i) for i, c in enumerate(coefficients[:-1]))
    for _ in range(steps):
        x = torch.where(valid, roots, torch.zeros_like(roots))
        dp = evaluate(derivative, x)
        scale = evaluate(tuple(c.abs() for c in derivative), x.abs())
        threshold = 64 * torch.finfo(x.dtype).eps * scale.clamp_min(torch.finfo(x.dtype).tiny)
        safe = valid & (dp.abs() > threshold)
        candidate = x - evaluate(coefficients, x) / torch.where(safe, dp, torch.ones_like(dp))
        accept = safe & torch.isfinite(candidate) & (
            normalized_residual(coefficients, candidate) <= normalized_residual(coefficients, x)
        )
        roots = torch.where(accept, candidate, roots)
    return roots


def make_result(roots, valid, degree, coefficients):
    valid = valid & torch.isfinite(roots)
    sortable = torch.where(valid, roots, torch.full_like(roots, float("inf")))
    sortable, indices = sortable.sort(dim=-1)
    valid = valid.gather(-1, indices)
    roots = torch.where(valid, sortable, torch.full_like(sortable, float("nan")))
    residual = normalized_residual(coefficients, torch.where(valid, roots, torch.zeros_like(roots)))
    residual = torch.where(valid, residual, torch.full_like(residual, float("nan")))
    return RootResult(roots, valid, None, degree, residual)
