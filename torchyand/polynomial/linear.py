"""Batched linear roots, including constant and zero polynomials."""

import torch
from .common import validate_coefficients, normalize_coefficients, tolerance, make_result


def linear_roots(a: torch.Tensor, b: torch.Tensor, *, degree_tol: float | None = 0.0):
    """Solve ax+b=0, returning three slots on the input device and promoted dtype.

    degree_tol is applied after coefficient normalization; None also means 0.
    degree_tol=0 preserves algebraic degree unless the leading coefficient is
    exactly zero. degree_tol>0 intentionally treats sufficiently small
    normalized leading coefficients as numerical degree degeneration.
    """
    a, b = validate_coefficients(a, b)
    tol = tolerance(degree_tol, 0.0)
    return _linear_roots_unchecked(a, b, degree_tol=tol)


def _linear_roots_unchecked(a, b, *, degree_tol=0.0):
    """Tensor kernel; coefficients are validated and broadcast by the caller."""
    a, b = normalize_coefficients(a, b)
    tol = degree_tol
    linear = a.abs() > tol
    degree = torch.where(linear, 1, torch.where(b.abs() > tol, 0, -1)).to(torch.int64)
    root = -b / torch.where(linear, a, torch.ones_like(a))
    roots = torch.stack((root, torch.zeros_like(root), torch.zeros_like(root)), -1)
    valid = torch.stack((linear, torch.zeros_like(linear), torch.zeros_like(linear)), -1)
    return make_result(roots, valid, degree, (a, b))
