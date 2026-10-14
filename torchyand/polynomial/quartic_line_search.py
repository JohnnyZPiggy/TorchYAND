"""Exact stationary-point/endpoint enumeration on finite closed intervals."""

from dataclasses import dataclass
from numbers import Real

import torch

from .common import RootResult, tolerance, validate_coefficients
from .cubic import _cubic_real_roots_unchecked
from .evaluate import quartic_shifted_horner


@dataclass
class QuarticMinResult:
    """Batch-shaped x/value/success and five fixed candidate slots.

    Slots are lower, upper, root0, root1, root2. Repeats remain; invalid
    candidate x/value are NaN. Failure sets x/value to NaN while preserving
    candidate diagnostics. root_result describes the derivative polynomial.
    """

    x: torch.Tensor
    value: torch.Tensor
    candidate_x: torch.Tensor
    candidate_value: torch.Tensor
    candidate_valid: torch.Tensor
    root_result: RootResult
    success: torch.Tensor


def _bound_tensor(bound, coeff):
    if isinstance(bound, torch.Tensor):
        return bound
    if not isinstance(bound, Real):
        raise TypeError("Bounds must be real Python scalars or torch.Tensor objects")
    return torch.tensor(bound, dtype=coeff.dtype, device=coeff.device)


def minimize_quartic_interval(
    coeff: torch.Tensor, lower: float | torch.Tensor, upper: float | torch.Tensor,
    *, interval_tol: float | None = None, value_tol: float | None = None,
) -> QuarticMinResult:
    """Minimize sum(A_i*x**i) over a finite closed interval, batch-first.

    coeff must have shape [...,5], ordered [A0,A1,A2,A3,A4]. Bounds broadcast
    with its leading dimensions. Tensor inputs must share a device and be
    float32/64; Torch dtype promotion applies. Python bounds inherit coeff's
    dtype/device. Nonfinite inputs or lower>upper raise ValueError.

    Comparison excludes A0; returned values include A0. Default absolute
    interval tolerance is 64*eps*max(1,|lower|,|upper|); default absolute tie
    tolerance is 128*eps*max(1,max|valid shifted values|). Explicit tolerances
    are finite nonnegative absolute scalars. Among candidates within tolerance
    of the smallest shifted value choose the smallest x. Boundary-near roots
    are clamped into the interval. lower==upper and constant polynomials are
    legal. Cubic default degree preservation is used without approximation.

    success requires valid candidates, finite shifted values for EVERY valid
    candidate and a finite selected full value. Numerical derivative overflow
    also fails that batch item. On failure x/value are NaN. No argmin gradient
    guarantee is made. 'Exact' refers to candidate enumeration, not precision.
    """
    if not isinstance(coeff, torch.Tensor):
        raise TypeError("coeff must be a torch.Tensor")
    if coeff.ndim < 1 or coeff.shape[-1] != 5:
        raise ValueError("coeff must have last dimension 5")
    lower, upper = _bound_tensor(lower, coeff), _bound_tensor(upper, coeff)
    *parts, lower, upper = validate_coefficients(*coeff.unbind(-1), lower, upper)
    coeff = torch.stack(parts, -1)
    if (lower > upper).any().item():
        raise ValueError("Every interval must satisfy lower <= upper")
    if interval_tol is not None:
        interval_tol = tolerance(interval_tol, 0.)
    if value_tol is not None:
        value_tol = tolerance(value_tol, 0.)
    return _minimize_quartic_interval_unchecked(
        coeff, lower, upper, interval_tol=interval_tol, value_tol=value_tol)


def _minimize_quartic_interval_unchecked(
    coeff: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
    *, interval_tol: float | None = None, value_tol: float | None = None,
) -> QuarticMinResult:
    """Tensor kernel for finite, broadcast, same-dtype/device validated inputs.

    Assumes lower<=upper and validated nonnegative scalar tolerances. Only
    Python branches on static settings are used; there are no tensor scalar
    reads or public root calls. Numerical formulas and failure semantics match
    the safe public API. Intended for strict fullgraph compilation.
    """
    eps = torch.finfo(coeff.dtype).eps
    if interval_tol is None:
        boundary_tol = 64 * eps * torch.maximum(lower.abs(), upper.abs()).clamp_min(1)
    else:
        boundary_tol = torch.full_like(lower, interval_tol)

    a0, a1, a2, a3, a4 = coeff.unbind(-1)
    derivative = torch.stack((4*a4, 3*a3, 2*a2, a1), -1)
    derivative_finite = torch.isfinite(derivative).all(-1)
    # Derived overflow is a numerical failure, not an invalid-input exception.
    # Benign zero rows let other batch items continue through the frozen solver.
    safe_derivative = torch.where(derivative_finite.unsqueeze(-1), derivative, torch.zeros_like(derivative))
    roots = _cubic_real_roots_unchecked(*safe_derivative.unbind(-1))
    rx = torch.where(roots.valid, roots.roots, lower.unsqueeze(-1))
    inside = ((rx >= lower.unsqueeze(-1)) | ((lower.unsqueeze(-1)-rx) <= boundary_tol.unsqueeze(-1)))
    inside &= ((rx <= upper.unsqueeze(-1)) | ((rx-upper.unsqueeze(-1)) <= boundary_tol.unsqueeze(-1)))
    root_valid = roots.valid & inside & derivative_finite.unsqueeze(-1)
    clamped = torch.maximum(lower.unsqueeze(-1), torch.minimum(upper.unsqueeze(-1), rx))
    candidate_x = torch.cat((lower.unsqueeze(-1), upper.unsqueeze(-1), clamped), -1)
    valid = torch.cat((torch.ones_like(candidate_x[..., :2], dtype=torch.bool), root_valid), -1)
    safe_x = torch.where(valid, candidate_x, torch.zeros_like(candidate_x))
    shifted = quartic_shifted_horner(coeff, safe_x)
    full = a0.unsqueeze(-1) + shifted
    finite_all = (torch.isfinite(shifted) | ~valid).all(-1)
    comparable = valid & torch.isfinite(shifted)
    masked = torch.where(comparable, shifted, torch.full_like(shifted, float("inf")))
    best = masked.amin(-1, keepdim=True)
    if value_tol is None:
        scale = torch.where(comparable, shifted.abs(), torch.zeros_like(shifted)).amax(-1, keepdim=True).clamp_min(1)
        tie_tol = 128 * eps * scale
    else:
        tie_tol = value_tol
    tied = comparable & ((shifted-best) <= tie_tol)
    ordered = torch.where(tied, candidate_x, torch.full_like(candidate_x, float("inf")))
    selected = ordered.argmin(-1, keepdim=True)
    x = candidate_x.gather(-1, selected).squeeze(-1)
    value = full.gather(-1, selected).squeeze(-1)
    success = valid.any(-1) & finite_all & torch.isfinite(value) & derivative_finite
    nan = torch.full_like(value, float("nan"))
    return QuarticMinResult(
        torch.where(success, x, nan), torch.where(success, value, nan),
        torch.where(valid, candidate_x, torch.full_like(candidate_x, float("nan"))),
        torch.where(valid, full, torch.full_like(full, float("nan"))),
        valid, roots, success,
    )
