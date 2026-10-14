"""Validated batched PCG with per-item convergence and breakdown diagnostics."""
from dataclasses import dataclass
import math
import numbers
import torch
from . import kernels


@dataclass(frozen=True)
class PCGResult:
    x: torch.Tensor
    residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    breakdown: torch.Tensor
    nonpositive_curvature: torch.Tensor
    max_iter_reached: torch.Tensor


def _input(v, name):
    if not isinstance(v, torch.Tensor) or v.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a float32/64 tensor")
    if v.ndim < 1 or v.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [...,n], n>=1")
    if not torch.isfinite(v).all().item():
        raise ValueError(f"{name} must be finite")


def _callback(fn, name):
    if not callable(fn):
        raise TypeError(f"{name} must be callable")
    def checked(v):
        result = fn(v)
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"{name} must return a tensor")
        if result.shape != v.shape:
            raise ValueError(f"{name} must preserve vector shape")
        if result.dtype != v.dtype:
            raise TypeError(f"{name} must preserve dtype")
        if result.device != v.device:
            raise ValueError(f"{name} must preserve device")
        # Nonfinite numerical values are handled per item by tensor kernels.
        return result
    return checked


def batched_pcg(matvec, b, *, x0=None, M_inv=None, projector=None,
                rtol=None, atol=0.0, max_iter=100, mode="adaptive"):
    """Solve independent SPD systems on CPU/CUDA without materializing matrices.

    b=[...,n]; x0 can broadcast to b but cannot change its n/batch. Callbacks
    preserve shape/dtype/device and must represent fixed linear maps, with no
    coupling between batch items. A projector must be an orthogonal projector;
    b must already belong to its range (checked within roundoff tolerance).
    Defaults: float32 rtol=1e-5, float64 rtol=1e-10, atol=0, x0=0, M_inv=I.
    Curvature guard is 128*eps*||p||*||Kp||. No damping retry or fallback.
    Failed items keep their last finite x. Recurrence convergence freezes an
    item; final convergence/residuals use b-matvec(x), without projection.
    A final residual gap may leave all three termination flags False: it is
    neither a breakdown nor necessarily an iteration-limit termination.
    Fixed mode always executes max_iter masked steps; adaptive may stop early.
    Callback exceptions propagate; nonfinite returned values mark breakdown.
    """
    _input(b, "b")
    rtol = (1e-5 if b.dtype == torch.float32 else 1e-10) if rtol is None else rtol
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if mode not in ("adaptive", "fixed"):
        raise ValueError("mode must be adaptive or fixed")
    matvec = _callback(matvec, "matvec")
    M_inv = None if M_inv is None else _callback(M_inv, "M_inv")
    projector = None if projector is None else _callback(projector, "projector")
    if x0 is None:
        x0 = torch.zeros_like(b)
    else:
        _input(x0, "x0")
        if x0.dtype != b.dtype:
            raise TypeError("x0 must match b dtype")
        if x0.device != b.device:
            raise ValueError("x0 must match b device")
        if x0.shape[-1] != b.shape[-1]:
            raise ValueError("x0 must match b's last dimension n")
        x0 = torch.broadcast_to(x0, b.shape)
    if projector is not None:
        pb = projector(b)
        if not torch.isfinite(pb).all().item():
            raise ValueError("projector(b) must be finite")
        error = kernels.norm(pb-b)
        scale = kernels.norm(b).clamp_min(1)
        floor = 128*torch.finfo(b.dtype).eps*b.shape[-1]**.5*scale
        if (error > floor).any().item():
            raise ValueError("b must already belong to the projector subspace")
    b_norm = kernels.norm(b)
    tolerance = atol+rtol*b_norm
    if mode == "fixed":
        state = kernels.fixed(matvec, b, x0, M_inv, projector, tolerance, max_iter)
    else:
        state = kernels.initialize(matvec, b, x0, M_inv, projector, tolerance)
        for _ in range(max_iter):
            if not (~state[4] & ~state[6]).any().item():
                break
            state = kernels.step(matvec, state, M_inv, projector, tolerance)
    return PCGResult(*kernels.finish(matvec, b, state, b_norm, tolerance, max_iter))
