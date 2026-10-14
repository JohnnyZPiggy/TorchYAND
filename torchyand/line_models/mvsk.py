"""Validated exact quartic line models without line-search integration."""
import torch
from ..oracle import MVSKOracle
from .kernels import mvsk_quartic_coefficients_unchecked


def mvsk_quartic_coefficients(oracle, x, d, *, cache=None):
    """Return [...,5] ordered [A0,A1,A2,A3,A4] for f(x+alpha*d).

    Only leading dimensions broadcast; asset dimensions, dtype and device
    must match the oracle. No direction projection or feasibility check.
    Validate cache against the original x object before expanding any views.
    Cache/data/x must not be mutated during cache lifetime (oracle contract).
    No precision promotion, device transfer or autograd detachment occurs.
    Nonfinite inputs raise ValueError; nonfinite intermediate arithmetic or
    coefficients raise FloatingPointError, even if a zero weight could
    mathematically eliminate an overflowing power.
    """
    if not isinstance(oracle, MVSKOracle):
        raise TypeError('oracle must be MVSKOracle')
    oracle._validate_tensor(x, 'x')
    oracle._validate_tensor(d, 'd')
    oracle._validate_tensor(oracle.coefficients, 'coefficients', (4,))
    if not (torch.isfinite(oracle.A).all() & torch.isfinite(oracle.mu).all()).item():
        raise ValueError('oracle data must be finite')
    batch = torch.broadcast_shapes(x.shape[:-1], d.shape[:-1])
    cached = oracle._projection(x, cache)
    w = d @ oracle.A.T
    oracle._finite_output(w)
    coeff, finite = mvsk_quartic_coefficients_unchecked(
        x.expand(batch+(oracle.n,)), d.expand(batch+(oracle.n,)),
        cached.z.expand(batch+(oracle.T,)), w.expand(batch+(oracle.T,)),
        oracle.mu, oracle.coefficients)
    if not finite.all().item():
        raise FloatingPointError('MVSK quartic coefficient assembly produced NaN/Inf')
    return coeff
