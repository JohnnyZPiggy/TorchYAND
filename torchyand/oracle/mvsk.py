"""Safe public MVSK oracle with explicit, minimal projection cache."""
from dataclasses import dataclass, field
import torch

from ..data.mvsk_data import MVSKData, SUPPORTED_DTYPES
from . import kernels


@dataclass(frozen=True)
class MVSKCache:
    """Only x reference, z and metadata are retained; no power/direction cache.

    During its lifetime do not modify x, z, oracle data or coefficients in
    place. In-place mutation/version tracking is intentionally not implemented.
    Another x tensor object, even with identical values, requires a new cache.
    """
    x: torch.Tensor
    z: torch.Tensor
    owner: object = field(repr=False)
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class MVSKOracle:
    """Analytic MVSK on arbitrary portfolios (no simplex requirement).

    Shared coefficients must be finite, nonnegative shape [4]. Tensor inputs
    must match data dtype/device exactly; Python coefficient sequences inherit
    them. Data/coefficients must not be modified during use. Public methods
    validate finite inputs/outputs; arithmetic overflow raises FloatingPointError.
    All tensor/autograd chains are retained. Caches may retain an autograd graph
    and should be recreated after updates/backward as appropriate.
    """

    def __init__(self, data: MVSKData, coefficients):
        if not isinstance(data, MVSKData):
            raise TypeError("data must be MVSKData")
        if not isinstance(data.A, torch.Tensor) or not isinstance(data.mu, torch.Tensor):
            raise TypeError("A and mu must be tensors")
        if data.A.dtype not in SUPPORTED_DTYPES or data.mu.dtype != data.A.dtype:
            raise TypeError("A/mu must have matching float32/64 dtype")
        if data.mu.device != data.A.device:
            raise ValueError("A/mu must share device")
        if data.T < 1 or data.n < 1 or data.A.shape != (data.T, data.n) or data.mu.shape != (data.n,):
            raise ValueError("Inconsistent MVSKData shapes/T/n")
        if not (torch.isfinite(data.A).all() & torch.isfinite(data.mu).all()).item():
            raise ValueError("MVSKData must be finite")
        if not isinstance(coefficients, torch.Tensor):
            coefficients = torch.as_tensor(coefficients, dtype=data.A.dtype, device=data.A.device)
        self.data = data
        self.A, self.mu, self.T, self.n = data.A, data.mu, data.T, data.n
        self.coefficients = coefficients
        self._owner = object()
        self._validate_tensor(coefficients, "coefficients", (4,))
        if (coefficients < 0).any().item():
            raise ValueError("coefficients must be nonnegative")

    def _validate_tensor(self, tensor, name, exact_shape=None):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype not in SUPPORTED_DTYPES or tensor.dtype != self.A.dtype:
            raise TypeError(f"{name} must match data float32/64 dtype")
        if tensor.device != self.A.device:
            raise ValueError(f"{name} must match data device")
        if exact_shape is not None:
            if tensor.shape != exact_shape:
                raise ValueError(f"{name} must have shape {exact_shape}")
        elif tensor.ndim < 1 or tensor.shape[-1] != self.n:
            raise ValueError(f"{name} must have last dimension n={self.n}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _finite_output(*tensors):
        for tensor in tensors:
            if not torch.isfinite(tensor).all().item():
                raise FloatingPointError("MVSK evaluation produced NaN/Inf")

    def _projection(self, x, cache):
        self._validate_tensor(x, "x")
        if cache is None:
            return self.cache(x)
        if not isinstance(cache, MVSKCache) or cache.owner is not self._owner:
            raise ValueError("cache must belong to this oracle")
        if cache.x is not x:
            raise ValueError("cache must correspond to the same x tensor object")
        if cache.shape != x.shape or cache.dtype != x.dtype or cache.device != x.device:
            raise ValueError("cache metadata does not match x")
        if cache.z.shape != x.shape[:-1]+(self.T,) or cache.z.dtype != x.dtype or cache.z.device != x.device:
            raise ValueError("cache projection metadata does not match x/data")
        self._finite_output(cache.z)
        return cache

    def cache(self, x):
        self._validate_tensor(x, "x")
        z = kernels.projection(x, self.A)
        self._finite_output(z)
        return MVSKCache(x, z, self._owner, x.shape, x.dtype, x.device)

    def value(self, x, cache=None):
        cached = self._projection(x, cache)
        result = kernels.value(x, cached.z, self.mu, self.coefficients)
        self._finite_output(result)
        return result

    def gradient(self, x, cache=None):
        cached = self._projection(x, cache)
        result = kernels.gradient(cached.z, self.A, self.mu, self.coefficients)
        self._finite_output(result)
        return result

    def value_grad(self, x, cache=None):
        cached = self._projection(x, cache)
        value, gradient = kernels.value_grad(x, cached.z, self.A, self.mu, self.coefficients)
        self._finite_output(value, gradient)
        return value, gradient, cached

    def hvp(self, x, v, cache=None):
        self._validate_tensor(v, "v")
        cached = self._projection(x, cache)
        torch.broadcast_shapes(x.shape[:-1], v.shape[:-1])
        result = kernels.hvp(cached.z, v, self.A, self.coefficients)
        self._finite_output(result)
        return result

    def third_action(self, x, u, v, cache=None):
        self._validate_tensor(u, "u")
        self._validate_tensor(v, "v")
        cached = self._projection(x, cache)
        torch.broadcast_shapes(x.shape[:-1], u.shape[:-1], v.shape[:-1])
        result = kernels.third_action(cached.z, u, v, self.A, self.coefficients)
        self._finite_output(result)
        return result
