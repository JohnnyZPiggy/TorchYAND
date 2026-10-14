"""Safe matrix-free level-tangent Hessian operator."""
import torch
from ..oracle import MVSKOracle
from ..geometry.projectors import _validate_geometry
from . import kernels


class ProjectedHessianOperator:
    """Bind an oracle point/cache/geometry; do not mutate bound tensors.

    K=P_S H P_S + damping P_S. Positive damping does not itself guarantee
    positive definiteness. Degenerate geometry uses the face tangent space.
    matvec is validated; matvec_unchecked is a tensor-only callable for PCG
    or future compilation, assuming validated, broadcast-compatible inputs.
    """
    def __init__(self, oracle, x, geom, *, cache=None, damping=0.0):
        if not isinstance(oracle, MVSKOracle):
            raise TypeError("oracle must be MVSKOracle")
        oracle._validate_tensor(x, "x")
        _validate_geometry(geom, x)
        torch.broadcast_shapes(x.shape[:-1], geom.grad_face.shape[:-1])
        if not isinstance(damping, torch.Tensor):
            damping = torch.as_tensor(damping, dtype=x.dtype, device=x.device)
        if damping.dtype != x.dtype:
            raise TypeError("damping must match oracle dtype")
        if damping.device != x.device:
            raise ValueError("damping must match oracle device")
        if not (torch.isfinite(damping).all() & (damping >= 0).all()).item():
            raise ValueError("damping must be finite and nonnegative")
        self.oracle, self.x, self.geom = oracle, x, geom
        self.cache = oracle._projection(x, cache)
        self.damping = damping

    def matvec_unchecked(self, v):
        return kernels.projected_hessian(
            v, self.cache.z, self.oracle.A, self.oracle.coefficients,
            self.geom.free_mask, self.geom.nu, self.geom.free_count,
            self.geom.level_regular, self.damping)

    def matvec(self, v):
        self.oracle._validate_tensor(v, "v")
        self.oracle._projection(self.x, self.cache)
        _validate_geometry(self.geom, v)
        batch = torch.broadcast_shapes(self.x.shape[:-1], v.shape[:-1],
                                       self.geom.grad_face.shape[:-1])
        # Damping can broadcast to this batch, but cannot introduce new items.
        damping = torch.broadcast_to(self.damping, batch)
        if not (torch.isfinite(damping).all() & (damping >= 0).all()).item():
            raise ValueError("damping must remain finite and nonnegative")
        result = kernels.projected_hessian(
            v.expand(batch+(self.oracle.n,)), self.cache.z, self.oracle.A,
            self.oracle.coefficients, self.geom.free_mask, self.geom.nu,
            self.geom.free_count, self.geom.level_regular, damping)
        self.oracle._finite_output(result)
        return result


def apply_projected_hessian(oracle, x, v, geom, *, cache=None, damping=0.0):
    """Validated K(v), float32/64 CPU/CUDA; broadcast leading dimensions only."""
    return ProjectedHessianOperator(oracle, x, geom, cache=cache, damping=damping).matvec(v)
