"""Validated matrix-free orthogonal projectors, preserving device and autograd."""
from dataclasses import dataclass
import torch
from . import kernels

DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class GeometryState:
    """Built by build_geometry; do not mutate its tensors during use.

    True in free_mask means free. level_regular=False uses nu=0 and the face
    projector, not a regular level-set normal. Zero free coordinates define
    the zero projector; this does not certify a feasible simplex face.
    """
    grad_face: torch.Tensor
    grad_norm: torch.Tensor
    nu: torch.Tensor
    free_mask: torch.Tensor
    free_count: torch.Tensor
    level_regular: torch.Tensor
    grad_threshold: torch.Tensor


def _vector(v, name):
    if not isinstance(v, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if v.dtype not in DTYPES:
        raise TypeError(f"{name} must be float32 or float64")
    if v.ndim < 1 or v.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [..., n], n>=1")
    if not torch.isfinite(v).all().item():
        raise ValueError(f"{name} must be finite")


def _mask(mask, v):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("free_mask must be a bool tensor")
    if mask.device != v.device:
        raise ValueError("free_mask must share the vector device")
    if mask.ndim < 1 or mask.shape[-1] != v.shape[-1]:
        raise ValueError("free_mask must have the same last dimension n")


def _broadcast(v, mask):
    _mask(mask, v)
    batch = torch.broadcast_shapes(v.shape[:-1], mask.shape[:-1])
    shape = batch+(v.shape[-1],)
    return v.expand(shape), mask.expand(shape)


def _finite(*values):
    for value in values:
        if not torch.isfinite(value).all().item():
            raise FloatingPointError("Geometry arithmetic produced NaN/Inf")


def project_simplex_tangent(v):
    """Project finite float32/64 [...,n] onto sum-zero directions."""
    _vector(v, "v")
    result = kernels.simplex_tangent(v)
    _finite(result)
    return result


def project_face_tangent(v, free_mask):
    """True means free. Broadcast leading dimensions only; m<=1 returns zero."""
    _vector(v, "v")
    v, free_mask = _broadcast(v, free_mask)
    result = kernels.face_tangent(v, free_mask)
    _finite(result)
    return result


def build_geometry(gradient, free_mask=None, *, grad_tol=None):
    """Construct face gradient/normal without a dense projector.

    Default threshold is 128*eps*sqrt(max(m,1))*max(1,||masked gradient||_2)
    per portfolio. Explicit grad_tol is an absolute nonnegative finite norm
    tolerance (scalar or tensor broadcastable to the existing batch shape).
    Tensor tolerances must share gradient dtype/device. No precision promotion.
    grad_norm<=threshold sets nu=0; level projection then equals face projection.
    """
    _vector(gradient, "gradient")
    if free_mask is None:
        free_mask = torch.ones_like(gradient, dtype=torch.bool)
    gradient, free_mask = _broadcast(gradient, free_mask)
    if grad_tol is not None:
        if not isinstance(grad_tol, torch.Tensor):
            grad_tol = torch.as_tensor(grad_tol, dtype=gradient.dtype, device=gradient.device)
        if grad_tol.dtype != gradient.dtype:
            raise TypeError("grad_tol must match gradient dtype")
        if grad_tol.device != gradient.device:
            raise ValueError("grad_tol must match gradient device")
        if not (torch.isfinite(grad_tol).all() & (grad_tol >= 0).all()).item():
            raise ValueError("grad_tol must be finite and nonnegative")
        grad_tol = torch.broadcast_to(grad_tol, gradient.shape[:-1])
    gf, norm, nu, count, regular, threshold = kernels.geometry(gradient, free_mask, grad_tol)
    _finite(gf, norm, nu, threshold)
    return GeometryState(gf, norm, nu, free_mask, count, regular, threshold)


def _validate_geometry(geom, v):
    if not isinstance(geom, GeometryState):
        raise TypeError("geom must be a GeometryState from build_geometry")
    _vector(geom.grad_face, "geom.grad_face")
    _vector(geom.nu, "geom.nu")
    shape = geom.grad_face.shape
    if shape[-1] != v.shape[-1] or geom.nu.shape != shape:
        raise ValueError("Geometry vectors must share shape [...,n] and v's n")
    for value in (geom.grad_face, geom.nu, geom.grad_norm, geom.grad_threshold):
        if not isinstance(value, torch.Tensor) or value.dtype != v.dtype:
            raise TypeError("Geometry floating tensors must match v dtype")
        if value.device != v.device:
            raise ValueError("Geometry must match v device")
        if not torch.isfinite(value).all().item():
            raise ValueError("Geometry must be finite")
    _mask(geom.free_mask, v)
    if geom.free_mask.shape != shape:
        raise ValueError("Geometry mask must match geometry vector shape")
    for value, dtype in ((geom.free_count, torch.int64), (geom.level_regular, torch.bool)):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype:
            raise TypeError("Invalid geometry metadata dtype")
        if value.device != v.device:
            raise ValueError("Geometry metadata must match v device")
    if any(value.shape != shape[:-1] for value in
           (geom.grad_norm, geom.grad_threshold, geom.free_count, geom.level_regular)):
        raise ValueError("Geometry metadata must match geometry batch shape")


def project_level_tangent(v, geom):
    """Broadcast v and geometry leading dimensions; nu=0 returns P_F(v).

    Geometry must come from build_geometry and remain unmodified. Validation
    checks tensor metadata/finite values, not reconstructed mathematical state.
    """
    _vector(v, "v")
    _validate_geometry(geom, v)
    v, mask = _broadcast(v, geom.free_mask)
    result = kernels.level_tangent(v, mask, geom.nu)
    _finite(result)
    return result
