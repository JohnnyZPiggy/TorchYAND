"""Small-n reduced Cholesky reference, never an ambient Hessian assembly."""
from dataclasses import dataclass, replace
from enum import IntEnum
import torch
from ..geometry import project_level_tangent
from ..oracle import MVSKCache
from ..operators import ProjectedHessianOperator
from ..linalg import PCGResult


class LogDetStatus(IntEnum):
    SUCCESS = 0
    DEGENERATE = 1
    NOT_SPD = 2
    LINEAR_SOLVE_FAILED = 3
    NONFINITE = 4


@dataclass(frozen=True)
class LogDetResult:
    """correction=[...,n], status/success/damping=[...].

    Failed correction slots are NaN; inspect success/status before use.
    Stochastic pcg_result fields have [...,q] batch shape and x=[...,q,n].
    Cholesky info is -1 when not attempted, 0 on success, >0 for SPD failure.
    No retry, silent fallback, or cached correction across damping changes.
    """
    correction: torch.Tensor
    success: torch.Tensor
    status: torch.Tensor
    damping: torch.Tensor
    probe_count: int | None = None
    pcg_result: PCGResult | None = None
    cholesky_info: torch.Tensor | None = None


def _prepare(oracle, x, geom, cache, damping, extra_batch=()):
    operator = ProjectedHessianOperator(oracle, x, geom, cache=cache, damping=damping)
    batch = torch.broadcast_shapes(x.shape[:-1], geom.grad_face.shape[:-1], extra_batch)
    # Geometry and damping are held fixed in the local third-order contraction.
    fixed_geom = replace(geom, nu=geom.nu.detach())
    damping = torch.broadcast_to(operator.damping.detach(), batch)
    return (batch, x.expand(batch+(oracle.n,)),
            operator.cache.z.expand(batch+(oracle.T,)), fixed_geom,
            damping, operator.cache.owner)


def _expanded_geometry(geom, batch):
    return replace(geom,
        grad_face=geom.grad_face.expand(batch+(geom.grad_face.shape[-1],)),
        nu=geom.nu.expand(batch+(geom.nu.shape[-1],)),
        free_mask=geom.free_mask.expand(batch+(geom.free_mask.shape[-1],)),
        grad_norm=geom.grad_norm.expand(batch), free_count=geom.free_count.expand(batch),
        level_regular=geom.level_regular.expand(batch), grad_threshold=geom.grad_threshold.expand(batch))


def _basis(geom, dtype):
    # Exact-only ragged basis construction. Helmert spans the free sum-zero
    # face; SVD removes its one normal coordinate, without a dense projector.
    indices = torch.nonzero(geom.free_mask, as_tuple=False).flatten()
    m, n = indices.numel(), geom.free_mask.numel()
    if m <= 2:
        return torch.empty(n, 0, dtype=dtype, device=indices.device)
    j = torch.arange(1, m, dtype=dtype, device=indices.device).unsqueeze(0)
    i = torch.arange(m, device=indices.device).unsqueeze(-1)
    denominator = (j*(j+1)).sqrt()
    helmert = torch.where(i < j, 1/denominator, torch.where(i == j, -j/denominator, 0))
    U = torch.zeros(n, m-1, dtype=dtype, device=indices.device).index_copy(0, indices, helmert)
    normal = (U.T@geom.nu).unsqueeze(0)
    _, _, Vh = torch.linalg.svd(normal, full_matrices=True)
    return (U@Vh[1:].T).detach()


def _validate_basis(W, geom, n):
    k = W.shape[-1]
    expected = max(int(geom.free_count.item())-2, 0)
    if k != expected:
        raise ValueError("basis_W must span the complete regular level tangent space")
    tol = 256*torch.finfo(W.dtype).eps*max(n, 1)
    identity = torch.eye(k, dtype=W.dtype, device=W.device)
    if not torch.allclose(W.T@W, identity, rtol=tol, atol=tol):
        raise ValueError("basis_W columns must be orthonormal")
    if not torch.allclose(project_level_tangent(W.T, geom), W.T, rtol=tol, atol=tol):
        raise ValueError("basis_W columns must lie in level tangent space")


def exact_logdet_correction(oracle, x, geom, *, basis_W=None, cache=None, damping=0.0):
    """Sum P_S T3[W solve(K_red,e_j), W e_j], followed by P_S.

    basis_W=[...,n,k] can be supplied; otherwise a small-n face/level basis is
    constructed internally. W/P/damping are fixed, with no differentiation of
    their dependence on x. Reduced Cholesky verifies SPD; failures are per item.
    Nonregular geometry fails, while regular k=0 succeeds with correction=0.
    Supports float32/64 CPU/CUDA and arbitrary broadcast leading dimensions.
    """
    extra = ()
    if basis_W is not None:
        if not isinstance(basis_W, torch.Tensor):
            raise TypeError("basis_W must be a tensor")
        if basis_W.dtype != x.dtype:
            raise TypeError("basis_W must match x dtype")
        if basis_W.device != x.device:
            raise ValueError("basis_W must match x device")
        if basis_W.ndim < 2 or basis_W.shape[-2] != x.shape[-1]:
            raise ValueError("basis_W must have shape [...,n,k]")
        if not torch.isfinite(basis_W).all().item():
            raise ValueError("basis_W must be finite")
        extra = basis_W.shape[:-2]
    batch, xb, zb, geom, damping, owner = _prepare(oracle, x, geom, cache, damping, extra)
    geom = _expanded_geometry(geom, batch)
    flat = {field: getattr(geom, field).reshape((-1, oracle.n) if field in
            ("grad_face", "nu", "free_mask") else (-1,)) for field in geom.__dataclass_fields__}
    xx, zz, dd = xb.reshape(-1, oracle.n), zb.reshape(-1, oracle.T), damping.reshape(-1)
    Ws = None if basis_W is None else basis_W.detach().expand(batch+basis_W.shape[-2:]).reshape(xx.shape[0], *basis_W.shape[-2:])
    corrections, statuses, infos = [], [], []
    for item in range(xx.shape[0]):
        gi = replace(geom, **{name: tensor[item] for name, tensor in flat.items()})
        zero = torch.zeros(oracle.n, dtype=x.dtype, device=x.device)
        status, info, a = LogDetStatus.DEGENERATE, -1, torch.full_like(zero, float('nan'))
        if gi.level_regular.item():
            W = _basis(gi, x.dtype) if Ws is None else Ws[item]
            _validate_basis(W, gi, oracle.n)
            k = W.shape[-1]
            if k == 0:
                status, a = LogDetStatus.SUCCESS, zero
            else:
                xi = xx[item]
                ci = MVSKCache(xi, zz[item], owner, xi.shape, xi.dtype, xi.device)
                try:
                    HW = oracle.hvp(xi, W.T, cache=ci).T
                    identity = torch.eye(k, dtype=x.dtype, device=x.device)
                    reduced = W.T@HW+dd[item]*identity
                    # Remove roundoff asymmetry in the mathematically symmetric matrix.
                    reduced = .5*reduced+.5*reduced.T
                    if not torch.isfinite(reduced).all().item():
                        status = LogDetStatus.NONFINITE
                    else:
                        L, info_tensor = torch.linalg.cholesky_ex(reduced)
                        info = int(info_tensor.item())
                        if info != 0:
                            status = LogDetStatus.NOT_SPD
                        else:
                            Y = torch.cholesky_solve(identity, L)
                            if not torch.isfinite(Y).all().item():
                                status = LogDetStatus.NONFINITE
                            else:
                                raw = oracle.third_action(xi, (W@Y).T, W.T, cache=ci)
                                projected = project_level_tangent(raw, gi)
                                a = project_level_tangent(projected.sum(dim=-2), gi)
                                status = LogDetStatus.SUCCESS
                except FloatingPointError:
                    status = LogDetStatus.NONFINITE
        if status != LogDetStatus.SUCCESS:
            a = torch.full_like(zero, float('nan'))
        corrections.append(a)
        statuses.append(int(status))
        infos.append(info)
    status = torch.tensor(statuses, dtype=torch.int64, device=x.device).reshape(batch)
    correction = torch.stack(corrections).reshape(batch+(oracle.n,)) if corrections else torch.empty_like(xb)
    return LogDetResult(correction, status == LogDetStatus.SUCCESS, status, damping,
                        cholesky_info=torch.tensor(infos, dtype=torch.int64, device=x.device).reshape(batch))
