"""Unnormalized projected-probe correction with per-item PCG diagnostics."""
from dataclasses import replace
import torch
from ..geometry.kernels import level_tangent
from ..linalg import batched_pcg
from ..operators import ProjectedHessianOperator
from ..oracle import MVSKCache
from .exact import LogDetResult, LogDetStatus, _prepare, _expanded_geometry
from . import kernels


def stochastic_logdet_correction(oracle, x, geom, *, q=4, cache=None, damping=0.0,
                                 pcg_options=None, generator=None, probes=None,
                                 probe="rademacher"):
    """Average P_S T3[y_l,xi_l], xi=P_S(probes), K_lambda y=xi.

    Explicit probes have [...,q,n], same dtype/device, with leading dimensions
    broadcast against x/geometry. They are ambient probes, never normalized.
    Generated probes are iid Rademacher (default) or Gaussian. Generator must
    match the sampling device. All q true-residual solves must converge.
    Nonregular items fail; regular zero-dimensional items succeed with zero.
    No retry/fallback and no reuse of correction when damping changes.
    pcg_options accepts rtol, atol, max_iter, mode and M_inv only.
    """
    if isinstance(q, bool) or not isinstance(q, int) or q < 1:
        raise ValueError("q must be an integer >=1")
    if probe not in ("rademacher", "gaussian"):
        raise ValueError("probe must be rademacher or gaussian")
    extra = ()
    if probes is not None:
        if not isinstance(probes, torch.Tensor) or probes.dtype != x.dtype:
            raise TypeError("probes must be a tensor matching x dtype")
        if probes.device != x.device:
            raise ValueError("probes must match x device")
        if probes.ndim < 2 or probes.shape[-2:] != (q, x.shape[-1]):
            raise ValueError("probes must have shape [...,q,n]")
        if not torch.isfinite(probes).all().item():
            raise ValueError("probes must be finite")
        extra = probes.shape[:-2]
    options = {} if pcg_options is None else dict(pcg_options)
    if set(options)-{"rtol", "atol", "max_iter", "mode", "M_inv"}:
        raise ValueError("Unsupported pcg_options key")
    batch, xb, zb, geom, damping, owner = _prepare(oracle, x, geom, cache, damping, extra)
    geom = _expanded_geometry(geom, batch)
    shape = batch+(q, oracle.n)
    if probes is None:
        if probe == "gaussian":
            probes = torch.randn(shape, dtype=x.dtype, device=x.device, generator=generator)
        else:
            probes = torch.randint(0, 2, shape, device=x.device, generator=generator).to(x.dtype)*2-1
    else:
        probes = probes.expand(shape)
    nonempty = (geom.free_count-1-geom.level_regular.to(torch.int64)) > 0
    usable = geom.level_regular & nonempty
    xi = kernels.project_probes(probes, geom.free_mask, geom.nu, usable)
    probes_finite = torch.isfinite(xi).all(dim=-1)
    xi = torch.where(probes_finite.unsqueeze(-1), xi, 0)
    # Add a probe axis to cache and geometry, preserving z instead of recomputing
    # it q times. A new cache references the corresponding lifted x object.
    lifted_x = xb.unsqueeze(-2)
    lifted_cache = MVSKCache(lifted_x, zb.unsqueeze(-2), owner,
                             lifted_x.shape, lifted_x.dtype, lifted_x.device)
    lifted_geom = replace(geom,
        grad_face=geom.grad_face.unsqueeze(-2), nu=geom.nu.unsqueeze(-2),
        free_mask=geom.free_mask.unsqueeze(-2), grad_norm=geom.grad_norm.unsqueeze(-1),
        free_count=geom.free_count.unsqueeze(-1), level_regular=geom.level_regular.unsqueeze(-1),
        grad_threshold=geom.grad_threshold.unsqueeze(-1))
    operator = ProjectedHessianOperator(oracle, lifted_x, lifted_geom,
                                        cache=lifted_cache, damping=damping.unsqueeze(-1))
    projector = lambda v: level_tangent(v, lifted_geom.free_mask, lifted_geom.nu)
    # Align the RHS with the exact numerical projector used by PCG. This
    # preserves unnormalized projected probes while removing path roundoff.
    xi = projector(xi)
    pcg_result = batched_pcg(operator.matvec_unchecked, xi, projector=projector, **options)
    solves_ok = pcg_result.converged.all(dim=-1)
    y = torch.where(pcg_result.converged.unsqueeze(-1), pcg_result.x, 0)
    correction, finite = kernels.stochastic_contraction(
        zb, y, xi, oracle.A, oracle.coefficients, geom.free_mask, geom.nu)
    zero_dim = geom.level_regular & ~nonempty
    status = torch.full(batch, int(LogDetStatus.SUCCESS), dtype=torch.int64, device=x.device)
    status = torch.where(usable & (~finite | ~probes_finite.all(dim=-1)), int(LogDetStatus.NONFINITE), status)
    status = torch.where(usable & ~solves_ok, int(LogDetStatus.LINEAR_SOLVE_FAILED), status)
    status = torch.where(~geom.level_regular, int(LogDetStatus.DEGENERATE), status)
    correction = torch.where(zero_dim.unsqueeze(-1), 0, correction)
    success = status == LogDetStatus.SUCCESS
    correction = torch.where(success.unsqueeze(-1), correction, float('nan'))
    return LogDetResult(correction, success, status, damping, probe_count=q, pcg_result=pcg_result)
