"""Regularized affine-normal construction with per-item retries and fallback.

Frozen oracle/projector/operator/PCG/logdet modules are composed, not modified.
Import from torchyand.geometry.affine_normal (the frozen __init__ is unchanged).
"""
from dataclasses import dataclass, replace
import math
import numbers
import torch
from .projectors import build_geometry, _validate_geometry
from .kernels import level_tangent
from . import affine_normal_kernels as kernels
from ..oracle import MVSKOracle
from ..oracle.kernels import hvp as ordinary_hvp_unchecked
from ..operators import ProjectedHessianOperator
from ..linalg import batched_pcg, PCGResult
from ..logdet import exact_logdet_correction, stochastic_logdet_correction, LogDetResult, LogDetStatus


@dataclass(frozen=True)
class DirectionResult:
    direction: torch.Tensor
    normal_component: torch.Tensor
    tangent_correction: torch.Tensor
    rhs: torch.Tensor
    lambda_reg: torch.Tensor
    grad_norm: torch.Tensor
    descent_dot: torch.Tensor
    success: torch.Tensor
    degenerate: torch.Tensor
    fallback_used: torch.Tensor
    logdet_failed: torch.Tensor
    linear_failed: torch.Tensor
    descent_failed: torch.Tensor
    logdet_status: LogDetResult | None
    pcg_result: PCGResult | None
    regularization_retries: torch.Tensor


def _expand_geom(geom, batch):
    return replace(geom, **{name: value.expand(batch+((value.shape[-1],) if name in
                   ("grad_face", "nu", "free_mask") else ()))
                   for name, value in vars(geom).items()})


def _masked_geom(geom, mask):
    return replace(geom, grad_face=torch.where(mask.unsqueeze(-1), geom.grad_face, 0),
                   grad_norm=torch.where(mask, geom.grad_norm, 0),
                   nu=torch.where(mask.unsqueeze(-1), geom.nu, 0), level_regular=mask)


def _choose(mask, new, old):
    return torch.where(mask.reshape(mask.shape+(1,)*(new.ndim-mask.ndim)), new, old)


def _merge_pcg(old, new, mask):
    if new is None:
        return old
    if old is None:
        old = PCGResult(**{name: torch.zeros_like(value) for name, value in vars(new).items()})
    return PCGResult(**{name: _choose(mask, value, getattr(old, name)) for name, value in vars(new).items()})


def _merge_logdet(old, new, mask, zero_dim):
    if old is None:
        correction = torch.where(zero_dim.unsqueeze(-1), 0, torch.full_like(new.correction, float('nan')))
        old = LogDetResult(correction, zero_dim,
                           torch.where(zero_dim, int(LogDetStatus.SUCCESS), int(LogDetStatus.DEGENERATE)),
                           new.damping, probe_count=new.probe_count,
                           cholesky_info=None if new.cholesky_info is None else torch.full_like(new.cholesky_info, -1))
    return LogDetResult(_choose(mask, new.correction, old.correction),
                        _choose(mask, new.success, old.success), _choose(mask, new.status, old.status),
                        _choose(mask, new.damping, old.damping), probe_count=new.probe_count,
                        pcg_result=_merge_pcg(old.pcg_result, new.pcg_result, mask),
                        cholesky_info=None if new.cholesky_info is None else _choose(mask, new.cholesky_info, old.cholesky_info))


def _options(options, dtype):
    options = {} if options is None else dict(options)
    if set(options)-{"rtol", "atol", "max_iter", "mode", "M_inv"}:
        raise ValueError("Unsupported pcg_options key")
    rtol = options.get("rtol")
    rtol = (1e-5 if dtype == torch.float32 else 1e-10) if rtol is None else rtol
    atol = options.get("atol", 0.)
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    count = options.get("max_iter", 100)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if options.get("mode", "adaptive") not in ("adaptive", "fixed"):
        raise ValueError("PCG mode must be adaptive or fixed")
    if options.get("M_inv") is not None and not callable(options["M_inv"]):
        raise TypeError("M_inv must be callable")
    return options, rtol, atol


def affine_normal_direction(oracle, x, *, geom=None, free_mask=None, cache=None,
                            lambda0=1e-6, lambda_growth=10.0, max_regularization_retries=5,
                            logdet_mode="stochastic", q=4, basis_W=None, probes=None,
                            generator=None, pcg_options=None, descent_factor=64.0):
    """Solve K_lambda u=P_S(H nu)-(||g_F||/n)*a_lambda, return P_F(u-nu).

    lambda0 may be a tensor broadcast to the existing batch. A zero lambda's
    first retry uses 128*eps*max(1,||diag(H)||_inf); subsequent retries multiply
    by lambda_growth. Raw Rademacher probes are sampled ONCE per call and reused.
    Geometry/point/data must remain unmodified, and supplied geom must describe
    this point's gradient. No outer iteration or active-set logic is performed.
    Success includes valid -nu fallback; fallback_used distinguishes it from
    full construction. Failure flags describe the last attempt, not history.
    Failed/degenerate output directions are zero. rhs and nested diagnostics
    retain the last attempted calculation (and may be nonfinite on failure).
    pcg_result is the last attempted direction solve, which can precede the
    last logdet attempt if that attempt failed before reaching direction PCG.
    In mixed batches inactive items have zero RHS/directions and no Krylov
    updates; their stored diagnostics freeze. No dynamic batch slicing.
    """
    if not isinstance(oracle, MVSKOracle):
        raise TypeError("oracle must be MVSKOracle")
    oracle._validate_tensor(x, "x")
    if geom is not None and free_mask is not None:
        raise ValueError("geom and free_mask cannot both be provided")
    if logdet_mode not in ("exact", "stochastic"):
        raise ValueError("logdet_mode must be exact or stochastic")
    for name, value, lower in (("lambda_growth", lambda_growth, 1.), ("descent_factor", descent_factor, 0.)):
        if not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= lower:
            raise ValueError(f"{name} must be finite and >{lower}")
    if isinstance(max_regularization_retries, bool) or not isinstance(max_regularization_retries, int) or max_regularization_retries < 0:
        raise ValueError("max_regularization_retries must be a nonnegative integer")
    options, rtol, atol = _options(pcg_options, x.dtype)
    if logdet_mode == "stochastic":
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError("q must be an integer >=1")
        if basis_W is not None:
            raise ValueError("basis_W is only used in exact mode")
    elif probes is not None:
        raise ValueError("probes are only used in stochastic mode")
    cached = oracle._projection(x, cache)
    gradient = oracle.gradient(x, cache=cached)
    geom = build_geometry(gradient, free_mask) if geom is None else geom
    _validate_geometry(geom, x)
    batch = torch.broadcast_shapes(x.shape[:-1], geom.grad_face.shape[:-1])
    for tensor, name, tail in ((basis_W, "basis_W", 2), (probes, "probes", 2)):
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != x.dtype:
                raise TypeError(f"{name} must match x dtype")
            if tensor.device != x.device:
                raise ValueError(f"{name} must match x device")
            if tensor.ndim < tail or tensor.shape[-1 if name == "probes" else -2] != oracle.n:
                raise ValueError(f"Invalid {name} shape")
            if name == "probes" and tensor.shape[-2] != q:
                raise ValueError("probes must have shape [...,q,n]")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must be finite")
            batch = torch.broadcast_shapes(batch, tensor.shape[:-2])
    geom = _expand_geom(geom, batch)
    gradient = gradient.expand(batch+(oracle.n,))
    binding = ProjectedHessianOperator(oracle, x, geom, cache=cached, damping=lambda0)
    damping = torch.broadcast_to(binding.damping, batch).clone()
    zero = torch.zeros(batch+(oracle.n,), dtype=x.dtype, device=x.device)
    flags = torch.zeros(batch, dtype=torch.bool, device=x.device)
    degenerate = ~geom.level_regular
    zero_dim = geom.level_regular & (geom.free_count <= 2)
    _, initial_valid = kernels.assess_direction(-geom.nu, gradient, geom.grad_norm, geom.free_mask, descent_factor)
    success = zero_dim & initial_valid
    direction = _choose(success, kernels.face_tangent(-geom.nu, geom.free_mask), zero)
    normal = _choose(success, -geom.nu, zero)
    tangent, rhs = zero, zero
    logdet_failed, linear_failed = flags.clone(), flags.clone()
    descent_failed = zero_dim & ~initial_valid
    retries = torch.zeros(batch, dtype=torch.int64, device=x.device)
    logdet_status, pcg_status = None, None
    active = geom.level_regular & ~zero_dim
    if active.any().item():
        normals = _choose(active, geom.nu, zero)
        try:
            hv = oracle.hvp(x, normals, cache=cached)  # Ordinary H nu, NOT K(nu).
        except FloatingPointError:
            # Recover raw per-item finite flags after a public batch-wide error.
            # Same frozen ordinary HVP kernel, no alternative mathematics.
            hv = ordinary_hvp_unchecked(cached.z, normals, oracle.A, oracle.coefficients)
        h = level_tangent(hv, geom.free_mask, geom.nu)
        h_finite = torch.isfinite(hv).all(-1) & torch.isfinite(h).all(-1)
        seed = kernels.retry_seed(cached.z.expand(batch+(oracle.T,)), oracle.A, oracle.coefficients)
        if logdet_mode == "stochastic":
            probes = (torch.randint(0, 2, batch+(q, oracle.n), device=x.device, generator=generator).to(x.dtype)*2-1
                      if probes is None else probes.expand(batch+(q, oracle.n)))
        for attempt in range(max_regularization_retries+1):
            if not active.any().item():
                break
            ld_mask = active & h_finite
            attempted = _masked_geom(geom, ld_mask)
            candidate_ld = None
            if ld_mask.any().item():
                if logdet_mode == "exact":
                    candidate_ld = exact_logdet_correction(oracle, x, attempted, basis_W=basis_W,
                                                           cache=cached, damping=damping)
                else:
                    candidate_ld = stochastic_logdet_correction(oracle, x, attempted, q=q, probes=probes,
                                                                 cache=cached, damping=damping, pcg_options=options)
                logdet_status = _merge_logdet(logdet_status, candidate_ld, ld_mask, zero_dim)
            ld_ok = flags if candidate_ld is None else candidate_ld.success & ld_mask
            a = torch.full_like(zero, float('nan')) if candidate_ld is None else candidate_ld.correction
            candidate_rhs = kernels.assemble_rhs(h, a, geom.grad_norm, geom.free_mask, geom.nu)
            rhs = _choose(active, candidate_rhs, rhs)
            rhs_finite = torch.isfinite(candidate_rhs).all(-1)
            eligible = active & ld_ok & rhs_finite
            logdet_failed = torch.where(active, ld_mask & ~ld_ok, logdet_failed)
            linear_failed = torch.where(active, ~h_finite | (ld_ok & ~rhs_finite), linear_failed)
            descent_failed = torch.where(active, False, descent_failed)
            accepted = flags
            if eligible.any().item():
                solve_geom = _masked_geom(geom, eligible)
                operator = ProjectedHessianOperator(oracle, x, solve_geom, cache=cached, damping=damping)
                projector = lambda v: level_tangent(v, solve_geom.free_mask, solve_geom.nu)
                solve_rhs = _choose(eligible, candidate_rhs, zero)
                solve = batched_pcg(operator.matvec_unchecked, solve_rhs, projector=projector, **options)
                u, d = kernels.assemble_direction(solve.x, geom.free_mask, geom.nu)
                true_residual = candidate_rhs-operator.matvec_unchecked(u)
                residual_norm = torch.linalg.vector_norm(true_residual, dim=-1)
                rhs_norm = torch.linalg.vector_norm(candidate_rhs, dim=-1)
                finite = torch.isfinite(u).all(-1) & torch.isfinite(true_residual).all(-1) & torch.isfinite(residual_norm)
                linear_ok = solve.converged & ~solve.breakdown & finite & (residual_norm <= atol+rtol*rhs_norm)
                relative = residual_norm/torch.where(rhs_norm > 0, rhs_norm, 1)
                relative = torch.where(rhs_norm > 0, relative, torch.where(residual_norm == 0, 0, float('inf')))
                solve = replace(solve, x=u, residual_norm=residual_norm, relative_residual=relative,
                                converged=linear_ok, breakdown=solve.breakdown | ~finite)
                pcg_status = _merge_pcg(pcg_status, solve, eligible)
                _, descent_ok = kernels.assess_direction(d, gradient, geom.grad_norm, geom.free_mask, descent_factor)
                linear_failed = torch.where(eligible, ~linear_ok, linear_failed)
                descent_failed = torch.where(eligible & linear_ok, ~descent_ok, descent_failed)
                accepted = eligible & linear_ok & descent_ok
                direction = _choose(accepted, d, direction)
                normal = _choose(accepted, -geom.nu, normal)
                tangent = _choose(accepted, u, tangent)
                success = success | accepted
            active = active & ~accepted
            if attempt < max_regularization_retries:
                proposed = kernels.next_damping(damping, seed, lambda_growth)
                valid_damping = torch.isfinite(proposed) & (proposed > 0)
                linear_failed = linear_failed | (active & ~valid_damping)
                can_retry = active & valid_damping
                damping = torch.where(can_retry, proposed, damping)
                retries = retries+can_retry.to(torch.int64)
                active = can_retry
    fallback_candidates = geom.level_regular & ~success
    fallback = -geom.nu
    _, fallback_ok = kernels.assess_direction(fallback, gradient, geom.grad_norm, geom.free_mask, descent_factor)
    fallback_used = fallback_candidates & fallback_ok
    direction = _choose(fallback_used, fallback, direction)
    normal = _choose(fallback_used, fallback, normal)
    tangent = _choose(fallback_used, zero, tangent)
    success = success | fallback_used
    descent_failed = descent_failed | (fallback_candidates & ~fallback_ok)
    descent_dot = (gradient*direction).sum(-1)
    return DirectionResult(direction, normal, tangent, rhs, damping, geom.grad_norm, descent_dot,
                            success, degenerate, fallback_used, logdet_failed, linear_failed,
                            descent_failed, logdet_status, pcg_status, retries)
