"""Dense affine-normal references and regularization/failure contracts."""
from dataclasses import replace
import pytest
import torch
from torchyand import MVSKData, MVSKOracle, prepare_mvsk_data
from torchyand.geometry import build_geometry, project_face_tangent, project_level_tangent
from torchyand.geometry.affine_normal import affine_normal_direction
import torchyand.geometry.affine_normal as module
from torchyand.logdet import LogDetStatus
from torchyand.operators import ProjectedHessianOperator


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def problem(dtype, batch=(), masked=False):
    rng = torch.Generator().manual_seed(20261030)
    R = torch.randn(24, 5, dtype=dtype, generator=rng)
    oracle = MVSKOracle(prepare_mvsk_data(R), [0., 1., .02, .04])
    x = .2*torch.randn(batch+(5,), dtype=dtype, generator=rng)
    mask = torch.tensor([True, True, False, True, True]) if masked else None
    geom = build_geometry(oracle.gradient(x), mask)
    assert geom.level_regular.all()
    return oracle, x, geom


def close(a, b, dtype):
    tol = 1e-4 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def dense_reference(oracle, x, geom, damping):
    # Independent small-n sample formulas, eigenspace basis, trace and solve.
    m = geom.free_mask.to(x.dtype)
    F = torch.diag(m)-torch.outer(m, m)/m.sum()
    P = F-torch.outer(geom.nu, geom.nu)
    values, vectors = torch.linalg.eigh(P)
    W = vectors[:, values > .5]
    z = oracle.A@x
    _, c2, c3, c4 = oracle.coefficients
    weight = 2*c2-6*c3*z+12*c4*z.square()
    H = oracle.A.T@(weight[:, None]*oracle.A)/oracle.T
    K = W.T@H@W+damping*torch.eye(W.shape[-1], dtype=x.dtype, device=x.device)
    Y = W@torch.linalg.solve(K, torch.eye(W.shape[-1], dtype=x.dtype, device=x.device))
    third = -6*c3+24*c4*z
    raw = (third[:, None]*(oracle.A@Y)*(oracle.A@W)).sum(-1)
    a = P@(oracle.A.T@raw/oracle.T)
    a = P@a
    h = P@(H@geom.nu)
    rhs = P@(h-geom.grad_norm/oracle.n*a)
    u = W@torch.linalg.solve(K, W.T@rhs)
    d = F@(u-geom.nu)
    return W, a, rhs, u, d


@pytest.mark.parametrize("damping", [0., .3])
@pytest.mark.parametrize("masked", [False, True])
def test_exact_dense_reference(dtype, damping, masked):
    oracle, x, geom = problem(dtype, masked=masked)
    W, a, rhs, u, d = dense_reference(oracle, x, geom, damping)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=damping,
                                     logdet_mode="exact", basis_W=W, max_regularization_retries=0)
    assert result.success and not result.fallback_used
    close(result.logdet_status.correction, a, dtype)
    close(result.rhs, rhs, dtype)
    close(result.tangent_correction, u, dtype)
    close(result.direction, d, dtype)
    assert not result.logdet_failed and not result.linear_failed and not result.descent_failed


def test_decomposition_tangent_descent(dtype):
    oracle, x, geom = problem(dtype, masked=True)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.2, logdet_mode="exact")
    assert result.success and not result.fallback_used
    close(result.direction, result.normal_component+result.tangent_correction, dtype)
    torch.testing.assert_close(result.normal_component, -geom.nu, rtol=0, atol=0)
    close(project_level_tangent(result.tangent_correction, geom), result.tangent_correction, dtype)
    close(geom.nu@result.tangent_correction, torch.zeros((), dtype=dtype), dtype)
    close(project_face_tangent(result.direction, geom.free_mask), result.direction, dtype)
    assert torch.equal(torch.where(geom.free_mask, 0, result.direction), torch.zeros_like(x))
    close(result.direction.sum(), torch.zeros((), dtype=dtype), dtype)
    close(geom.grad_face@result.direction, -geom.grad_norm, dtype)
    close(result.descent_dot, oracle.gradient(x)@result.direction, dtype)
    assert result.descent_dot < -64*torch.finfo(dtype).eps*geom.grad_norm*result.direction.norm()
    operator = ProjectedHessianOperator(oracle, x, geom, damping=result.lambda_reg)
    residual = (result.rhs-operator.matvec(result.tangent_correction)).norm()
    assert residual <= (1e-5 if dtype == torch.float32 else 1e-10)*result.rhs.norm()
    close(result.pcg_result.residual_norm, residual, dtype)


def test_auto_geometry_cache_and_stochastic_exact(dtype):
    oracle, x, geom = problem(dtype)
    W, _, _, _, _ = dense_reference(oracle, x, geom, .2)
    q = W.shape[-1]
    probes = q**.5*W.T
    exact = affine_normal_direction(oracle, x, geom=geom, lambda0=.2, logdet_mode="exact")
    stochastic = affine_normal_direction(oracle, x, lambda0=.2, q=q, probes=probes, cache=oracle.cache(x))
    assert exact.success and stochastic.success and not stochastic.fallback_used
    close(stochastic.direction, exact.direction, dtype)
    close(stochastic.rhs, exact.rhs, dtype)


def indefinite_problem(dtype, x=None):
    R = torch.tensor([[2., 0., 0.], [-1., 0., 0.], [-1., 0., 0.],
                      [0., 2., 0.], [0., -1., 0.], [0., -1., 0.],
                      [0., 0., 2.], [0., 0., -1.], [0., 0., -1.]], dtype=dtype)
    oracle = MVSKOracle(prepare_mvsk_data(R), [0., 1., 1., 0.])
    x = torch.tensor([1., 1.1, 1.2], dtype=dtype) if x is None else x
    return oracle, x, build_geometry(oracle.gradient(x))


def test_regularization_retry(dtype, monkeypatch):
    oracle, x, geom = indefinite_problem(dtype)
    called = []
    real = module.exact_logdet_correction
    def observed(*args, **kwargs):
        result = real(*args, **kwargs)
        called.append((kwargs['damping'].clone(), result.success.clone()))
        return result
    monkeypatch.setattr(module, 'exact_logdet_correction', observed)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, lambda_growth=10.,
                                     max_regularization_retries=2, logdet_mode="exact")
    assert result.success and not result.fallback_used
    assert result.regularization_retries == 2
    close(result.lambda_reg, torch.tensor(10., dtype=dtype), dtype)
    assert [bool(item[1]) for item in called] == [False, False, True]
    assert not result.logdet_failed and not result.linear_failed
    close(result.direction, dense_reference(oracle, x, geom, result.lambda_reg)[-1], dtype)


def fail_first_direction_solve(monkeypatch):
    real = module.batched_pcg
    calls = []
    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            return replace(result, converged=torch.zeros_like(result.converged),
                           max_iter_reached=torch.ones_like(result.max_iter_reached))
        return result
    monkeypatch.setattr(module, 'batched_pcg', wrapped)
    return calls


def test_correction_recomputed_on_retry(dtype, monkeypatch):
    oracle, x, geom = problem(dtype)
    fail_first_direction_solve(monkeypatch)
    real = module.exact_logdet_correction
    called = []
    def observed(*args, **kwargs):
        result = real(*args, **kwargs)
        called.append(result)
        return result
    monkeypatch.setattr(module, 'exact_logdet_correction', observed)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, max_regularization_retries=1,
                                     logdet_mode="exact")
    assert result.success and not result.fallback_used
    assert len(called) == 2 and all(item.success for item in called)
    assert (called[0].correction-called[1].correction).norm() > 1e-4
    close(result.direction, dense_reference(oracle, x, geom, 1.)[-1], dtype)


@pytest.mark.parametrize("explicit", [False, True])
def test_stochastic_probes_reused(dtype, monkeypatch, explicit):
    oracle, x, geom = problem(dtype)
    fail_first_direction_solve(monkeypatch)
    real = module.stochastic_logdet_correction
    seen = []
    def observed(*args, **kwargs):
        seen.append(kwargs['probes'])
        return real(*args, **kwargs)
    monkeypatch.setattr(module, 'stochastic_logdet_correction', observed)
    generator = torch.Generator().manual_seed(20261031)
    raw = torch.arange(20, dtype=dtype).reshape(4, 5)/10 if explicit else None
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, q=4, probes=raw,
                                     generator=generator, max_regularization_retries=1)
    assert result.success and not result.fallback_used and result.regularization_retries == 1
    assert len(seen) == 2 and seen[0] is seen[1]
    expected = torch.Generator().manual_seed(20261031)
    if not explicit:
        torch.randint(0, 2, (4, 5), generator=expected)
    assert torch.equal(generator.get_state(), expected.get_state())


@pytest.mark.parametrize("cause", ["max_iter", "breakdown", "logdet"])
def test_failure_fallback(dtype, cause):
    oracle, x, geom = indefinite_problem(dtype) if cause == "logdet" else problem(dtype)
    options = {"max_iter": 0} if cause == "max_iter" else ({"M_inv": lambda r: -r} if cause == "breakdown" else None)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, logdet_mode="exact",
                                     pcg_options=options, max_regularization_retries=0)
    assert result.success and result.fallback_used
    assert result.logdet_failed == (cause == "logdet")
    assert result.linear_failed == (cause != "logdet")
    torch.testing.assert_close(result.direction, -geom.nu, rtol=0, atol=0)
    assert torch.equal(result.tangent_correction, torch.zeros_like(x))
    assert result.descent_dot < 0
    if cause == "logdet":
        assert result.logdet_status.status == LogDetStatus.NOT_SPD and result.pcg_result is None
    else:
        assert not result.pcg_result.converged
        assert result.pcg_result.breakdown == (cause == "breakdown")
        assert result.pcg_result.max_iter_reached == (cause == "max_iter")


def test_postprojection_true_residual_recheck(dtype, monkeypatch):
    oracle, x, geom = problem(dtype)
    real = module.batched_pcg
    def spuriously_converged(*args, **kwargs):
        result = real(*args, **kwargs)
        return replace(result, x=torch.zeros_like(result.x), converged=torch.ones_like(result.converged))
    monkeypatch.setattr(module, 'batched_pcg', spuriously_converged)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, logdet_mode="exact",
                                     max_regularization_retries=0)
    assert result.linear_failed and result.fallback_used
    assert not result.pcg_result.converged
    close(result.pcg_result.residual_norm, result.rhs.norm(), dtype)


def test_descent_and_fallback_failure(dtype):
    oracle, x, geom = problem(dtype)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, logdet_mode="exact",
                                     max_regularization_retries=0, descent_factor=2/torch.finfo(dtype).eps)
    assert not result.success and not result.fallback_used
    assert result.descent_failed and not result.linear_failed and not result.logdet_failed
    assert torch.equal(result.direction, torch.zeros_like(x))


@pytest.mark.parametrize("mode", ["exact", "stochastic"])
def test_degenerate_skips_solves(dtype, mode, monkeypatch):
    oracle, x, _ = problem(dtype)
    x = torch.zeros_like(x)
    def forbidden(*args, **kwargs):
        pytest.fail("Degenerate items must not run logdet or PCG")
    monkeypatch.setattr(module, 'exact_logdet_correction', forbidden)
    monkeypatch.setattr(module, 'stochastic_logdet_correction', forbidden)
    monkeypatch.setattr(module, 'batched_pcg', forbidden)
    result = affine_normal_direction(oracle, x, logdet_mode=mode)
    assert result.degenerate and not result.success and not result.fallback_used
    for value in (result.direction, result.normal_component, result.tangent_correction):
        assert torch.equal(value, torch.zeros_like(x))
    assert result.logdet_status is None and result.pcg_result is None


def test_zero_level_tangent_skips_solves(dtype, monkeypatch):
    rng = torch.Generator().manual_seed(20261101)
    oracle = MVSKOracle(prepare_mvsk_data(torch.randn(8, 2, dtype=dtype, generator=rng)), [0., 1., 0., .1])
    x = torch.tensor([.2, .5], dtype=dtype)
    geom = build_geometry(oracle.gradient(x))
    assert geom.level_regular
    def forbidden(*args, **kwargs):
        pytest.fail("Empty level tangent needs no solve")
    monkeypatch.setattr(module, 'exact_logdet_correction', forbidden)
    monkeypatch.setattr(module, 'stochastic_logdet_correction', forbidden)
    monkeypatch.setattr(module, 'batched_pcg', forbidden)
    result = affine_normal_direction(oracle, x, geom=geom)
    assert result.success and not result.fallback_used
    close(result.direction, -geom.nu, dtype)
    assert torch.equal(result.rhs, torch.zeros_like(x))
    assert torch.equal(result.tangent_correction, torch.zeros_like(x))
    assert result.logdet_status is None and result.pcg_result is None


def test_zero_lambda_positive_retry_seed(dtype):
    oracle, x, _ = problem(dtype)
    mean = MVSKOracle(oracle.data, [1., 0., 0., 0.])
    result = affine_normal_direction(mean, x, lambda0=0., logdet_mode="exact", max_regularization_retries=1)
    assert result.success and not result.fallback_used and result.regularization_retries == 1
    expected = torch.tensor(128*torch.finfo(dtype).eps, dtype=dtype)
    torch.testing.assert_close(result.lambda_reg, expected, rtol=0, atol=0)
    assert result.logdet_status.damping == expected


def test_mixed_batch_freezes_success(dtype):
    x = torch.tensor([[-.1, 0., .1], [1., 1.1, 1.2], [10., 10.1, 10.2], [0., 0., 0.]], dtype=dtype)
    oracle, x, geom = indefinite_problem(dtype, x)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.1, logdet_mode="exact",
                                     max_regularization_retries=2)
    assert torch.equal(result.success, torch.tensor([True, True, True, False]))
    assert torch.equal(result.fallback_used, torch.tensor([False, False, True, False]))
    assert torch.equal(result.degenerate, torch.tensor([False, False, False, True]))
    assert torch.equal(result.regularization_retries, torch.tensor([0, 2, 2, 0]))
    close(result.lambda_reg, torch.tensor([.1, 10., 10., .1], dtype=dtype), dtype)
    close(result.logdet_status.damping[0], torch.tensor(.1, dtype=dtype), dtype)
    gi = build_geometry(oracle.gradient(x[0]))
    first = affine_normal_direction(oracle, x[0], geom=gi, lambda0=.1, logdet_mode="exact", max_regularization_retries=0)
    close(result.direction[0], first.direction, dtype)
    close(result.rhs[0], first.rhs, dtype)
    assert result.pcg_result.iterations[0] == first.pcg_result.iterations
    assert result.logdet_failed[2]
    assert torch.equal(result.direction[3], torch.zeros(3, dtype=dtype))


def test_batch_vs_single(dtype):
    oracle, x, geom = problem(dtype, (2, 3), masked=True)
    result = affine_normal_direction(oracle, x, geom=geom, lambda0=.2, logdet_mode="exact")
    assert result.success.all() and not result.fallback_used.any()
    for i in range(2):
        for j in range(3):
            gi = build_geometry(oracle.gradient(x[i, j]), geom.free_mask[i, j])
            single = affine_normal_direction(oracle, x[i, j], geom=gi, lambda0=.2, logdet_mode="exact")
            close(result.direction[i, j], single.direction, dtype)
            close(result.tangent_correction[i, j], single.tangent_correction, dtype)
    assert result.direction.shape == (2, 3, 5)
    assert result.lambda_reg.shape == (2, 3)


@pytest.mark.parametrize("bad", ["geom_mask", "negative", "nan", "dtype", "mode", "q", "growth", "retries", "options"])
def test_validation(bad):
    oracle, x, geom = problem(torch.float32)
    kwargs = {"geom_mask": {"geom": geom, "free_mask": geom.free_mask},
              "negative": {"lambda0": -1.}, "nan": {"lambda0": float('nan')},
              "dtype": {"lambda0": torch.tensor(.1, dtype=torch.float64)},
              "mode": {"logdet_mode": "none"}, "q": {"q": 0},
              "growth": {"lambda_growth": 1.}, "retries": {"max_regularization_retries": -1},
              "options": {"pcg_options": {"projector": lambda v: v}}}[bad]
    with pytest.raises((TypeError, ValueError)):
        affine_normal_direction(oracle, x, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    oracle, x, geom = problem(dtype, (2,), masked=True)
    gpu = MVSKOracle(MVSKData(oracle.A.cuda(), oracle.mu.cuda(), oracle.T, oracle.n), oracle.coefficients.cuda())
    xx = x.cuda()
    gg = build_geometry(gpu.gradient(xx), geom.free_mask.cuda())
    cpu = affine_normal_direction(oracle, x, geom=geom, lambda0=.2, logdet_mode="exact")
    result = affine_normal_direction(gpu, xx, geom=gg, lambda0=.2, logdet_mode="exact")
    assert result.success.all() and not result.fallback_used.any()
    close(result.direction.cpu(), cpu.direction, dtype)
    probes = torch.arange(40, dtype=dtype).reshape(2, 4, 5)/10
    cpu_s = affine_normal_direction(oracle, x, geom=geom, lambda0=.2, q=4, probes=probes)
    gpu_s = affine_normal_direction(gpu, xx, geom=gg, lambda0=.2, q=4, probes=probes.cuda())
    assert gpu_s.success.all() and not gpu_s.fallback_used.any()
    close(gpu_s.direction.cpu(), cpu_s.direction, dtype)
