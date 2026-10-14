"""Independent reduced trace/autograd references and deterministic probes."""
import pytest
import torch
from torchyand import MVSKData, MVSKOracle, prepare_mvsk_data
from torchyand.geometry import build_geometry, project_level_tangent
from torchyand.logdet import (
    LogDetStatus, exact_logdet_correction, stochastic_logdet_correction,
)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def problem(dtype, batch=()):
    rng = torch.Generator().manual_seed(20261024)
    R = torch.randn(24, 5, dtype=dtype, generator=rng)
    oracle = MVSKOracle(prepare_mvsk_data(R), [0., 1., .02, .04])
    x = .2*torch.randn(batch+(5,), dtype=dtype, generator=rng)
    geom = build_geometry(oracle.gradient(x))
    assert geom.level_regular.all()
    return oracle, x, geom


def close(a, b, dtype):
    tol = 8e-5 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def independent_basis(geom):
    m = geom.free_mask.to(geom.nu.dtype)
    F = torch.diag(m)-torch.outer(m, m)/m.sum().clamp_min(1)
    P = F-torch.outer(geom.nu, geom.nu)
    values, vectors = torch.linalg.eigh(P)
    return vectors[:, values > .5], P


def reference(oracle, x, geom, damping, probes=None):
    # Direct sample formulas, dense reduced solve, and independent contraction;
    # neither frozen analytic kernels nor either correction backend is called.
    W, P = independent_basis(geom)
    _, c2, c3, c4 = oracle.coefficients
    z = oracle.A@x
    weight = 2*c2-6*c3*z+12*c4*z.square()
    H = oracle.A.T@(weight[:, None]*oracle.A)/oracle.T
    reduced = W.T@H@W+damping*torch.eye(W.shape[1], dtype=x.dtype, device=x.device)
    if probes is None:
        xi = W
    else:
        xi = P@probes.T
    y = W@torch.linalg.solve(reduced, W.T@xi)
    third = -6*c3+24*c4*z
    raw = third[:, None]*(oracle.A@y)*(oracle.A@xi)
    weight = raw.sum(-1) if probes is None else raw.mean(-1)
    a = P@(oracle.A.T@weight/oracle.T)
    return P@a


@pytest.mark.parametrize("damping", [0., .3])
def test_exact_dense_reference(dtype, damping):
    oracle, x, geom = problem(dtype)
    W, _ = independent_basis(geom)
    result = exact_logdet_correction(oracle, x, geom, basis_W=W, damping=damping)
    assert result.success and result.status == LogDetStatus.SUCCESS
    assert result.cholesky_info == 0
    close(result.correction, reference(oracle, x, geom, damping), dtype)
    close(result.correction, exact_logdet_correction(oracle, x, geom, damping=damping).correction, dtype)
    close(project_level_tangent(result.correction, geom), result.correction, dtype)


def test_exact_independent_logdet_gradient(dtype):
    oracle, x, geom = problem(dtype)
    W, P = independent_basis(geom)
    W, P = W.detach(), P.detach()
    xx = x.detach().requires_grad_()
    _, c2, c3, c4 = oracle.coefficients
    z = oracle.A@xx
    weight = 2*c2-6*c3*z+12*c4*z.square()
    reduced = W.T@(oracle.A.T@(weight[:, None]*oracle.A)/oracle.T)@W
    reduced = reduced+.2*torch.eye(W.shape[1], dtype=dtype)
    sign, logdet = torch.linalg.slogdet(reduced)
    assert sign > 0
    gradient, = torch.autograd.grad(logdet, xx)
    result = exact_logdet_correction(oracle, x, geom, damping=.2)
    close(result.correction, P@gradient, dtype)


@pytest.mark.parametrize("q", [1, 8])
@pytest.mark.parametrize("damping", [0., .3])
def test_stochastic_fixed_probe_reference(dtype, q, damping):
    oracle, x, geom = problem(dtype)
    rng = torch.Generator().manual_seed(20261025)
    probes = torch.randn(q, 5, dtype=dtype, generator=rng)
    result = stochastic_logdet_correction(oracle, x, geom, q=q, damping=damping, probes=probes)
    assert result.success and result.pcg_result.converged.all()
    assert result.pcg_result.residual_norm.shape == (q,)
    close(result.correction, reference(oracle, x, geom, damping, probes), dtype)
    close(project_level_tangent(result.correction, geom), result.correction, dtype)


def test_projected_probes_not_normalized(dtype):
    oracle, x, geom = problem(dtype)
    probes = torch.tensor([[1., 2., -3., .4, 5.]], dtype=dtype)
    a = stochastic_logdet_correction(oracle, x, geom, q=1, damping=.2, probes=probes)
    b = stochastic_logdet_correction(oracle, x, geom, q=1, damping=.2, probes=3*probes)
    assert a.success and b.success
    assert a.correction.norm() > 1e-3
    close(b.correction, 9*a.correction, dtype)


def test_canonical_probes_reproduce_exact(dtype):
    oracle, x, geom = problem(dtype)
    W, _ = independent_basis(geom)
    q = W.shape[1]
    probes = q**.5*W.T
    stochastic = stochastic_logdet_correction(oracle, x, geom, q=q, probes=probes, damping=.2)
    exact = exact_logdet_correction(oracle, x, geom, basis_W=W, damping=.2)
    assert stochastic.success
    close(stochastic.correction, exact.correction, dtype)


def test_stochastic_large_q_statistics(dtype):
    oracle, x, geom = problem(dtype)
    exact = exact_logdet_correction(oracle, x, geom, damping=.2).correction
    rng = torch.Generator().manual_seed(20261026)
    # Eight independent fixed-seed estimates per q; compare mean squared error,
    # never require monotonic improvement for an individual realization.
    errors = []
    for q in (8, 1024):
        probes = torch.randn(8, q, 5, dtype=dtype, generator=rng)
        result = stochastic_logdet_correction(oracle, x, geom, q=q, probes=probes, damping=.2)
        assert result.success.all()
        errors.append((result.correction-exact).square().sum(-1).mean())
    assert errors[1] < .2*errors[0]


def test_zero_dimensional_regular(dtype):
    R = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], dtype=dtype)
    oracle = MVSKOracle(prepare_mvsk_data(R), [0., 1., 0., .1])
    x = torch.tensor([.2, .5], dtype=dtype)
    geom = build_geometry(oracle.gradient(x))
    assert geom.level_regular and geom.free_count == 2
    exact = exact_logdet_correction(oracle, x, geom, basis_W=torch.empty(2, 0, dtype=dtype))
    stochastic = stochastic_logdet_correction(oracle, x, geom, q=3, probes=torch.ones(3, 2, dtype=dtype))
    for result in (exact, stochastic):
        assert result.success
        assert torch.equal(result.correction, torch.zeros_like(x))
    assert (stochastic.pcg_result.iterations == 0).all()


def test_degenerate_geometry(dtype):
    oracle, x, _ = problem(dtype)
    geom = build_geometry(torch.zeros_like(x))
    for result in (exact_logdet_correction(oracle, x, geom),
                   stochastic_logdet_correction(oracle, x, geom, q=1)):
        assert not result.success and result.status == LogDetStatus.DEGENERATE
        assert torch.isnan(result.correction).all()


def test_cholesky_and_pcg_failure(dtype):
    oracle, x, geom = problem(dtype)
    zero = MVSKOracle(oracle.data, [0., 0., 0., 0.])
    exact = exact_logdet_correction(zero, x, geom)
    assert exact.status == LogDetStatus.NOT_SPD and exact.cholesky_info > 0
    assert not exact.success and torch.isnan(exact.correction).all()
    probes = torch.tensor([[1., 0., -1., 2., .5]], dtype=dtype)
    stochastic = stochastic_logdet_correction(zero, x, geom, q=1, probes=probes)
    assert stochastic.status == LogDetStatus.LINEAR_SOLVE_FAILED
    assert stochastic.pcg_result.breakdown.all() and stochastic.pcg_result.nonpositive_curvature.all()
    assert not stochastic.success and torch.isnan(stochastic.correction).all()
    damped = exact_logdet_correction(zero, x, geom, damping=.5)
    assert damped.success and torch.equal(damped.correction, torch.zeros_like(x))


def test_one_failed_probe_invalidates_item(dtype):
    oracle, x, geom = problem(dtype)
    W, _ = independent_basis(geom)
    probes = torch.stack((torch.zeros_like(x), W[:, 0]))
    result = stochastic_logdet_correction(oracle, x, geom, q=2, probes=probes,
                                          pcg_options={"max_iter": 0})
    assert torch.equal(result.pcg_result.converged, torch.tensor([True, False]))
    assert not result.success and result.status == LogDetStatus.LINEAR_SOLVE_FAILED
    assert torch.isnan(result.correction).all()


def test_partial_batch_failure(dtype):
    oracle, x, geom = problem(dtype)
    W, _ = independent_basis(geom)
    probes = torch.stack((torch.zeros(1, 5, dtype=dtype), W[:, 0].reshape(1, 5)))
    result = stochastic_logdet_correction(oracle, x, geom, q=1, probes=probes,
                                          pcg_options={"max_iter": 0})
    assert torch.equal(result.success, torch.tensor([True, False]))
    assert torch.equal(result.correction[0], torch.zeros_like(x))
    assert torch.isnan(result.correction[1]).all()


def test_mixed_spd_and_indefinite_batch(dtype):
    R = torch.tensor([[2., 0., 0.], [-1., 0., 0.], [-1., 0., 0.],
                      [0., 2., 0.], [0., -1., 0.], [0., -1., 0.],
                      [0., 0., 2.], [0., 0., -1.], [0., 0., -1.]], dtype=dtype)
    oracle = MVSKOracle(prepare_mvsk_data(R), [0., 1., 1., 0.])
    x = torch.tensor([[-.1, 0., .1], [1., 1.1, 1.2]], dtype=dtype)
    geom = build_geometry(oracle.gradient(x))
    assert geom.level_regular.all()
    exact = exact_logdet_correction(oracle, x, geom)
    stochastic = stochastic_logdet_correction(oracle, x, geom, q=1,
                                               probes=torch.tensor([[.2, 1., -.3]], dtype=dtype))
    for result in (exact, stochastic):
        assert torch.equal(result.success, torch.tensor([True, False]))
        assert torch.isfinite(result.correction[0]).all()
        assert torch.isnan(result.correction[1]).all()
    assert exact.status[1] == LogDetStatus.NOT_SPD
    assert stochastic.status[1] == LogDetStatus.LINEAR_SOLVE_FAILED
    assert stochastic.pcg_result.nonpositive_curvature[1].all()


def test_finite_inputs_arithmetic_overflow(dtype):
    oracle, x, _ = problem(dtype)
    huge = MVSKOracle(oracle.data, [0., 0., 0., torch.finfo(dtype).max/2])
    x = torch.zeros_like(x)
    geom = build_geometry(torch.tensor([1., -1., 0., 0., 0.], dtype=dtype))
    exact = exact_logdet_correction(huge, x, geom)
    stochastic = stochastic_logdet_correction(huge, x, geom, q=1,
                                               probes=torch.tensor([[1., 0., 2., -1., .5]], dtype=dtype))
    assert exact.status == LogDetStatus.NONFINITE
    assert stochastic.status == LogDetStatus.LINEAR_SOLVE_FAILED
    for result in (exact, stochastic):
        assert not result.success and torch.isnan(result.correction).all()


def test_damping_recomputed(dtype):
    oracle, x, geom = problem(dtype)
    rng = torch.Generator().manual_seed(20261027)
    probes = torch.randn(4, 5, dtype=dtype, generator=rng)
    exact0 = exact_logdet_correction(oracle, x, geom, damping=0.)
    exact1 = exact_logdet_correction(oracle, x, geom, damping=2.)
    assert (exact0.correction-exact1.correction).norm() > 1e-3
    for damping in (0., 2.):
        result = stochastic_logdet_correction(oracle, x, geom, q=4, damping=damping, probes=probes)
        close(result.correction, reference(oracle, x, geom, damping, probes), dtype)
        assert result.damping == damping


def test_batch_vs_single(dtype):
    oracle, x, geom = problem(dtype, (2, 3))
    rng = torch.Generator().manual_seed(20261028)
    probes = torch.randn(2, 3, 4, 5, dtype=dtype, generator=rng)
    damping = torch.tensor([.1, .2, .3], dtype=dtype)
    cached = oracle.cache(x)
    exact = exact_logdet_correction(oracle, x, geom, cache=cached, damping=damping)
    stochastic = stochastic_logdet_correction(oracle, x, geom, q=4, probes=probes,
                                               cache=cached, damping=damping)
    assert exact.success.all() and stochastic.success.all()
    assert stochastic.pcg_result.x.shape == (2, 3, 4, 5)
    for i in range(2):
        for j in range(3):
            gi = build_geometry(oracle.gradient(x[i, j]))
            close(exact.correction[i, j], exact_logdet_correction(oracle, x[i, j], gi, damping=damping[j]).correction, dtype)
            single = stochastic_logdet_correction(oracle, x[i, j], gi, q=4,
                                                   probes=probes[i, j], damping=damping[j])
            close(stochastic.correction[i, j], single.correction, dtype)


def test_variable_face_dimensions(dtype):
    oracle, x, _ = problem(dtype, (4,))
    masks = torch.arange(5).unsqueeze(0) < torch.tensor([5, 4, 2, 1]).unsqueeze(-1)
    geom = build_geometry(oracle.gradient(x), masks)
    exact = exact_logdet_correction(oracle, x, geom, damping=.2)
    stochastic = stochastic_logdet_correction(oracle, x, geom, q=2,
                                               probes=torch.ones(4, 2, 5, dtype=dtype), damping=.2)
    for result in (exact, stochastic):
        assert torch.equal(result.success, torch.tensor([True, True, True, False]))
        assert torch.equal(result.correction[2], torch.zeros(5, dtype=dtype))
        assert torch.isnan(result.correction[3]).all()


@pytest.mark.parametrize("probe", ["rademacher", "gaussian"])
def test_seeded_generator(dtype, probe):
    oracle, x, geom = problem(dtype)
    a = stochastic_logdet_correction(oracle, x, geom, q=4, probe=probe,
                                      generator=torch.Generator().manual_seed(77), damping=.2)
    b = stochastic_logdet_correction(oracle, x, geom, q=4, probe=probe,
                                      generator=torch.Generator().manual_seed(77), damping=.2)
    assert a.success and b.success
    torch.testing.assert_close(a.correction, b.correction, rtol=0, atol=0)


def test_adaptive_fixed(dtype):
    oracle, x, geom = problem(dtype)
    probes = torch.arange(20, dtype=dtype).reshape(4, 5)/10
    a = stochastic_logdet_correction(oracle, x, geom, q=4, probes=probes)
    b = stochastic_logdet_correction(oracle, x, geom, q=4, probes=probes, pcg_options={"mode": "fixed"})
    assert a.success and b.success
    close(a.correction, b.correction, dtype)
    assert torch.equal(a.pcg_result.iterations, b.pcg_result.iterations)


@pytest.mark.parametrize("bad", ["negative", "nan", "inf", "dtype", "probe_dtype", "probe_shape", "probe_nan", "q", "basis"])
def test_validation(bad):
    oracle, x, geom = problem(torch.float32)
    kwargs = {}
    if bad in ("negative", "nan", "inf", "dtype"):
        kwargs['damping'] = {"negative": -1., "nan": float('nan'), "inf": float('inf'),
                             "dtype": torch.tensor(.1, dtype=torch.float64)}[bad]
        for fn in (exact_logdet_correction, stochastic_logdet_correction):
            with pytest.raises((TypeError, ValueError)):
                fn(oracle, x, geom, **kwargs)
    elif bad == "basis":
        with pytest.raises(ValueError):
            exact_logdet_correction(oracle, x, geom, basis_W=torch.ones(5, 3))
    else:
        kwargs = {"q": 1}
        if bad == "q":
            kwargs['q'] = 0
        else:
            kwargs['probes'] = {"probe_dtype": torch.ones(1, 5, dtype=torch.float64),
                                "probe_shape": torch.ones(1, 1),
                                "probe_nan": torch.full((1, 5), float('nan'))}[bad]
        with pytest.raises((TypeError, ValueError)):
            stochastic_logdet_correction(oracle, x, geom, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    oracle, x, geom = problem(dtype, (2,))
    gpu = MVSKOracle(MVSKData(oracle.A.cuda(), oracle.mu.cuda(), oracle.T, oracle.n), oracle.coefficients.cuda())
    xg = x.cuda()
    gg = build_geometry(gpu.gradient(xg))
    probes = torch.arange(30, dtype=dtype).reshape(2, 3, 5)/10
    exact = exact_logdet_correction(oracle, x, geom, damping=.2)
    exactg = exact_logdet_correction(gpu, xg, gg, damping=.2)
    stochastic = stochastic_logdet_correction(oracle, x, geom, q=3, probes=probes, damping=.2)
    stochasticg = stochastic_logdet_correction(gpu, xg, gg, q=3, probes=probes.cuda(), damping=.2)
    assert exactg.success.all() and stochasticg.success.all()
    close(exactg.correction.cpu(), exact.correction, dtype)
    close(stochasticg.correction.cpu(), stochastic.correction, dtype)
    with pytest.raises(ValueError):
        stochastic_logdet_correction(gpu, xg, gg, q=3, probes=probes)
