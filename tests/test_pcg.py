"""Independent dense solves and per-item termination/curvature contracts."""
import pytest
import torch
from torchyand.linalg import batched_pcg
from torchyand.geometry import build_geometry, project_level_tangent
from torchyand.operators import ProjectedHessianOperator
from torchyand import MVSKOracle, prepare_mvsk_data


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def close(a, b, dtype):
    tol = 5e-5 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def matvec(A):
    return lambda v: (A@v.unsqueeze(-1)).squeeze(-1)


def assert_result(result, b):
    assert result.x.shape == b.shape
    for name in ("residual_norm", "relative_residual", "converged", "iterations",
                 "breakdown", "nonpositive_curvature", "max_iter_reached"):
        assert getattr(result, name).shape == b.shape[:-1]
    assert torch.isfinite(result.x).all()
    assert not result.breakdown.any()
    assert result.converged.all()


@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
def test_identity(dtype, mode):
    b = torch.arange(24, dtype=dtype).reshape(2, 3, 4)/10
    result = batched_pcg(lambda v: v, b, mode=mode)
    assert_result(result, b)
    close(result.x, b, dtype)
    assert (result.iterations == 1).all()


def test_diagonal(dtype):
    d = torch.tensor([1., 2., 4., 8.], dtype=dtype)
    b = torch.tensor([[1., 2., 3., 4.], [-1., .3, 2., 1.]], dtype=dtype)
    result = batched_pcg(lambda v: d*v, b, max_iter=30)
    assert_result(result, b)
    close(result.x, b/d, dtype)


def random_spd(dtype):
    rng = torch.Generator().manual_seed(20261022)
    Q, _ = torch.linalg.qr(torch.randn(3, 6, 6, dtype=dtype, generator=rng))
    eigenvalues = torch.stack([torch.linspace(1., k, 6, dtype=dtype) for k in (1., 10., 100.)])
    A = (Q*eigenvalues.unsqueeze(-2))@Q.transpose(-1, -2)
    b = torch.randn(3, 6, dtype=dtype, generator=rng)
    return A, b


def test_random_spd_dense_solve(dtype):
    A, b = random_spd(dtype)
    result = batched_pcg(matvec(A), b, max_iter=100)
    assert_result(result, b)
    close(result.x, torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1), dtype)
    close(result.residual_norm, torch.linalg.vector_norm(b-matvec(A)(result.x), dim=-1), dtype)
    assert (result.residual_norm <= (1e-5 if dtype == torch.float32 else 1e-10)*b.norm(dim=-1)).all()


def test_zero_rhs(dtype):
    b = torch.zeros(2, 3, 5, dtype=dtype)
    result = batched_pcg(lambda v: v, b)
    assert_result(result, b)
    assert (result.iterations == 0).all()
    assert torch.equal(result.relative_residual, torch.zeros_like(result.relative_residual))


def test_exact_initial_solution(dtype):
    b = torch.tensor([[1., 2., 3.], [1., 2., 3.]], dtype=dtype)
    result = batched_pcg(lambda v: 2*v, b, x0=b[0]/2)
    assert_result(result, b)
    assert (result.iterations == 0).all()
    assert torch.equal(result.x, b/2)


def test_near_zero_rhs_absolute_tolerance(dtype):
    b = torch.full((3,), 1e-12, dtype=dtype)
    result = batched_pcg(lambda v: v, b, atol=1e-10)
    assert result.converged and result.iterations == 0
    assert torch.equal(result.x, torch.zeros_like(b))
    assert result.residual_norm > 0


def test_per_item_freezing(dtype):
    diagonal = torch.tensor([[1., 1., 1.], [1., 2., 3.], [1., 1., 1.]], dtype=dtype)
    b = torch.tensor([[1., 2., 3.], [1., 1., 1.], [0., 0., 0.]], dtype=dtype)
    calls = []
    def apply(v):
        calls.append(v.detach().clone())
        return diagonal*v
    result = batched_pcg(apply, b, mode="fixed", max_iter=15)
    assert_result(result, b)
    assert result.iterations[0] == 1 and result.iterations[2] == 0
    assert result.iterations[1] > 1
    assert len(calls) == 17  # initial matvec, 15 steps, final matvec
    assert all(torch.equal(v[0], torch.zeros_like(v[0])) for v in calls[2:-1])
    assert all(torch.equal(v[2], torch.zeros_like(v[2])) for v in calls[1:-1])
    assert torch.equal(result.x[0], b[0])


@pytest.mark.parametrize("max_iter", [0, 1])
def test_max_iter(dtype, max_iter):
    diagonal = torch.tensor([1., 2., 3.], dtype=dtype)
    b = torch.ones(3, dtype=dtype)
    result = batched_pcg(lambda v: diagonal*v, b, max_iter=max_iter)
    assert not result.converged and not result.breakdown
    assert result.max_iter_reached and result.iterations == max_iter


@pytest.mark.parametrize("case", ["negative", "zero", "small_positive"])
def test_curvature_breakdown(dtype, case):
    b = torch.ones(2, dtype=dtype)
    diagonal = {"negative": torch.full((2,), -1., dtype=dtype),
                "zero": torch.tensor([1., -1.], dtype=dtype),
                "small_positive": torch.tensor([1., -1.+torch.finfo(dtype).eps], dtype=dtype)}[case]
    result = batched_pcg(lambda v: v*diagonal, b)
    assert result.breakdown and result.nonpositive_curvature
    assert not result.converged and not result.max_iter_reached
    assert result.iterations == 0
    assert torch.equal(result.x, torch.zeros_like(b))


@pytest.mark.parametrize("source", ["matvec", "preconditioner"])
def test_nonfinite_per_item_breakdown(dtype, source):
    b = torch.ones(2, 3, dtype=dtype)
    def invalid(v):
        return torch.stack((v[0], torch.full_like(v[1], float('nan'))))
    result = batched_pcg(invalid if source == "matvec" else lambda v: v, b,
                         M_inv=invalid if source == "preconditioner" else None, mode="fixed", max_iter=5)
    assert torch.equal(result.breakdown, torch.tensor([False, True]))
    assert torch.equal(result.converged, torch.tensor([True, False]))
    assert not result.nonpositive_curvature.any()
    assert torch.isfinite(result.x).all()
    assert result.iterations[1] == 0


def test_negative_preconditioner(dtype):
    result = batched_pcg(lambda v: v, torch.ones(3, dtype=dtype), M_inv=lambda r: -r)
    assert result.breakdown and not result.nonpositive_curvature
    assert result.iterations == 0


def test_breakdown_preserves_last_update(dtype):
    diagonal = torch.tensor([[1., 1.], [1., 2.]], dtype=dtype)
    b = torch.ones(2, 2, dtype=dtype)
    truncated = batched_pcg(lambda v: diagonal*v, b, max_iter=1)
    calls = 0
    def failing(v):
        nonlocal calls
        calls += 1
        value = diagonal*v
        if calls == 3:
            return torch.stack((value[0], torch.full_like(value[1], float('nan'))))
        return value
    result = batched_pcg(failing, b, mode="fixed", max_iter=4)
    assert torch.equal(result.breakdown, torch.tensor([False, True]))
    assert result.iterations[1] == 1
    assert torch.equal(result.x[1], truncated.x[1])


def test_preconditioner_nonfinite_not_hidden_by_projection(dtype):
    projector = lambda v: torch.stack((v[..., 0], torch.zeros_like(v[..., 1])), dim=-1)
    def invalid(r):
        return torch.stack((r[..., 0], torch.full_like(r[..., 1], float('nan'))), dim=-1)
    result = batched_pcg(lambda v: v, torch.tensor([1., 0.], dtype=dtype),
                         M_inv=invalid, projector=projector)
    assert result.breakdown and result.iterations == 0


def test_callable_diagonal_preconditioner(dtype):
    diagonal = torch.tensor([1., 2., 8., 100.], dtype=dtype)
    b = torch.ones(2, 4, dtype=dtype)
    result = batched_pcg(lambda v: diagonal*v, b, M_inv=lambda r: r/diagonal)
    assert_result(result, b)
    close(result.x, b/diagonal, dtype)
    assert (result.iterations == 1).all()


def test_update_overflow():
    b = torch.full((2,), 1e10, dtype=torch.float32)
    result = batched_pcg(lambda v: 1e-30*v, b)
    assert result.breakdown and not result.converged
    assert torch.isfinite(result.x).all()


def test_nonfinite_initial_norm():
    b = torch.full((2,), torch.finfo(torch.float32).max)
    result = batched_pcg(lambda v: v, b)
    assert result.breakdown and not result.converged


def test_projected_system_reduced_reference(dtype):
    rng = torch.Generator().manual_seed(20261023)
    R = torch.randn(32, 6, dtype=dtype, generator=rng)
    oracle = MVSKOracle(prepare_mvsk_data(R), [0., 1., 0., 0.])
    x = torch.randn(6, dtype=dtype, generator=rng)
    geom = build_geometry(oracle.gradient(x), torch.tensor([True, True, True, True, True, False]))
    operator = ProjectedHessianOperator(oracle, x, geom, damping=.5)
    project = lambda v: project_level_tangent(v, geom)
    b = project(torch.randn(6, dtype=dtype, generator=rng))
    result = batched_pcg(operator.matvec_unchecked, b, projector=project, max_iter=30,
                         x0=torch.ones(6, dtype=dtype))
    assert_result(result, b)
    # Independent reduced subspace construction via a dense face projector.
    m = geom.free_mask.to(dtype)
    F = torch.diag(m)-torch.outer(m, m)/m.sum()
    gf = F@oracle.gradient(x)
    nu = gf/gf.norm()
    P = F-torch.outer(nu, nu)
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    W = eigenvectors[:, eigenvalues > .5]
    H = 2*oracle.A.T@oracle.A/oracle.T
    reduced = W.T@H@W+.5*torch.eye(W.shape[1], dtype=dtype)
    expected = W@torch.linalg.solve(reduced, W.T@b)
    close(result.x, expected, dtype)
    close(project(result.x), result.x, dtype)
    assert result.x[-1] == 0
    close(result.x.sum(), torch.zeros((), dtype=dtype), dtype)
    close(geom.nu@result.x, torch.zeros((), dtype=dtype), dtype)


def test_illegal_projected_rhs(dtype):
    projector = lambda v: v-v.mean(-1, keepdim=True)
    with pytest.raises(ValueError, match="subspace"):
        batched_pcg(lambda v: v, torch.ones(3, dtype=dtype), projector=projector)


def test_adaptive_fixed_agreement(dtype):
    A, b = random_spd(dtype)
    adaptive = batched_pcg(matvec(A), b)
    fixed = batched_pcg(matvec(A), b, mode="fixed")
    close(fixed.x, adaptive.x, dtype)
    close(fixed.residual_norm, adaptive.residual_norm, dtype)
    for name in ("converged", "iterations", "breakdown", "nonpositive_curvature", "max_iter_reached"):
        assert torch.equal(getattr(fixed, name), getattr(adaptive, name))


def test_leading_batch_and_single(dtype):
    A, b = random_spd(dtype)
    b = b.expand(2, 3, 6)
    result = batched_pcg(matvec(A), b)
    assert_result(result, b)
    for i in range(3):
        single = batched_pcg(matvec(A[i]), b[0, i])
        close(result.x[0, i], single.x, dtype)


def test_final_true_residual_gap(dtype):
    # Deliberately stateful callback injects an end-of-solve residual gap.
    calls = 0
    def changing(v):
        nonlocal calls
        calls += 1
        return (2 if calls == 3 else 1)*v
    b = torch.ones(3, dtype=dtype)
    result = batched_pcg(changing, b, max_iter=10)
    assert result.iterations == 1
    assert not result.converged and not result.breakdown and not result.max_iter_reached
    close(result.residual_norm, b.norm(), dtype)
    close(result.relative_residual, torch.ones((), dtype=dtype), dtype)


def test_final_nonfinite_marks_breakdown(dtype):
    calls = 0
    def changing(v):
        nonlocal calls
        calls += 1
        return torch.full_like(v, float('inf')) if calls == 3 else v
    result = batched_pcg(changing, torch.ones(3, dtype=dtype))
    assert result.breakdown and not result.converged
    assert torch.isinf(result.residual_norm)
    assert torch.isfinite(result.x).all()


@pytest.mark.parametrize("case", ["dtype", "nonfinite", "shape", "x0_n", "x0_dtype",
                                  "rtol", "atol", "iterations", "mode", "callback_shape", "callback_dtype"])
def test_validation(case):
    b = torch.ones(3)
    kwargs = {}
    apply = lambda v: v
    if case == "dtype":
        b = torch.ones(3, dtype=torch.int64)
    elif case == "nonfinite":
        b[0] = float('nan')
    elif case == "shape":
        b = torch.tensor(1.)
    elif case == "x0_n":
        kwargs['x0'] = torch.ones(1)
    elif case == "x0_dtype":
        kwargs['x0'] = torch.ones(3, dtype=torch.float64)
    elif case == "rtol":
        kwargs['rtol'] = -1.
    elif case == "atol":
        kwargs['atol'] = float('inf')
    elif case == "iterations":
        kwargs['max_iter'] = -1
    elif case == "mode":
        kwargs['mode'] = "other"
    elif case == "callback_shape":
        apply = lambda v: v[:1]
    elif case == "callback_dtype":
        apply = lambda v: v.double()
    with pytest.raises((TypeError, ValueError)):
        batched_pcg(apply, b, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    A, b = random_spd(dtype)
    cpu = batched_pcg(matvec(A), b)
    gpu = batched_pcg(matvec(A.cuda()), b.cuda())
    assert_result(cpu, b)
    assert_result(gpu, b.cuda())
    close(gpu.x.cpu(), cpu.x, dtype)
    assert torch.equal(gpu.converged.cpu(), cpu.converged)
    with pytest.raises(ValueError):
        batched_pcg(matvec(A.cuda()), b.cuda(), x0=b)
