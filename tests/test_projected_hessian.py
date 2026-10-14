"""Small dense references exist only in tests."""
import pytest
import torch
from torchyand import MVSKOracle, prepare_mvsk_data
from torchyand.geometry import build_geometry, project_level_tangent
from torchyand.operators import ProjectedHessianOperator, apply_projected_hessian


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def problem(dtype):
    rng = torch.Generator().manual_seed(20261021)
    R = torch.randn(24, 6, dtype=dtype, generator=rng)
    oracle = MVSKOracle(prepare_mvsk_data(R), [1., 1., .03, .02])
    x, u, v = [torch.randn(2, 3, 6, dtype=dtype, generator=rng)*.1 for _ in range(3)]
    mask = torch.tensor([True, False, True, True, True, False])
    geom = build_geometry(oracle.gradient(x), mask)
    return oracle, x, u, v, geom


def dense(oracle, x, geom, gradient=None):
    # Independently assemble the Hessian and face/level matrices for small n.
    z = x@oracle.A.T
    _, c2, c3, c4 = oracle.coefficients
    w = 2*c2-6*c3*z+12*c4*z.square()
    H = torch.einsum('...t,ti,tj->...ij', w, oracle.A, oracle.A)/oracle.T
    m = geom.free_mask.to(x.dtype)
    count = m.sum(-1).clamp_min(1)
    F = torch.diag_embed(m)-m.unsqueeze(-1)*m.unsqueeze(-2)/count[..., None, None]
    gradient = oracle.gradient(x) if gradient is None else gradient
    gf = (F@gradient.unsqueeze(-1)).squeeze(-1)
    norm = torch.linalg.vector_norm(gf, dim=-1)
    regular = norm > geom.grad_threshold
    nu = gf/torch.where(regular, norm, 1).unsqueeze(-1)
    nu = torch.where(regular.unsqueeze(-1), nu, 0)
    P = F-nu.unsqueeze(-1)*nu.unsqueeze(-2)
    return H, P


def close(a, b, dtype):
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(a, b, rtol=512*eps, atol=512*eps)


@pytest.mark.parametrize("damping", [0., .4])
def test_dense_reference(dtype, damping):
    oracle, x, _, v, geom = problem(dtype)
    H, P = dense(oracle, x, geom)
    K = P@H@P+damping*P
    actual = apply_projected_hessian(oracle, x, v, geom, damping=damping)
    expected = (K@v.unsqueeze(-1)).squeeze(-1)
    close(actual, expected, dtype)
    assert actual.shape == v.shape


def test_symmetry_and_tangent(dtype):
    oracle, x, u, v, geom = problem(dtype)
    operator = ProjectedHessianOperator(oracle, x, geom, damping=.3)
    ku, kv = operator.matvec(u), operator.matvec(v)
    close((u*kv).sum(-1), (ku*v).sum(-1), dtype)
    close(project_level_tangent(kv, geom), kv, dtype)
    close(kv.sum(-1), torch.zeros_like(kv[..., 0]), dtype)
    close((geom.nu*kv).sum(-1), torch.zeros_like(kv[..., 0]), dtype)
    assert torch.equal(torch.where(geom.free_mask, 0, kv), torch.zeros_like(kv))
    close(operator.matvec_unchecked(v), kv, dtype)


def test_degenerate_geometry(dtype):
    oracle, x, _, v, _ = problem(dtype)
    geom = build_geometry(torch.zeros_like(x))
    assert not geom.level_regular.any()
    H, P = dense(oracle, x, geom, gradient=torch.zeros_like(x))
    K = P@H@P+.2*P
    close(apply_projected_hessian(oracle, x, v, geom, damping=.2),
          (K@v.unsqueeze(-1)).squeeze(-1), dtype)


@pytest.mark.parametrize("count,regular", [(0, False), (1, False), (2, True)])
def test_zero_tangent_dimension(dtype, count, regular):
    oracle, x, _, v, _ = problem(dtype)
    mask = torch.arange(6) < count
    g = torch.tensor([1., -1., 0., 0., 0., 0.], dtype=dtype)
    geom = build_geometry(g, mask)
    assert geom.level_regular.item() == regular
    result = apply_projected_hessian(oracle, x, v, geom, damping=1.)
    assert torch.equal(result, torch.zeros_like(result))


def test_damping_broadcast_and_cache(dtype):
    oracle, x, _, v, geom = problem(dtype)
    damping = torch.tensor([.1, .2, .3], dtype=dtype)
    cached = oracle.cache(x)
    close(apply_projected_hessian(oracle, x, v, geom, cache=cached, damping=damping),
          apply_projected_hessian(oracle, x, v, geom, damping=damping.expand(2, 3)), dtype)
    base = apply_projected_hessian(oracle, x, v, geom)
    close(apply_projected_hessian(oracle, x, v, geom, damping=damping)-base,
          damping.unsqueeze(-1)*project_level_tangent(v, geom), dtype)


def test_batch_vs_single(dtype):
    oracle, x, _, v, geom = problem(dtype)
    batch = apply_projected_hessian(oracle, x, v, geom, damping=.5)
    for i in range(2):
        for j in range(3):
            single_geom = build_geometry(oracle.gradient(x[i, j]), geom.free_mask[i, j])
            close(apply_projected_hessian(oracle, x[i, j], v[i, j], single_geom, damping=.5),
                  batch[i, j], dtype)


def test_leading_broadcast(dtype):
    oracle, x, _, v, _ = problem(dtype)
    shared_x = x[0, 0]
    geom = build_geometry(oracle.gradient(shared_x))
    batch = apply_projected_hessian(oracle, shared_x, v, geom)
    assert batch.shape == (2, 3, 6)
    close(batch[1, 2], apply_projected_hessian(oracle, shared_x, v[1, 2], geom), dtype)
    geom_batch = build_geometry(oracle.gradient(x))
    assert apply_projected_hessian(oracle, x, v[0, 0], geom_batch).shape == (2, 3, 6)


@pytest.mark.parametrize("bad", ["negative", "nan", "inf", "dtype", "shape", "v_dim", "cache"])
def test_validation(bad):
    oracle, x, _, v, geom = problem(torch.float32)
    damping = {"negative": -1., "nan": float('nan'), "inf": float('inf'),
               "dtype": torch.tensor(1., dtype=torch.float64), "shape": torch.ones(4)}.get(bad, 0.)
    cache = oracle.cache(x.clone()) if bad == "cache" else None
    if bad == "v_dim":
        v = torch.ones(2, 3, 1)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        apply_projected_hessian(oracle, x, v, geom, damping=damping, cache=cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    from torchyand import MVSKData
    oracle, x, _, v, geom = problem(dtype)
    gpu = MVSKOracle(MVSKData(oracle.A.cuda(), oracle.mu.cuda(), oracle.T, oracle.n), oracle.coefficients.cuda())
    gpu_geom = build_geometry(gpu.gradient(x.cuda()), geom.free_mask.cuda())
    close(apply_projected_hessian(gpu, x.cuda(), v.cuda(), gpu_geom, damping=.3).cpu(),
          apply_projected_hessian(oracle, x, v, geom, damping=.3), dtype)
    with pytest.raises(ValueError):
        apply_projected_hessian(gpu, x.cuda(), v.cuda(), gpu_geom, damping=torch.tensor(.3, dtype=dtype))
