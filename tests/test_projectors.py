"""Independent dense references and matrix-free projector properties."""
from dataclasses import replace
import pytest
import torch

from torchyand.geometry import (
    build_geometry, project_face_tangent, project_level_tangent,
    project_simplex_tangent,
)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def close(a, b, dtype):
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(a, b, rtol=256*eps, atol=256*eps)


def inputs(dtype):
    rng = torch.Generator().manual_seed(20261020)
    v, u, g = [torch.randn(2, 3, 7, dtype=dtype, generator=rng) for _ in range(3)]
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 0, 1, 1, 0, 1]], dtype=torch.bool)
    return v, u, g, mask


def test_simplex_closed_form(dtype):
    v = torch.tensor([1., 2., 6.], dtype=dtype)
    close(project_simplex_tangent(v), torch.tensor([-2., -1., 3.], dtype=dtype), dtype)


def test_face_closed_form(dtype):
    v = torch.tensor([1., 200., 5., 3.], dtype=dtype)
    mask = torch.tensor([True, False, True, True])
    close(project_face_tangent(v, mask), torch.tensor([-2., 0., 2., 0.], dtype=dtype), dtype)


def test_level_closed_form(dtype):
    g = torch.tensor([1., 999., -1., 0.], dtype=dtype)
    v = torch.tensor([4., 200., 2., 0.], dtype=dtype)
    mask = torch.tensor([True, False, True, True])
    geom = build_geometry(g, mask)
    assert geom.level_regular.item()
    close(project_level_tangent(v, geom), torch.tensor([1., 0., 1., -2.], dtype=dtype), dtype)


@pytest.mark.parametrize("kind", ["simplex", "face", "level"])
def test_projector_properties(dtype, kind):
    v, u, g, mask = inputs(dtype)
    geom = build_geometry(g, mask)
    project = {"simplex": project_simplex_tangent,
               "face": lambda x: project_face_tangent(x, mask),
               "level": lambda x: project_level_tangent(x, geom)}[kind]
    pv, pu = project(v), project(u)
    close(project(pv), pv, dtype)
    close((u*pv).sum(-1), (pu*v).sum(-1), dtype)
    close(pv.sum(-1), torch.zeros_like(pv[..., 0]), dtype)
    if kind != "simplex":
        assert torch.equal(torch.where(mask, 0, pv), torch.zeros_like(pv))
    if kind == "level":
        assert geom.level_regular.all()
        close((geom.nu*pv).sum(-1), torch.zeros_like(geom.grad_norm), dtype)
        close((geom.grad_face*pv).sum(-1), torch.zeros_like(geom.grad_norm), dtype)


def test_all_free(dtype):
    v, _, _, _ = inputs(dtype)
    close(project_face_tangent(v, torch.ones(7, dtype=torch.bool)), project_simplex_tangent(v), dtype)


@pytest.mark.parametrize("count", [0, 1])
def test_degenerate_face(dtype, count):
    v, _, g, _ = inputs(dtype)
    mask = torch.arange(7) < count
    assert torch.equal(project_face_tangent(v, mask), torch.zeros_like(v))
    geom = build_geometry(g, mask)
    assert (geom.free_count == count).all()
    assert not geom.level_regular.any()
    assert torch.equal(geom.nu, torch.zeros_like(g))
    assert torch.equal(project_level_tangent(v, geom), torch.zeros_like(v))


def test_n_one(dtype):
    v = torch.tensor([[3.], [-5.]], dtype=dtype)
    assert torch.equal(project_simplex_tangent(v), torch.zeros_like(v))
    geom = build_geometry(v)
    assert not geom.level_regular.any()
    assert torch.equal(project_level_tangent(v, geom), torch.zeros_like(v))


def test_zero_gradient(dtype):
    v, _, g, mask = inputs(dtype)
    geom = build_geometry(torch.zeros_like(g), mask)
    assert not geom.level_regular.any()
    assert torch.equal(geom.nu, torch.zeros_like(g))
    assert torch.equal(project_level_tangent(v, geom), project_face_tangent(v, mask))


def test_threshold_formula(dtype):
    _, _, g, mask = inputs(dtype)
    geom = build_geometry(g, mask)
    count = mask.sum(-1).expand(2, 3)
    expected = (128*torch.finfo(dtype).eps*count.clamp_min(1).to(dtype).sqrt()
                * torch.linalg.vector_norm(torch.where(mask, g, 0), dim=-1).clamp_min(1))
    torch.testing.assert_close(geom.grad_threshold, expected, rtol=0, atol=0)
    assert torch.equal(geom.free_count, count)


@pytest.mark.parametrize("explicit", [False, True])
def test_threshold_sides(dtype, explicit):
    normal = torch.tensor([1., -1., 0., 0.], dtype=dtype)
    normal = normal/normal.norm()
    threshold = .01 if explicit else 128*torch.finfo(dtype).eps*2
    g = torch.tensor([.5, 2.], dtype=dtype).unsqueeze(-1)*threshold*normal
    geom = build_geometry(g, grad_tol=threshold if explicit else None)
    assert torch.equal(geom.level_regular, torch.tensor([False, True]))
    close(geom.grad_threshold, torch.full((2,), threshold, dtype=dtype), dtype)
    assert torch.equal(geom.nu[0], torch.zeros_like(normal))
    close(geom.nu[1], normal, dtype)
    v = torch.tensor([3., 2., -1., 8.], dtype=dtype)
    assert torch.equal(project_level_tangent(v, geom)[0], project_simplex_tangent(v))


def test_explicit_absolute_tolerance(dtype):
    g = torch.tensor([[1., -1., 100., 100.], [1e3, -1e3, 1e5, 1e5]], dtype=dtype)
    mask = torch.tensor([True, True, False, False])
    tol = torch.tensor([2., 10.], dtype=dtype)
    geom = build_geometry(g, mask, grad_tol=tol)
    assert torch.equal(geom.grad_threshold, tol)
    assert torch.equal(geom.level_regular, torch.tensor([False, True]))


def test_batch_vs_single(dtype):
    v, _, g, mask = inputs(dtype)
    geom = build_geometry(g, mask)
    for i in range(2):
        for j in range(3):
            single = build_geometry(g[i, j], mask[j])
            close(single.grad_face, geom.grad_face[i, j], dtype)
            close(single.nu, geom.nu[i, j], dtype)
            assert torch.equal(single.level_regular, geom.level_regular[i, j])
            close(project_simplex_tangent(v[i, j]), project_simplex_tangent(v)[i, j], dtype)
            close(project_face_tangent(v[i, j], mask[j]), project_face_tangent(v, mask)[i, j], dtype)
            close(project_level_tangent(v[i, j], single), project_level_tangent(v, geom)[i, j], dtype)


def test_mask_broadcasting(dtype):
    v, _, g, mask = inputs(dtype)
    expanded = mask.expand_as(v)
    close(project_face_tangent(v, mask), project_face_tangent(v, expanded), dtype)
    assert build_geometry(g[0, 0], mask).grad_face.shape == (3, 7)
    close(project_face_tangent(v[0, 0], mask), project_face_tangent(v[0, 0].expand(3, 7), mask), dtype)


def test_geometry_broadcasting(dtype):
    v, _, g, mask = inputs(dtype)
    geom = build_geometry(g[0], mask)
    close(project_level_tangent(v, geom), project_level_tangent(v, build_geometry(g[0].expand_as(g), mask)), dtype)
    assert project_level_tangent(v[0, 0], build_geometry(g, mask)).shape == (2, 3, 7)


def dense_face(mask, dtype):
    # Test-only independent matrix construction, never imported by package.
    m = mask.to(dtype)
    return torch.diag(m)-torch.outer(m, m)/max(int(mask.sum()), 1)


def test_independent_dense_reference(dtype):
    v, _, g, mask = inputs(dtype)
    simplex = torch.eye(7, dtype=dtype)-torch.ones(7, 7, dtype=dtype)/7
    close(project_simplex_tangent(v), v@simplex.T, dtype)
    masks = torch.cat((mask, torch.zeros(1, 7, dtype=torch.bool),
                       (torch.arange(7) == 0).unsqueeze(0)))
    for item in masks:
        face = dense_face(item, dtype)
        for i in range(2):
            vv, gg = v[i, 0], g[i, 0]
            gf = face@gg
            threshold = (128*torch.finfo(dtype).eps*max(int(item.sum()), 1)**.5
                         * max(1., torch.linalg.vector_norm(torch.where(item, gg, 0)).item()))
            nu = gf/gf.norm() if gf.norm().item() > threshold else torch.zeros_like(gf)
            level = face-torch.outer(nu, nu)
            geom = build_geometry(gg, item)
            close(geom.grad_face, gf, dtype)
            close(project_face_tangent(vv, item), face@vv, dtype)
            close(project_level_tangent(vv, geom), level@vv, dtype)


@pytest.mark.parametrize("bad", ["integer", "half", "scalar", "empty", "nan", "inf"])
def test_invalid_vector(bad):
    values = {"integer": torch.ones(3, dtype=torch.int64), "half": torch.ones(3, dtype=torch.float16),
              "scalar": torch.tensor(1.), "empty": torch.empty(0),
              "nan": torch.tensor([float("nan")]), "inf": torch.tensor([float("inf")])}
    for call in (project_simplex_tangent, build_geometry):
        with pytest.raises((TypeError, ValueError)):
            call(values[bad])


@pytest.mark.parametrize("bad", ["dtype", "scalar", "last_dim", "batch"])
def test_invalid_mask(bad):
    v = torch.ones(2, 3)
    masks = {"dtype": torch.ones(3), "scalar": torch.tensor(True),
             "last_dim": torch.ones(1, dtype=torch.bool), "batch": torch.ones(4, 3, dtype=torch.bool)}
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        project_face_tangent(v, masks[bad])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        build_geometry(v, masks[bad])


@pytest.mark.parametrize("tol", [-1., float("nan"), float("inf"), torch.ones(3), torch.tensor(1., dtype=torch.float64)])
def test_invalid_tolerance(tol):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        build_geometry(torch.ones(2, 4), grad_tol=tol)


@pytest.mark.parametrize("bad", ["last_dim", "dtype", "nonfinite", "metadata", "type"])
def test_invalid_geometry(bad):
    geom = build_geometry(torch.tensor([1., -1., 0.]))
    cases = {"last_dim": (torch.ones(1), geom),
             "dtype": (torch.ones(3, dtype=torch.float64), geom),
             "nonfinite": (torch.ones(3), replace(geom, nu=torch.full((3,), float("nan")))),
             "metadata": (torch.ones(3), replace(geom, free_count=torch.ones(2, dtype=torch.int64))),
             "type": (torch.ones(3), None)}
    with pytest.raises((TypeError, ValueError)):
        project_level_tangent(*cases[bad])


def test_finite_input_overflow(dtype):
    g = torch.tensor([torch.finfo(dtype).max, -torch.finfo(dtype).max], dtype=dtype)
    with pytest.raises(FloatingPointError):
        build_geometry(g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    v, _, g, mask = inputs(dtype)
    cpu = build_geometry(g, mask)
    gpu = build_geometry(g.cuda(), mask.cuda())
    for name in ("grad_face", "grad_norm", "nu", "grad_threshold"):
        close(getattr(gpu, name).cpu(), getattr(cpu, name), dtype)
    assert torch.equal(gpu.level_regular.cpu(), cpu.level_regular)
    close(project_simplex_tangent(v.cuda()).cpu(), project_simplex_tangent(v), dtype)
    close(project_face_tangent(v.cuda(), mask.cuda()).cpu(), project_face_tangent(v, mask), dtype)
    close(project_level_tangent(v.cuda(), gpu).cpu(), project_level_tangent(v, cpu), dtype)
    with pytest.raises(ValueError):
        project_face_tangent(v.cuda(), mask)
    with pytest.raises(ValueError):
        project_level_tangent(v.cuda(), cpu)
    with pytest.raises(ValueError):
        build_geometry(g.cuda(), grad_tol=torch.tensor(1., dtype=dtype))
