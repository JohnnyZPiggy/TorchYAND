"""Seeded batched correctness checks; no timings or variable scaling."""
import pytest
import torch

from torchyand.polynomial import cubic_real_roots, quadratic_real_roots

BATCH = 1000


def coefficients_from_roots(roots):
    r1, r2, r3 = roots.unbind(-1)
    return torch.stack((torch.ones_like(r1), -(r1 + r2 + r3),
                        r1*r2 + r1*r3 + r2*r3, -r1*r2*r3), -1)


def assert_contract(result, dtype):
    assert result.roots.shape == result.valid.shape == result.residual.shape
    assert result.roots.shape[-1] == 3
    assert torch.isfinite(result.roots[result.valid]).all()
    assert torch.isnan(result.roots[~result.valid]).all()
    assert torch.isnan(result.residual[~result.valid]).all()
    comparable = result.valid[..., 1:] & result.valid[..., :-1]
    assert ((result.roots[..., 1:] >= result.roots[..., :-1]) | ~comparable).all()
    assert (result.residual[result.valid] <= 2048 * torch.finfo(dtype).eps).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("root_scale", [1e-4, 1., 1e4])
@pytest.mark.parametrize("kind", ["distinct", "double", "triple", "complex_pair"])
def test_randomized(dtype, root_scale, kind):
    generator = torch.Generator().manual_seed(20261007)
    random = torch.rand((BATCH, 3), generator=generator, dtype=dtype)
    # Separated intervals give well-conditioned distinct roots. Repeated roots
    # are tested separately with conditioning-appropriate comparison bounds.
    roots = (random * .2 + torch.tensor([-1., .3, 1.3], dtype=dtype)) * root_scale
    if kind == "double":
        roots[:, 2] = roots[:, 1]
    elif kind == "triple":
        roots[:] = roots[:, :1].clone()
    if kind == "complex_pair":
        r = roots[:, 0]
        u = roots[:, 1]
        v = (.5 + random[:, 2]) * root_scale
        coeff = torch.stack((torch.ones_like(r), -(r + 2*u),
                             2*r*u + u*u + v*v, -r*(u*u + v*v)), -1)
        expected = torch.full_like(roots, float("nan"))
        expected[:, 0] = r
    else:
        coeff = coefficients_from_roots(roots)
        expected = roots.sort(-1).values
    # Randomly assign the requested five coefficient exponents.
    exponents = torch.tensor([-20., -10., 0., 10., 20.], dtype=dtype)
    exponent_indices = torch.randint(5, (BATCH,), generator=generator)
    coeff *= torch.pow(torch.tensor(10., dtype=dtype), exponents[exponent_indices, None])
    # The public default preserves algebraic degree at every root scale.
    analytic = cubic_real_roots(*coeff.unbind(-1))
    # For genuine complex pairs use a strict imaginary tolerance: the default
    # heuristic for repeated roots can admit tiny complex roots around 1e-4.
    imag_tol = 128 * torch.finfo(dtype).eps if kind == "complex_pair" else None
    reference = cubic_real_roots(*coeff.unbind(-1), method="companion",
                                 imag_tol=imag_tol)
    assert_contract(analytic, dtype)
    assert_contract(reference, dtype)
    assert (analytic.degree == 3).all()
    assert torch.equal(analytic.valid, torch.isfinite(expected))
    assert torch.equal(reference.valid, analytic.valid)
    # Cubic roots at multiplicity m are sensitive to coefficient error^(1/m).
    eps = torch.finfo(dtype).eps
    analytic_tol = (512 * eps if kind != "double" else 32 * eps**.5)
    reference_tol = 32 * eps**(1/3) if kind == "triple" else (
        32 * eps**.5 if kind == "double" else 512 * eps)
    torch.testing.assert_close(analytic.roots, expected, equal_nan=True,
                               rtol=analytic_tol, atol=analytic_tol * root_scale)
    torch.testing.assert_close(reference.roots, expected, equal_nan=True,
                               rtol=reference_tol, atol=reference_tol * root_scale)
    torch.testing.assert_close(analytic.roots, reference.roots, equal_nan=True,
                               rtol=reference_tol, atol=reference_tol * root_scale)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_degree_resolution(dtype):
    expected = torch.tensor([[1e4, 2e4, 3e4]], dtype=dtype)
    coeff = coefficients_from_roots(expected)
    for method in ["analytic", "companion"]:
        result = cubic_real_roots(*coeff.unbind(-1), method=method)
        assert result.degree.item() == 3
        assert result.valid.all()
        torch.testing.assert_close(result.roots, expected,
                                   rtol=512 * torch.finfo(dtype).eps, atol=0.)


@pytest.mark.parametrize("method", ["analytic", "companion"])
def test_explicit_cubic_degree_degeneration(method):
    coeff = coefficients_from_roots(torch.tensor([[1e4, 2e4, 3e4]], dtype=torch.float32))
    # a/s ~ 1.67e-13 < tol < |b|/s ~ 1e-8: intentionally quadratic.
    result = cubic_real_roots(*coeff.unbind(-1), method=method, degree_tol=1e-10)
    expected = quadratic_real_roots(*coeff[:, 1:].unbind(-1), degree_tol=1e-10)
    assert result.degree.item() == 2
    torch.testing.assert_close(result.roots, expected.roots, equal_nan=True)
    assert torch.equal(result.valid, expected.valid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_companion_degenerate_batch(dtype, monkeypatch):
    coeff = torch.tensor([[0., 1., -3., 2.], [0., 0., 2., -6.],
                          [0., 0., 0., 2.], [0., 0., 0., 0.]], dtype=dtype)
    def forbidden(*args, **kwargs):
        raise AssertionError("Degenerate polynomials must not reach eigvals")
    monkeypatch.setattr(torch.linalg, "eigvals", forbidden)
    actual = cubic_real_roots(*coeff.unbind(-1), method="companion")
    expected = quadratic_real_roots(*coeff[:, 1:].unbind(-1))
    torch.testing.assert_close(actual.roots, expected.roots, equal_nan=True)
    assert torch.equal(actual.degree, expected.degree)


def test_near_multiple_resolution(record_property):
    gaps = torch.tensor([1e-4, 1e-6, 1e-8, 1e-10, 1e-12], dtype=torch.float64)
    roots = torch.stack((torch.ones_like(gaps), 1 + gaps, 2 * torch.ones_like(gaps)), -1)
    coeff = coefficients_from_roots(roots)
    # Raw roots show branch merging without a later Newton update.
    analytic = cubic_real_roots(*coeff.unbind(-1), polish_steps=0)
    reference = cubic_real_roots(*coeff.unbind(-1), method="companion", polish_steps=0)
    assert_contract(analytic, torch.float64)
    assert_contract(reference, torch.float64)
    merged = analytic.valid.all(-1) & (analytic.roots[:, 0] == analytic.roots[:, 1])
    observations = {
        "gaps": gaps.tolist(), "analytic_merged": merged.tolist(),
        "analytic_max_residual": analytic.residual.nan_to_num().amax(-1).tolist(),
        "companion_counts": reference.valid.sum(-1).tolist(),
        "companion_max_root_error": (reference.roots-roots).abs().nan_to_num().amax(-1).tolist(),
    }
    record_property("near_multiple_resolution", str(observations))
    print("near-multiple observations:", observations)
    assert not merged[0]
    assert merged[-1]
