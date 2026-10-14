import pytest
import torch

from torchyand.polynomial import cubic_real_roots, quadratic_real_roots


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


CASES = [
    ([1., -6., 11., -6.], [1., 2., 3.]),
    ([1., -3., 3., -1.], [1., 1., 1.]),
    ([1., 0., -3., 2.], [-2., 1., 1.]),
    ([1., 0., 1., 1.], [-0.6823278038280193, float("nan"), float("nan")]),
    ([1., 0., 0., -1.], [1., float("nan"), float("nan")]),
    ([1., 0., 0., 0.], [0., 0., 0.]),
]


def check(result, expected, dtype):
    expected = torch.tensor(expected, dtype=dtype, device=result.roots.device)
    torch.testing.assert_close(result.roots, expected, equal_nan=True,
                               atol=256 * torch.finfo(dtype).eps, rtol=256 * torch.finfo(dtype).eps)
    assert torch.equal(result.valid, torch.isfinite(expected))
    assert torch.isfinite(result.roots[result.valid]).all()
    assert torch.isnan(result.roots[~result.valid]).all()
    assert torch.isnan(result.residual[~result.valid]).all()
    assert (result.residual[result.valid] <= 1024 * torch.finfo(dtype).eps).all()
    assert result.roots.dtype == dtype
    assert result.valid.dtype == torch.bool
    assert result.degree.dtype == torch.int64


@pytest.mark.parametrize("coeff,expected", CASES)
@pytest.mark.parametrize("scale", [1e-20, 1., 1e20])
def test_known_roots(dtype, coeff, expected, scale):
    coefficients = torch.tensor(coeff, dtype=dtype) * scale
    result = cubic_real_roots(*coefficients.unbind())
    check(result, expected, dtype)
    assert result.degree.item() == 3


@pytest.mark.parametrize("coeff", [[0., 1., -3., 2.], [0., 0., 2., -6.],
                                 [0., 0., 0., 2.], [0., 0., 0., 0.]])
def test_degree_fallback(dtype, coeff):
    a, b, c, d = torch.tensor(coeff, dtype=dtype).unbind()
    result = cubic_real_roots(a, b, c, d)
    reference = quadratic_real_roots(b, c, d)
    torch.testing.assert_close(result.roots, reference.roots, equal_nan=True)
    assert torch.equal(result.valid, reference.valid)
    assert torch.equal(result.degree, reference.degree)


def mixed_coefficients(dtype, device="cpu"):
    return torch.tensor([CASES[0][0], CASES[3][0], CASES[2][0], [0., 1., -3., 2.]],
                        dtype=dtype, device=device)


def test_mixed_batch(dtype):
    result = cubic_real_roots(*mixed_coefficients(dtype).unbind(-1))
    check(result, [CASES[0][1], CASES[3][1], CASES[2][1], [1., 2., float("nan")]], dtype)
    assert result.degree.tolist() == [3, 3, 3, 2]
    assert result.roots.shape == result.valid.shape == result.residual.shape == (4, 3)
    assert result.roots.device.type == "cpu"


def test_broadcast(dtype):
    coeff = mixed_coefficients(dtype)
    result = cubic_real_roots(coeff[:, 0].reshape(1, 4, 1),
                              coeff[:, 1].reshape(1, 4, 1),
                              coeff[:, 2].reshape(1, 4, 1),
                              coeff[:, 3].reshape(1, 4, 1) + torch.zeros((2, 1, 3), dtype=dtype))
    assert result.roots.shape == (2, 4, 3, 3)
    reference = cubic_real_roots(*coeff.unbind(-1))
    torch.testing.assert_close(result.roots, reference.roots[None, :, None, :].expand(2, 4, 3, 3), equal_nan=True)


def test_polishing_does_not_worsen(dtype):
    args = mixed_coefficients(dtype).unbind(-1)
    raw = cubic_real_roots(*args, polish_steps=0)
    polished = cubic_real_roots(*args, polish_steps=2)
    assert (polished.residual[polished.valid] <= raw.residual[raw.valid]).all()


def test_near_repeated_distinct(dtype):
    # Centered roots avoid the shifted-triple cancellation ambiguity.
    spacing = 1e-3
    args = torch.tensor([1., 0., -spacing * spacing, 0.], dtype=dtype).unbind()
    check(cubic_real_roots(*args), [-spacing, 0., spacing], dtype)


def test_discriminant_tolerance(dtype):
    args = torch.tensor([1., 0., -3., 2.0001], dtype=dtype).unbind()
    strict = cubic_real_roots(*args, discriminant_tol=0., polish_steps=0)
    relaxed = cubic_real_roots(*args, discriminant_tol=1e-3, polish_steps=0)
    assert strict.valid.sum() == 1
    assert relaxed.valid.sum() == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(dtype, bad):
    with pytest.raises(ValueError):
        cubic_real_roots(*torch.tensor([1., 0., bad, 0.], dtype=dtype).unbind())


def test_mixed_dtype():
    args = [torch.tensor(v, dtype=torch.float32) for v in CASES[0][0]]
    args[-1] = args[-1].double()
    check(cubic_real_roots(*args), CASES[0][1], torch.float64)


@pytest.mark.parametrize("steps", [-1, 1.5])
def test_invalid_polish_steps(steps):
    with pytest.raises(ValueError):
        cubic_real_roots(*torch.tensor(CASES[0][0]).unbind(), polish_steps=steps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu(dtype):
    cpu = cubic_real_roots(*mixed_coefficients(dtype).unbind(-1))
    gpu = cubic_real_roots(*mixed_coefficients(dtype, "cuda").unbind(-1))
    assert gpu.roots.device.type == "cuda"
    torch.testing.assert_close(gpu.roots.cpu(), cpu.roots, equal_nan=True)
    assert torch.equal(gpu.valid.cpu(), cpu.valid)
