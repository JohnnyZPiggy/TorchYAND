import pytest
import torch

from torchyand.polynomial import RootResult, linear_roots, quadratic_real_roots, cubic_real_roots
from torchyand.polynomial.common import validate_coefficients, polish_roots, normalized_residual


@pytest.fixture(params=[torch.float32, torch.float64])
def tensor(request):
    return lambda value: torch.tensor(value, dtype=request.param)


def assert_result(result, roots, degree):
    assert isinstance(result, RootResult)
    expected = torch.tensor(roots, dtype=result.roots.dtype, device=result.roots.device)
    torch.testing.assert_close(result.roots, expected, equal_nan=True)
    assert torch.equal(result.valid, torch.isfinite(expected))
    assert result.degree.item() == degree
    assert result.degree.dtype == torch.int64
    assert result.residual.shape == result.roots.shape
    assert torch.isnan(result.residual[~result.valid]).all()
    assert (result.residual[result.valid] <= 1024 * torch.finfo(result.roots.dtype).eps).all()


def test_shape_contract(tensor):
    result = quadratic_real_roots(tensor([[1.0], [1.0]]), tensor([0.0, 0.0, 0.0]), tensor(-1.0))
    assert result.roots.shape == result.valid.shape == result.residual.shape == (2, 3, 3)
    assert result.degree.shape == (2, 3)
    assert result.valid.dtype == torch.bool


def test_linear_root(tensor):
    assert_result(linear_roots(tensor(2.0), tensor(-6.0)), [3.0, float("nan"), float("nan")], 1)


def test_constant(tensor):
    assert_result(linear_roots(tensor(0.0), tensor(2.0)), [float("nan")] * 3, 0)


def test_zero(tensor):
    assert_result(linear_roots(tensor(0.0), tensor(0.0)), [float("nan")] * 3, -1)


def test_quadratic_two_roots(tensor):
    assert_result(quadratic_real_roots(tensor(1.0), tensor(-3.0), tensor(2.0)), [1.0, 2.0, float("nan")], 2)


def test_quadratic_double(tensor):
    assert_result(quadratic_real_roots(tensor(1.0), tensor(-2.0), tensor(1.0)), [1.0, 1.0, float("nan")], 2)


def test_quadratic_no_real(tensor):
    assert_result(quadratic_real_roots(tensor(1.0), tensor(0.0), tensor(1.0)), [float("nan")] * 3, 2)


def test_quadratic_zero_b(tensor):
    assert_result(quadratic_real_roots(tensor(1.0), tensor(0.0), tensor(-1.0)), [-1.0, 1.0, float("nan")], 2)


def test_quadratic_zero_double(tensor):
    assert_result(quadratic_real_roots(tensor(1.0), tensor(0.0), tensor(0.0)), [0.0, 0.0, float("nan")], 2)


@pytest.mark.parametrize("coeff,degree,roots", [
    ([0., 2., -6.], 1, [3., float("nan"), float("nan")]),
    ([0., 0., 2.], 0, [float("nan")] * 3),
    ([0., 0., 0.], -1, [float("nan")] * 3),
])
def test_quadratic_fallback(tensor, coeff, degree, roots):
    assert_result(quadratic_real_roots(*(tensor(c) for c in coeff)), roots, degree)


def test_cubic_companion(tensor):
    assert_result(cubic_real_roots(*(tensor(c) for c in [1., 0., 0., 0.]),
                                  method="companion"), [0., 0., 0.], 3)


def test_dtype_promotion():
    result = linear_roots(torch.tensor(2., dtype=torch.float32), torch.tensor(-6., dtype=torch.float64))
    assert result.roots.dtype == torch.float64
    assert result.roots.device.type == "cpu"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(tensor, bad):
    with pytest.raises(ValueError):
        quadratic_real_roots(tensor(1.), tensor(bad), tensor(1.))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_unsupported_dtype(dtype):
    with pytest.raises(TypeError):
        validate_coefficients(torch.tensor(1, dtype=dtype))


def test_device_mismatch():
    with pytest.raises(ValueError, match="same device"):
        validate_coefficients(torch.tensor(1.), torch.empty((), device="meta"))


def test_cancellation_and_scaling(tensor):
    for scale in [1e-20, 1., 1e20]:
        result = quadratic_real_roots(tensor(scale), tensor(-1000 * scale), tensor(scale))
        assert result.valid.tolist() == [True, True, False]
        torch.testing.assert_close(result.roots[:2].prod(), tensor(1.))
        assert (result.residual[:2] <= 1024 * torch.finfo(result.roots.dtype).eps).all()


def test_polishing_guards(tensor):
    coeff = (tensor(1.), tensor(0.), tensor(-1.))
    roots = tensor([0., 0.9, 1.1])
    valid = torch.ones(3, dtype=torch.bool)
    polished = polish_roots(coeff, roots, valid)
    assert polished[0] == roots[0]  # Zero derivative: do not divide or update.
    assert (normalized_residual(coeff, polished) <= normalized_residual(coeff, roots)).all()


def test_small_nonzero_quadratic_leading_coefficient(tensor):
    args = (tensor(1e-10), tensor(-1.), tensor(1.))
    result = quadratic_real_roots(*args)
    assert result.degree.item() == 2
    assert result.valid.tolist() == [True, True, False]
    torch.testing.assert_close(result.roots[:2], tensor([1., 1e10]))
    lower = quadratic_real_roots(*args, degree_tol=1e-8)
    assert lower.degree.item() == 1
    assert lower.valid.tolist() == [True, False, False]
    torch.testing.assert_close(lower.roots, tensor([1., float("nan"), float("nan")]), equal_nan=True)
    # Deliberately dropping a nonzero term need not give a machine-epsilon
    # residual for the original quadratic; the reported residual still uses it.
    torch.testing.assert_close(lower.residual[0], tensor(1e-10 / (2 + 1e-10)))


def test_small_nonzero_linear_leading_coefficient(tensor):
    result = linear_roots(tensor(1e-10), tensor(-1.))
    assert_result(result, [1e10, float("nan"), float("nan")], 1)
    lower = linear_roots(tensor(1e-10), tensor(-1.), degree_tol=1e-8)
    assert_result(lower, [float("nan")] * 3, 0)


@pytest.mark.parametrize("solver,args", [
    (linear_roots, [1e-10, -1.]),
    (quadratic_real_roots, [1e-10, -1., 1.]),
    (cubic_real_roots, [1e-10, -1., 1., 0.]),
])
def test_none_degree_tolerance_matches_zero(tensor, solver, args):
    coefficients = tuple(tensor(c) for c in args)
    actual = solver(*coefficients, degree_tol=None)
    expected = solver(*coefficients, degree_tol=0.)
    assert torch.equal(actual.degree, expected.degree)
    assert torch.equal(actual.valid, expected.valid)
    torch.testing.assert_close(actual.roots, expected.roots, equal_nan=True)
