import pytest
import torch

from torchyand.polynomial import minimize_quartic_interval
from torchyand.polynomial.evaluate import quartic_horner


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.mark.parametrize("coeff,lower,upper,x,value", [
    ([1., 0., 2., 0., 1.], -2., 3., 0., 1.),
    ([0., 0., 0., 0., 1.], 1., 3., 1., 1.),
    ([0., 0., 0., 0., 1.], -3., -1., -1., 1.),
    ([1., 0., -2., 0., 1.], -2., 2., -1., 0.),
    ([0., -3., 0., 1., 0.], 0., 2., 1., -2.),
    ([1., -2., 1., 0., 0.], -1., 3., 1., 0.),
    ([3., -2., 0., 0., 0.], -1., 2., 2., -1.),
    ([3., 0., 0., 0., 0.], -1., 2., -1., 3.),
    ([0., 0., 0., 0., 0.], -1., 2., -1., 0.),
    ([1., 2., 3., 4., 5.], 1., 1., 1., 15.),
    ([1., -2., 1., 0., 0.], 1., 3., 1., 0.),
])
def test_known_minima(dtype, coeff, lower, upper, x, value):
    coeff = torch.tensor(coeff, dtype=dtype)
    result = minimize_quartic_interval(coeff, lower, upper)
    assert result.success
    tol = 256 * torch.finfo(dtype).eps
    torch.testing.assert_close(result.x, torch.tensor(x, dtype=dtype), atol=tol, rtol=tol)
    torch.testing.assert_close(result.value, torch.tensor(value, dtype=dtype), atol=tol, rtol=tol)
    assert result.candidate_x.shape == result.candidate_value.shape == result.candidate_valid.shape == (5,)
    assert torch.isnan(result.candidate_x[~result.candidate_valid]).all()
    assert torch.isnan(result.candidate_value[~result.candidate_valid]).all()
    torch.testing.assert_close(result.candidate_x[:2], torch.tensor([lower, upper], dtype=dtype))
    assert result.success.dtype == torch.bool


def test_near_boundary_and_absolute_tolerance(dtype):
    r = -.001
    coeff = torch.tensor([r*r, -2*r, 1., 0., 0.], dtype=dtype)
    result = minimize_quartic_interval(coeff, 0., 1., interval_tol=.002)
    assert result.candidate_valid[2]
    assert result.candidate_x[2] == 0
    strict = minimize_quartic_interval(coeff, 0., 1., interval_tol=.0001)
    assert not strict.candidate_valid[2]


def test_default_boundary_tolerance(dtype):
    gap = 16 * torch.finfo(dtype).eps
    coeff = torch.tensor([gap*gap, 2*gap, 1., 0., 0.], dtype=dtype)
    result = minimize_quartic_interval(coeff, 0., 1.)
    assert result.candidate_valid[2]
    assert result.candidate_x[2] == 0


def test_mixed_degree_broadcast(dtype):
    coeff = torch.tensor([[0., 0., 2., 0., 1.], [0., -3., 0., 1., 0.],
                          [1., -2., 1., 0., 0.], [0., -1., 0., 0., 0.],
                          [4., 0., 0., 0., 0.]], dtype=dtype).unsqueeze(0)
    lower = torch.tensor([[-2.], [0.]], dtype=dtype)
    upper = torch.tensor([2., 3., 2., 2., 2.], dtype=dtype)
    result = minimize_quartic_interval(coeff, lower, upper)
    assert result.x.shape == (2, 5)
    assert result.candidate_x.shape == (2, 5, 5)
    assert result.root_result.roots.shape == (2, 5, 3)
    assert result.success.all()
    assert result.root_result.degree[0].tolist() == [3, 2, 1, 0, -1]
    torch.testing.assert_close(result.x, torch.tensor([[0., -2., 1., 2., -2.],
                                                      [0., 1., 1., 2., 0.]], dtype=dtype))


def test_huge_constant_shifted_comparison(dtype):
    huge = 1e20
    result = minimize_quartic_interval(torch.tensor([huge, -2., 1., 0., 0.], dtype=dtype), 0., 2.)
    assert result.success
    assert result.x == 1
    # Full values all round to the same number; comparison must omit A0.
    assert (result.candidate_value[result.candidate_valid] == torch.tensor(huge, dtype=dtype)).all()


def test_tie_chooses_smaller_x_not_slot(dtype):
    coeff = torch.tensor([1., 0., -2., 0., 1.], dtype=dtype)
    result = minimize_quartic_interval(coeff, 2., 3., value_tol=100.)
    assert result.x == 2
    result = minimize_quartic_interval(coeff, -2., 2.)
    assert result.x < 0
    assert result.candidate_valid.all()  # endpoints plus three stationary roots


def test_value_tolerance_absolute(dtype):
    coeff = torch.tensor([1e20, -1., 0., 0., 0.], dtype=dtype)
    strict = minimize_quartic_interval(coeff, 0., 1., value_tol=.1)
    relaxed = minimize_quartic_interval(coeff, 0., 1., value_tol=2.)
    assert strict.x == 1
    assert relaxed.x == 0


def test_default_value_tolerance(dtype):
    eps = torch.finfo(dtype).eps
    for slope, expected in [(64*eps, 0.), (256*eps, 1.)]:
        coeff = torch.tensor([0., -slope, 0., 0., 0.], dtype=dtype)
        assert minimize_quartic_interval(coeff, 0., 1.).x == expected
    # At large shifted scale the default tolerance grows accordingly.
    coeff = torch.tensor([0., 1e6, 0., 0., 0.], dtype=dtype)
    lower = torch.tensor(1., dtype=dtype)
    upper = torch.tensor(1. + 32*eps, dtype=dtype)
    assert minimize_quartic_interval(coeff, lower, upper).x == lower


def test_nonzero_high_order_coefficient_preserved(dtype):
    result = minimize_quartic_interval(torch.tensor([0., 0., 1., 0., 1e-10], dtype=dtype), -1., 1.)
    assert result.root_result.degree == 3
    assert result.success
    torch.testing.assert_close(result.x, torch.tensor(0., dtype=dtype))


def test_duplicate_boundary_slots(dtype):
    result = minimize_quartic_interval(torch.tensor([0., 0., 0., 0., 1.], dtype=dtype), 0., 0.)
    assert result.candidate_valid.all()
    assert (result.candidate_x == 0).all()


def test_evaluation_overflow_fails_entire_item(dtype):
    large = 1e20 if dtype == torch.float32 else 1e200
    coeff = torch.tensor([0., 0., 0., 0., 1.], dtype=dtype)
    result = minimize_quartic_interval(coeff, 0., large)
    assert not result.success  # lower is finite, upper overflow cannot be ignored
    assert torch.isnan(result.x) and torch.isnan(result.value)
    assert torch.isinf(result.candidate_value[1])


def test_selected_full_value_overflow(dtype):
    maximum = torch.finfo(dtype).max
    coeff = torch.tensor([maximum, maximum/2, 0., 0., 0.], dtype=dtype)
    result = minimize_quartic_interval(coeff, 2., 2.)
    assert not result.success
    assert torch.isnan(result.value)


def test_derivative_overflow_batch_isolated(dtype):
    maximum = torch.finfo(dtype).max
    coeff = torch.tensor([[0., 0., 0., 0., maximum], [0., 0., 1., 0., 0.]], dtype=dtype)
    result = minimize_quartic_interval(coeff, 0., 1.)
    assert result.success.tolist() == [False, True]


def test_horner_order(dtype):
    coeff = torch.tensor([1., 2., 3., 4., 5.], dtype=dtype)
    torch.testing.assert_close(quartic_horner(coeff, torch.tensor([0., 1., 2.], dtype=dtype)),
                               torch.tensor([1., 15., 129.], dtype=dtype))


def test_dtype_promotion():
    result = minimize_quartic_interval(torch.tensor([0., 0., 1., 0., 0.], dtype=torch.float32),
                                      torch.tensor(-1., dtype=torch.float64), 1.)
    assert result.x.dtype == torch.float64


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs(dtype, bad):
    with pytest.raises(ValueError):
        minimize_quartic_interval(torch.tensor([bad, 0., 0., 0., 1.], dtype=dtype), 0., 1.)
    with pytest.raises(ValueError):
        minimize_quartic_interval(torch.zeros(5, dtype=dtype), 0., bad)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        minimize_quartic_interval(torch.zeros(4), 0., 1.)
    with pytest.raises(ValueError):
        minimize_quartic_interval(torch.zeros(5), 2., 1.)
    with pytest.raises(TypeError):
        minimize_quartic_interval(torch.zeros(5, dtype=torch.float16), 0., 1.)
    with pytest.raises(ValueError, match="same device"):
        minimize_quartic_interval(torch.zeros(5), torch.empty((), device="meta"), 1.)
    for name in ("interval_tol", "value_tol"):
        with pytest.raises(ValueError):
            minimize_quartic_interval(torch.zeros(5), 0., 1., **{name: -1.})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    coeff = torch.tensor([[1., 0., -2., 0., 1.], [1., -2., 1., 0., 0.]], dtype=dtype)
    cpu = minimize_quartic_interval(coeff, -2., 3.)
    gpu = minimize_quartic_interval(coeff.cuda(), -2., 3.)
    assert gpu.x.device.type == "cuda"
    assert torch.equal(cpu.success, gpu.success.cpu())
    assert torch.equal(cpu.candidate_valid, gpu.candidate_valid.cpu())
    torch.testing.assert_close(cpu.x, gpu.x.cpu())
    torch.testing.assert_close(cpu.value, gpu.value.cpu())
