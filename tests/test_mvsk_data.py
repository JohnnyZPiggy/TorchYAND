import pytest
import torch
from torchyand import prepare_mvsk_data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preprocessing(dtype):
    R = torch.randn(17, 5, generator=torch.Generator().manual_seed(7), dtype=dtype)
    data = prepare_mvsk_data(R)
    assert data.T == 17 and data.n == 5
    assert data.A.shape == (17, 5) and data.mu.shape == (5,)
    assert data.A.dtype == data.mu.dtype == dtype
    assert data.A.device == R.device
    torch.testing.assert_close(data.mu, R.double().mean(0).to(dtype))
    torch.testing.assert_close(data.A, (R.double()-R.double().mean(0)).to(dtype))
    torch.testing.assert_close(data.A.mean(0), torch.zeros(5, dtype=dtype),
                               atol=16*torch.finfo(dtype).eps, rtol=0)
    assert not hasattr(data, "R")


@pytest.mark.parametrize("center_dtype", [None, torch.float32, torch.float64])
@pytest.mark.parametrize("output_dtype", [None, torch.float32, torch.float64])
def test_precision_options(center_dtype, output_dtype):
    R = torch.tensor([[1., 2.], [3., 5.]], dtype=torch.float32)
    data = prepare_mvsk_data(R, center_dtype, output_dtype)
    assert data.A.dtype == (output_dtype or R.dtype)
    torch.testing.assert_close(data.A, torch.tensor([[-1., -1.5], [1., 1.5]], dtype=data.A.dtype))


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (8, 1)])
def test_small_dimensions(shape):
    data = prepare_mvsk_data(torch.arange(shape[0]*shape[1], dtype=torch.float64).reshape(shape))
    assert data.A.shape == shape
    torch.testing.assert_close(data.A.sum(0), torch.zeros(shape[1], dtype=torch.float64))


def test_preprocessing_autograd():
    R = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    data = prepare_mvsk_data(R)
    derivative = torch.autograd.grad(data.A.square().sum()+data.mu.sum(), R)[0]
    expected = 2*(R-R.mean(0))+torch.ones_like(R)/5
    torch.testing.assert_close(derivative, expected)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3,), (2, 3, 4)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError):
        prepare_mvsk_data(torch.empty(shape))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_invalid_dtype(dtype):
    with pytest.raises(TypeError):
        prepare_mvsk_data(torch.ones(2, 2, dtype=dtype))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(bad):
    with pytest.raises(ValueError):
        prepare_mvsk_data(torch.tensor([[bad]]))


def test_invalid_options_and_overflow():
    with pytest.raises(TypeError):
        prepare_mvsk_data(torch.ones(2, 2), center_dtype=torch.float16)
    with pytest.raises(TypeError):
        prepare_mvsk_data(torch.ones(2, 2), output_dtype=torch.float16)
    with pytest.raises(FloatingPointError):
        prepare_mvsk_data(torch.tensor([[1e300], [-1e300]], dtype=torch.float64), output_dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_centering():
    R = torch.randn(19, 3, dtype=torch.float64)
    cpu, gpu = prepare_mvsk_data(R), prepare_mvsk_data(R.cuda())
    assert gpu.A.device.type == "cuda"
    torch.testing.assert_close(cpu.A, gpu.A.cpu())
    torch.testing.assert_close(cpu.mu, gpu.mu.cpu())
