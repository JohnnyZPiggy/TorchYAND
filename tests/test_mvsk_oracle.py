from dataclasses import replace
import pytest
import torch

from torchyand import MVSKData, MVSKOracle, prepare_mvsk_data
from torchyand.oracle import kernels
from mvsk_reference import derivatives, scalar_objective


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def setup(dtype, coefficients=(1., .8, .3, .2), T=19, n=4):
    generator = torch.Generator().manual_seed(20261010)
    R = torch.randn(T, n, generator=generator, dtype=dtype)*.3
    oracle = MVSKOracle(prepare_mvsk_data(R), coefficients)
    x, u, v = torch.randn(3, n, generator=generator, dtype=dtype).unbind(0)
    return oracle, x, u, v


def close(actual, expected):
    eps = torch.finfo(actual.dtype).eps
    torch.testing.assert_close(actual, expected, atol=256*eps, rtol=256*eps)


@pytest.mark.parametrize("coeff", [(1., .8, .3, .2), (0., 0., 0., 0.),
                                  (1., 0., 0., 0.), (0., 1., 0., 0.),
                                  (0., 0., 1., 0.), (0., 0., 0., 1.)])
def test_autograd_reference(dtype, coeff):
    oracle, x, u, v = setup(dtype, coeff)
    expected = derivatives(x, u, v, oracle.A, oracle.mu, oracle.coefficients)
    value, grad, cache = oracle.value_grad(x)
    for actual, reference in zip((value, grad, oracle.hvp(x, v, cache),
                                  oracle.third_action(x, u, v, cache)), expected):
        close(actual, reference)
    close(oracle.gradient(x), grad)


def test_float64_finite_differences():
    oracle, x, u, v = setup(torch.float64)
    grad, hvp, third = oracle.gradient(x), oracle.hvp(x, v), oracle.third_action(x, u, v)
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        fd1 = (oracle.value(x+h*v)-oracle.value(x-h*v))/(2*h)
        fd2 = (oracle.gradient(x+h*v)-oracle.gradient(x-h*v))/(2*h)
        fd3 = (oracle.hvp(x+h*u, v)-oracle.hvp(x-h*u, v))/(2*h)
        errors.append(max((fd1-grad.dot(v)).abs().item(), (fd2-hvp).abs().max().item(),
                          (fd3-third).abs().max().item()))
    assert errors[-1] < 1e-8
    assert errors[-1] < errors[0]*.01


def test_linearity_bilinearity_symmetry(dtype):
    oracle, x, u, v = setup(dtype)
    w = u+2*v
    close(oracle.hvp(x, w), oracle.hvp(x, u)+2*oracle.hvp(x, v))
    close(oracle.third_action(x, w, v), oracle.third_action(x, u, v)+2*oracle.third_action(x, v, v))
    close(oracle.third_action(x, u, w), oracle.third_action(x, u, u)+2*oracle.third_action(x, u, v))
    close(oracle.third_action(x, u, v), oracle.third_action(x, v, u))


def test_batch_and_direction_broadcast(dtype):
    oracle, x, u, v = setup(dtype)
    X = torch.stack((x, x*.7)).unsqueeze(1)
    U = torch.stack((u, u*.3, u*2)).unsqueeze(0)
    V = v.expand(2, 3, -1)
    value, grad, cache = oracle.value_grad(X)
    hvp, third = oracle.hvp(X, V, cache), oracle.third_action(X, U, V, cache)
    assert value.shape == (2, 1) and grad.shape == (2, 1, 4)
    assert hvp.shape == third.shape == (2, 3, 4)
    for i in range(2):
        close(value[i, 0], oracle.value(X[i, 0]))
        close(grad[i, 0], oracle.gradient(X[i, 0]))
        for j in range(3):
            close(hvp[i, j], oracle.hvp(X[i, 0], V[i, j]))
            close(third[i, j], oracle.third_action(X[i, 0], U[0, j], V[i, j]))


def test_cache_reuse(dtype, monkeypatch):
    oracle, x, u, v = setup(dtype)
    expected = (oracle.value(x), oracle.gradient(x), oracle.hvp(x, v), oracle.third_action(x, u, v))
    cache = oracle.cache(x)
    assert cache.x is x and cache.z.shape == (oracle.T,)
    assert set(vars(cache)) == {"x", "z", "owner", "shape", "dtype", "device"}
    def forbidden(*args):
        raise AssertionError("cached projection must not be recomputed")
    monkeypatch.setattr(kernels, "projection", forbidden)
    actual = (oracle.value(x, cache), oracle.gradient(x, cache), oracle.hvp(x, v, cache), oracle.third_action(x, u, v, cache))
    for a, e in zip(actual, expected):
        close(a, e)
    assert oracle.value_grad(x, cache)[2] is cache


def test_bad_cache(dtype):
    oracle, x, _, _ = setup(dtype)
    cache = oracle.cache(x)
    other, _, _, _ = setup(dtype)
    with pytest.raises(ValueError, match="belong"):
        other.value(x, cache)
    with pytest.raises(ValueError, match="same x"):
        oracle.value(x.clone(), cache)
    for bad in (replace(cache, shape=torch.Size([99])), replace(cache, dtype=torch.float16),
                replace(cache, device=torch.device("meta")), replace(cache, z=cache.z[:-1])):
        with pytest.raises(ValueError):
            oracle.value(x, bad)


def test_full_autograd_chain():
    R = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    c = torch.tensor([1., .8, .3, .2], dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    oracle = MVSKOracle(prepare_mvsk_data(R), c)
    cached = oracle.cache(x)
    actual = torch.autograd.grad(oracle.value(x, cached), (R, c, x), create_graph=True)
    A, mu = R-R.mean(0), R.mean(0)
    expected = torch.autograd.grad(scalar_objective(x, A, mu, c), (R, c, x), create_graph=True)
    for a, e in zip(actual, expected):
        close(a, e)
    for result in (oracle.gradient(x, cached), oracle.hvp(x, x, cached), oracle.third_action(x, x, x, cached)):
        assert result.requires_grad
        assert torch.autograd.grad(result.sum(), R, retain_graph=True)[0] is not None


@pytest.mark.parametrize("T,n", [(1, 1), (1, 4), (7, 1)])
def test_small_dimensions(dtype, T, n):
    oracle, x, u, v = setup(dtype, T=T, n=n)
    expected = derivatives(x, u, v, oracle.A, oracle.mu, oracle.coefficients)
    actual = (oracle.value(x), oracle.gradient(x), oracle.hvp(x, v), oracle.third_action(x, u, v))
    for a, e in zip(actual, expected):
        close(a, e)
    if T == 1:
        assert torch.count_nonzero(oracle.A) == 0
        close(actual[0], -oracle.coefficients[0]*oracle.mu.dot(x))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs(dtype, bad):
    oracle, x, u, v = setup(dtype)
    bad_vector = x.clone()
    bad_vector[0] = bad
    for call in (lambda: oracle.value(bad_vector), lambda: oracle.gradient(bad_vector),
                 lambda: oracle.cache(bad_vector), lambda: oracle.hvp(x, bad_vector),
                 lambda: oracle.third_action(x, bad_vector, v),
                 lambda: oracle.third_action(x, u, bad_vector)):
        with pytest.raises(ValueError):
            call()


def test_invalid_contract(dtype):
    oracle, x, _, _ = setup(dtype)
    for bad in (x.half(), x.long(), x.to(torch.float64 if dtype == torch.float32 else torch.float32)):
        with pytest.raises(TypeError):
            oracle.value(bad)
    with pytest.raises(ValueError):
        oracle.value(x[:-1])
    with pytest.raises(ValueError, match="device"):
        oracle.value(torch.empty(4, dtype=dtype, device="meta"))
    with pytest.raises(ValueError):
        MVSKOracle(oracle.data, torch.ones(2, 4, dtype=dtype))
    with pytest.raises(ValueError):
        MVSKOracle(oracle.data, [-1., 0., 0., 0.])
    with pytest.raises(ValueError):
        MVSKOracle(oracle.data, [float("nan"), 0., 0., 0.])
    with pytest.raises(TypeError):
        MVSKOracle(oracle.data, torch.ones(4, dtype=torch.float16))
    with pytest.raises(ValueError):
        MVSKOracle(MVSKData(oracle.A, oracle.mu, 99, oracle.n), [1.]*4)


def test_numerical_overflow():
    oracle = MVSKOracle(prepare_mvsk_data(torch.tensor([[-1.], [1.]])), [0., 0., 0., 1.])
    with pytest.raises(FloatingPointError):
        oracle.value(torch.tensor([1e20]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity(dtype):
    cpu, x, u, v = setup(dtype)
    gpu = MVSKOracle(MVSKData(cpu.A.cuda(), cpu.mu.cuda(), cpu.T, cpu.n), cpu.coefficients.cuda())
    for a, e in zip((gpu.value(x.cuda()), gpu.gradient(x.cuda()), gpu.hvp(x.cuda(), v.cuda()),
                     gpu.third_action(x.cuda(), u.cuda(), v.cuda())),
                    (cpu.value(x), cpu.gradient(x), cpu.hvp(x, v), cpu.third_action(x, u, v))):
        close(a.cpu(), e)
