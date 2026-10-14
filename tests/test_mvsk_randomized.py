"""Seeded MVSK reference/property checks; no benchmarks or compilation."""
import pytest
import torch

from torchyand import MVSKOracle, prepare_mvsk_data
from mvsk_reference import derivatives


SIZES = [(1, 1, 1), (1, 4, 3), (8, 1, 16), (8, 16, 3),
         (64, 4, 1), (64, 64, 16), (256, 16, 3), (256, 64, 16)]
SEEDS = [20261010, 20261011, 20261012]
FAMILIES = ["zero", "mean", "variance", "skew", "kurtosis", "mixed"]


def make_problem(dtype, size, seed, family="mixed", r_scale=1., x_scale=1.):
    T, n, batch = size
    generator = torch.Generator().manual_seed(seed)
    R = torch.randn(T, n, generator=generator, dtype=dtype)*r_scale
    c = .1+torch.rand(4, generator=generator, dtype=dtype)
    mask = {
        "zero": [0, 0, 0, 0], "mean": [1, 0, 0, 0],
        "variance": [0, 1, 0, 0], "skew": [0, 0, 1, 0],
        "kurtosis": [0, 0, 0, 1], "mixed": [1, 1, 1, 1],
    }[family]
    c = c*torch.tensor(mask, dtype=dtype)
    shape = (n,) if batch == 1 else (batch, n)
    X = torch.randn(shape, generator=generator, dtype=dtype)*x_scale
    U = torch.randn(shape, generator=generator, dtype=dtype)
    V = torch.randn(shape, generator=generator, dtype=dtype)
    return MVSKOracle(prepare_mvsk_data(R), c), X, U, V


def outputs(oracle, x, u, v, cache=None):
    value, grad, _ = oracle.value_grad(x, cache)
    return value, grad, oracle.hvp(x, v, cache), oracle.third_action(x, u, v, cache)


def independent(oracle, x, u, v):
    # vmap only the test reference. derivatives uses a scalar objective and
    # autograd grad/JVP; neither it nor this helper calls analytic kernels.
    reference = lambda xx, uu, vv: derivatives(xx, uu, vv, oracle.A, oracle.mu, oracle.coefficients)
    return reference(x, u, v) if x.ndim == 1 else torch.vmap(reference)(x, u, v)


def close(actual, expected, scale=None, factor=256):
    """dtype/norm-aware error: cancellation near zero uses local output scale."""
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    eps = torch.finfo(actual.dtype).eps
    # Per portfolio, not a whole-batch scale that could hide a bad small item.
    if actual.ndim > 0:
        output_scale = expected.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(actual.dtype).tiny)
    else:
        output_scale = expected.abs().clamp_min(torch.finfo(actual.dtype).tiny)
    if scale is not None:
        output_scale = torch.maximum(output_scale, scale)
    error = (actual-expected).abs()
    assert (error <= factor*eps*(expected.abs()+output_scale)).all(), (
        error.max().item(), output_scale.max().item(), actual.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("family", FAMILIES)
def test_randomized_autograd(dtype, size, seed, family):
    oracle, x, u, v = make_problem(dtype, size, seed, family)
    actual, expected = outputs(oracle, x, u, v), independent(oracle, x, u, v)
    for a, e in zip(actual, expected):
        close(a, e)
    assert actual[0].shape == x.shape[:-1]
    assert all(t.shape == x.shape for t in actual[1:])
    if x.ndim > 1:
        for i in range(min(3, len(x))):
            single_actual = outputs(oracle, x[i], u[i], v[i])
            single_expected = independent(oracle, x[i], u[i], v[i])
            for name, single, single_ref, batch, batch_ref in zip(
                    ("value", "gradient", "hvp", "third_action"),
                    single_actual, single_expected, actual, expected):
                batch_slice, batch_ref_slice = batch[i], batch_ref[i]
                # Evaluate both checks so a failure includes all four outputs.
                failures = []
                for label, reference in (("single oracle vs single reference", single_ref),
                                         ("single oracle vs batch oracle", batch_slice)):
                    try:
                        if label == "single oracle vs single reference":
                            close(single, reference)
                        else:
                            # Batched GEMM/reduction and single GEMV/reduction
                            # may use different floating-point reduction paths
                            # on CUDA. Calibrate batch/single error elementwise
                            # using the independent reference's path sensitivity,
                            # with the same rule for both dtypes and all devices.
                            oracle_path_error = (single-batch_slice).abs()
                            reference_path_error = (single_ref-batch_ref_slice).abs()
                            # One operation-level scale per portfolio: near-zero
                            # components share the output's roundoff scale, and
                            # max(1, ...) supplies a machine-epsilon absolute floor.
                            scale = torch.maximum(single_ref.abs().amax(), batch_ref_slice.abs().amax()).clamp_min(1)
                            allowance = 4*reference_path_error + 32*torch.finfo(dtype).eps*scale
                            assert (oracle_path_error <= allowance).all(), (
                                f"reference path error: {reference_path_error.detach().cpu().tolist()}\n"
                                f"allowed error: {allowance.detach().cpu().tolist()}")
                    except AssertionError as error:
                        absolute = (single-reference).abs()
                        relative = absolute/reference.abs().clamp_min(torch.finfo(dtype).tiny)
                        failures.append(
                            f"{label}: {error}\n"
                            f"absolute error: {absolute.detach().cpu().tolist()}\n"
                            f"relative error (abs(error)/max(abs(reference), tiny)): "
                            f"{relative.detach().cpu().tolist()}")
                assert not failures, (
                    f"operation: {name}; portfolio index: {i}\n"
                    f"single oracle: {single.detach().cpu().tolist()}\n"
                    f"single reference: {single_ref.detach().cpu().tolist()}\n"
                    f"batch oracle slice: {batch_slice.detach().cpu().tolist()}\n"
                    f"batch reference slice: {batch_ref_slice.detach().cpu().tolist()}\n"
                    + "\n".join(failures))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_randomized_properties(dtype, size, seed):
    oracle, x, u, v = make_problem(dtype, size, seed)
    cache = oracle.cache(x)
    cached, uncached = outputs(oracle, x, u, v, cache), outputs(oracle, x, u, v)
    for a, e in zip(cached, uncached):
        close(a, e)
    close(cached[0], oracle.value(x))
    close(cached[1], oracle.gradient(x))
    # Signed constants exercise cancellation, not only positive homogeneity.
    alpha, beta = .37, -1.23
    hu, hv = oracle.hvp(x, u, cache), oracle.hvp(x, v, cache)
    scale = (abs(alpha)*hu.abs()+abs(beta)*hv.abs()).amax(-1, keepdim=True)
    close(oracle.hvp(x, alpha*u+beta*v, cache), alpha*hu+beta*hv, scale)
    uv, vv = oracle.third_action(x, u, v, cache), oracle.third_action(x, v, v, cache)
    scale = (abs(alpha)*uv.abs()+abs(beta)*vv.abs()).amax(-1, keepdim=True)
    close(oracle.third_action(x, alpha*u+beta*v, v, cache), alpha*uv+beta*vv, scale)
    uu = oracle.third_action(x, u, u, cache)
    scale = (abs(alpha)*uu.abs()+abs(beta)*uv.abs()).amax(-1, keepdim=True)
    close(oracle.third_action(x, u, alpha*u+beta*v, cache), alpha*uu+beta*uv, scale)
    close(oracle.third_action(x, v, u, cache), uv)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("size", [(8, 4, 3), (64, 16, 3), (256, 64, 3)])
def test_random_directional_finite_differences(seed, size):
    oracle, x, u, v = make_problem(torch.float64, size, seed, r_scale=.2, x_scale=.3)
    grad = oracle.gradient(x)
    hvp = oracle.hvp(x, v)
    third = oracle.third_action(x, u, v)
    scales = ((grad*v).abs().sum(-1).clamp_min(1), hvp.abs().amax(-1, keepdim=True).clamp_min(1),
              third.abs().amax(-1, keepdim=True).clamp_min(1))
    target = ((grad*v).sum(-1), hvp, third)
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        fd = ((oracle.value(x+h*v)-oracle.value(x-h*v))/(2*h),
              (oracle.gradient(x+h*v)-oracle.gradient(x-h*v))/(2*h),
              (oracle.hvp(x+h*u, v)-oracle.hvp(x-h*u, v))/(2*h))
        errors.append(max(((a-e).abs()/s).max().item() for a, e, s in zip(fd, target, scales)))
    assert errors[-1] < 1e-7
    assert errors[-1] < .01*errors[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("r_scale", [1e-3, 1., 1e3])
@pytest.mark.parametrize("x_scale", [1e-2, 1., 1e2])
def test_scale_stress(dtype, r_scale, x_scale):
    for family in FAMILIES:
        oracle, x, u, v = make_problem(dtype, (64, 16, 16), SEEDS[0], family, r_scale, x_scale)
        # These specified scales are representable. Any nonfinite output must
        # be an exception, not a silent NaN/Inf; compare all successful results.
        actual = outputs(oracle, x, u, v)
        expected = independent(oracle, x, u, v)
        for a, e in zip(actual, expected):
            close(a, e, factor=512)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_overflow_is_explicit(dtype):
    scale = 1e12 if dtype == torch.float32 else 1e100
    oracle, x, u, v = make_problem(dtype, (8, 4, 3), SEEDS[0], "kurtosis", scale, scale)
    for call in (lambda: oracle.value(x), lambda: oracle.gradient(x),
                 lambda: oracle.hvp(x, v), lambda: oracle.third_action(x, u, v)):
        try:
            result = call()
        except FloatingPointError:
            continue
        assert torch.isfinite(result).all()
    with pytest.raises(FloatingPointError):
        oracle.value(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [(1, 4, 3), (64, 16, 16), (256, 64, 3)])
def test_cuda_sample_parity(dtype, size):
    cpu, x, u, v = make_problem(dtype, size, SEEDS[1])
    from torchyand import MVSKData
    gpu = MVSKOracle(MVSKData(cpu.A.cuda(), cpu.mu.cuda(), cpu.T, cpu.n), cpu.coefficients.cuda())
    for a, e in zip(outputs(gpu, x.cuda(), u.cuda(), v.cuda()), outputs(cpu, x, u, v)):
        close(a.cpu(), e, factor=512)
