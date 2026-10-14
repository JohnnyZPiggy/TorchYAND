"""Independent NumPy references and seeded quartic interval properties.

Reference loops are test-only; production solvers receive whole tensor batches.
"""
import numpy as np
import pytest
import torch

from torchyand.polynomial import minimize_quartic_interval


COUNT = 10000
SEED = 20261008


def horner(coeff, x):
    a0, a1, a2, a3, a4 = np.moveaxis(coeff, -1, 0)
    return a0 + x * (a1 + x * (a2 + x * (a3 + x*a4)))


def reference(coeff, lower, upper):
    """NumPy polyroots of the actual rounded inputs; no project root solver.

    Compare shifted values to avoid cancellation from A0; return full values.
    Reference imaginary/boundary tolerances use float64 epsilon and root scale.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    x = np.empty(len(coeff))
    value = np.empty(len(coeff))
    for i, (c, lo, hi) in enumerate(zip(coeff, lower, upper)):
        derivative = c[1:] * np.arange(1, 5)
        roots = np.polynomial.polynomial.polyroots(derivative)
        real = roots.real
        real_mask = np.abs(roots.imag) <= 128*np.finfo(float).eps*np.maximum(1, np.abs(real))
        boundary_tol = 128*np.finfo(float).eps*max(1, abs(lo), abs(hi))
        inside = real_mask & (real >= lo-boundary_tol) & (real <= hi+boundary_tol)
        candidates = np.sort(np.r_[lo, hi, np.clip(real[inside], lo, hi)])
        shifted_coeff = c.copy()
        shifted_coeff[0] = 0
        shifted = horner(shifted_coeff, candidates)
        index = np.argmin(shifted)
        x[i] = candidates[index]
        value[i] = horner(c, x[i])
    return x, value


def problems(count=COUNT):
    rng = np.random.default_rng(SEED)
    kind = np.arange(count) % 7
    center = rng.uniform(-2, 2, count)
    center[::4] += 20  # intervals far from zero
    width = rng.uniform(.5, 4, count)
    width[::5] = rng.uniform(1e-4, 1e-3, len(width[::5]))
    lower = center-rng.uniform(.1, .9, count)*width
    upper = lower+width
    h = center+rng.uniform(-.2, .2, count)
    a = rng.uniform(.5, 2, count)
    b = rng.uniform(.5, 2, count)
    # Coefficients initially in powers of t=x-h, with seven distinct families.
    t = np.zeros((count, 5))
    t[:, 0] = rng.uniform(-2, 2, count)
    t[kind == 0, 4], t[kind == 0, 2] = a[kind == 0], b[kind == 0]
    t[kind == 1, 4], t[kind == 1, 2] = a[kind == 1], -b[kind == 1]
    t[kind == 1, 1] = rng.uniform(-.1, .1, (kind == 1).sum())
    monotone = kind == 2
    h[monotone] = lower[monotone]
    t[monotone, 1:5] = rng.uniform(.2, 1, (monotone.sum(), 4))
    t[kind == 3, 1:4] = rng.uniform(-2, 2, ((kind == 3).sum(), 3))
    t[kind == 4, 2] = a[kind == 4]
    t[kind == 5, 1] = rng.choice([-1., 1.], (kind == 5).sum())*a[kind == 5]
    c = np.empty_like(t)
    c[:, 4] = t[:, 4]
    c[:, 3] = t[:, 3]-4*h*t[:, 4]
    c[:, 2] = t[:, 2]-3*h*t[:, 3]+6*h*h*t[:, 4]
    c[:, 1] = t[:, 1]-2*h*t[:, 2]+3*h*h*t[:, 3]-4*h**3*t[:, 4]
    c[:, 0] = t[:, 0]-h*t[:, 1]+h*h*t[:, 2]-h**3*t[:, 3]+h**4*t[:, 4]
    return c, lower, upper, kind, h


def assert_properties(result, coeff, lower, upper):
    success = result.success.cpu().numpy()
    x, value = result.x.cpu().numpy(), result.value.cpu().numpy()
    assert np.all((x[success] >= lower[success]) & (x[success] <= upper[success]))
    assert np.isnan(x[~success]).all() and np.isnan(value[~success]).all()
    cx = result.candidate_x.cpu().numpy()
    valid = result.candidate_valid.cpu().numpy()
    assert np.isnan(cx[~valid]).all()
    assert np.isnan(result.candidate_value.cpu().numpy()[~valid]).all()
    assert np.all(np.any(valid[success] & (cx[success] == x[success, None]), axis=-1))
    # Independent direct Horner at actual selected and candidate locations.
    # Match the solver's arithmetic dtype here; a float64 evaluation of a
    # cancellation-heavy float32 polynomial is checked separately by reference.
    with np.errstate(over="ignore", invalid="ignore"):
        expected = horner(coeff.astype(cx.dtype)[:, None, :], np.where(valid, cx, 0))
    eps = torch.finfo(result.x.dtype).eps
    np.testing.assert_allclose(result.candidate_value.cpu().numpy()[valid], expected[valid],
                               rtol=16*eps, atol=16*eps)


def compare_reference(result, coeff, lower, upper):
    rx, rv = reference(coeff, lower, upper)
    x = result.x.cpu().numpy()
    # Horner error and default tie threshold depend on scale, including large
    # endpoint values. A0 alone must not hide shifted-objective discrepancies.
    shifted = coeff.copy()
    shifted[:, 0] = 0
    sx, sr = horner(shifted, x), horner(shifted, rx)
    bound_x = np.maximum.reduce([np.ones_like(x), abs(lower), abs(upper)])
    arithmetic_scale = sum(abs(shifted[:, i])*bound_x**i for i in range(1, 5))
    eps = torch.finfo(result.x.dtype).eps
    allowed = 2048*eps*np.maximum(1, arithmetic_scale)
    assert np.all(abs(sx-sr) <= allowed), (np.max(abs(sx-sr)/allowed), x[np.argmax(abs(sx-sr)/allowed)])
    np.testing.assert_array_less(abs(result.value.cpu().numpy()-rv),
                                 allowed+32*eps*np.maximum(1, abs(rv)))
    return rx


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_randomized_reference(dtype):
    c, lo, hi, kind, h = problems()
    tensors = [torch.tensor(v, dtype=dtype) for v in (c, lo, hi)]
    c, lo, hi = [v.numpy().astype(np.float64) for v in tensors]
    result = minimize_quartic_interval(*tensors)
    assert result.success.all()
    assert_properties(result, c, lo, hi)
    rx = compare_reference(result, c, lo, hi)
    # Strict x check only for isolated convex minima with healthy curvature,
    # modest coefficient scales and endpoints well separated from the optimum.
    healthy = ((kind == 0) | (kind == 4)) & (abs(h) < 1) & (rx-lo > .2) & (hi-rx > .2)
    assert healthy.sum() > 100
    eps = torch.finfo(dtype).eps
    np.testing.assert_allclose(result.x.numpy()[healthy], rx[healthy], rtol=128*eps, atol=128*eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shift_scaling_and_batch_properties(dtype):
    c, lo, hi, _, _ = problems(140)
    coeff, lower, upper = [torch.tensor(v, dtype=dtype) for v in (c, lo, hi)]
    base = minimize_quartic_interval(coeff, lower, upper)
    for constant in (0., 1e10, 1e20):
        shifted = coeff.clone()
        shifted[:, 0] += constant
        actual = minimize_quartic_interval(shifted, lower, upper)
        assert torch.equal(base.success, actual.success)
        torch.testing.assert_close(base.x, actual.x, rtol=0, atol=0)
    # Default absolute floor in tie tolerance is intentionally not scale
    # invariant. Disable ties to test mathematical positive-scale invariance.
    strict = minimize_quartic_interval(coeff, lower, upper, value_tol=0.)
    for scale in (1e-20, 1e-10, 1., 1e10, 1e20):
        scaled = coeff*scale
        actual = minimize_quartic_interval(scaled, lower, upper, value_tol=0.)
        assert actual.success.all()
        eps = torch.finfo(dtype).eps
        torch.testing.assert_close(actual.x, strict.x, rtol=256*eps**.5, atol=256*eps**.5)
        compare_reference(actual, scaled.double().numpy(), lower.double().numpy(), upper.double().numpy())
        default = minimize_quartic_interval(scaled, lower, upper)
        assert_properties(default, scaled.double().numpy(), lower.double().numpy(), upper.double().numpy())
        compare_reference(default, scaled.double().numpy(), lower.double().numpy(), upper.double().numpy())
    eps = torch.finfo(dtype).eps
    for i in range(28):
        single = minimize_quartic_interval(coeff[i], lower[i], upper[i])
        assert torch.equal(single.success, base.success[i])
        # Scalar and batched CUDA kernels may round differently by a few ULPs.
        torch.testing.assert_close(single.x, base.x[i], rtol=8*eps, atol=8*eps)
        torch.testing.assert_close(single.value, base.value[i], equal_nan=True,
                                   rtol=8*eps, atol=8*eps)
        torch.testing.assert_close(single.candidate_value, base.candidate_value[i], equal_nan=True,
                                   rtol=8*eps, atol=8*eps)
    broadcast = minimize_quartic_interval(coeff[:14].unsqueeze(0), lower[:14].unsqueeze(0),
                                          upper[:14].unsqueeze(0).expand(3, 14))
    torch.testing.assert_close(broadcast.x, base.x[:14].unsqueeze(0).expand(3, 14), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_near_boundary_and_tied_wells(dtype):
    eps = torch.finfo(dtype).eps
    gaps = np.array([0., 8*eps, -8*eps, 1e-4, -1e-4])
    roots = np.r_[gaps, 1+gaps]
    c = np.zeros((len(roots), 5))
    c[:, 0], c[:, 1], c[:, 2] = roots**2, -2*roots, 1
    tilts = np.array([0., eps, -eps, 1e-6, -1e-6])
    wells = np.tile([1., 0., -2., 0., 1.], (len(tilts), 1))
    wells[:, 1] = tilts
    c = np.r_[c, wells]
    lo, hi = np.r_[np.zeros(len(roots)), -2*np.ones(len(tilts))], np.r_[np.ones(len(roots)), 2*np.ones(len(tilts))]
    coeff, lower, upper = [torch.tensor(v, dtype=dtype) for v in (c, lo, hi)]
    result = minimize_quartic_interval(coeff, lower, upper)
    assert result.success.all()
    assert_properties(result, coeff.double().numpy(), lower.double().numpy(), upper.double().numpy())
    compare_reference(result, coeff.double().numpy(), lower.double().numpy(), upper.double().numpy())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_tie_floor_not_scale_invariant(dtype):
    c = torch.tensor([0., -2., 1., 0., 0.], dtype=dtype)
    assert minimize_quartic_interval(c, 0., 2.).x == 1
    assert minimize_quartic_interval(c*1e-20, 0., 2.).x == 0
    assert minimize_quartic_interval(c*1e-20, 0., 2., value_tol=0.).x == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_failure_semantics_batch(dtype):
    large = 1e20 if dtype == torch.float32 else 1e200
    c = torch.tensor([[0., 0., 0., 0., 1.], [0., 0., 1., 0., 0.]], dtype=dtype)
    lo = torch.zeros(2, dtype=dtype)
    hi = torch.tensor([large, 1.], dtype=dtype)
    result = minimize_quartic_interval(c, lo, hi)
    assert result.success.tolist() == [False, True]
    assert_properties(result, c.double().numpy(), lo.double().numpy(), hi.double().numpy())
