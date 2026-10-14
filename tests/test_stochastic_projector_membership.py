"""Stochastic RHS must belong to the exact projector supplied to PCG."""
import importlib

import pytest
import torch

from torchyand.data import MVSKData
from torchyand.geometry import build_geometry
from torchyand.oracle import MVSKOracle
from torchyand.solver import solve_yand, YANDOptions, SolverStatus


stochastic = importlib.import_module('torchyand.logdet.stochastic')
pcg = importlib.import_module('torchyand.linalg.pcg')


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.fixture(params=['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def device(request):
    return request.param


def problem(dtype, device):
    eye = torch.eye(3, dtype=dtype, device=device)
    A = torch.cat((eye, -eye), dim=0)
    oracle = MVSKOracle(MVSKData(A, torch.zeros(3, dtype=dtype, device=device), 6, 3),
                        [0, 1, .1, .05])
    x = torch.tensor([.1, .2, .7], dtype=dtype, device=device)
    return oracle, x


def observe_rhs(monkeypatch):
    original = stochastic.batched_pcg
    observed = []

    def checked(matvec, b, *, projector, **options):
        pb = projector(b)
        error = pcg.kernels.norm(pb - b)
        # Exactly the existing PCG membership floor, with no relaxed tolerance.
        floor = (128 * torch.finfo(b.dtype).eps * b.shape[-1]**.5
                 * pcg.kernels.norm(b).clamp_min(1))
        assert torch.isfinite(error).all()
        assert (error <= floor).all(), (error, floor)
        observed.append(b.detach().clone())
        return original(matvec, b, projector=projector, **options)

    monkeypatch.setattr(stochastic, 'batched_pcg', checked)
    return observed


def test_final_rhs_projection_removes_path_error(dtype, device, monkeypatch):
    oracle, x = problem(dtype, device)
    geom = build_geometry(oracle.gradient(x))
    original = stochastic.kernels.project_probes
    initial_rhs = []

    def perturbed_path(probes, free_mask, nu, usable):
        xi = original(probes, free_mask, nu, usable)
        # Model contamination along the normal from a different rounding path.
        # It is deliberately above the unchanged membership floor, so this
        # regression detects removal of the final projection on every platform.
        xi = xi + 4096 * torch.finfo(dtype).eps * nu.unsqueeze(-2)
        initial_rhs.append(xi)
        return xi

    monkeypatch.setattr(stochastic.kernels, 'project_probes', perturbed_path)
    observed = observe_rhs(monkeypatch)
    probes = torch.tensor([[1., -1., 1.], [-1., 1., 1.],
                           [1., 1., -1.], [-1., -1., 1.]], dtype=dtype, device=device)
    result = stochastic.stochastic_logdet_correction(
        oracle, x, geom, q=4, probes=probes, damping=.2)
    assert result.success
    assert len(observed) == 1
    from torchyand.geometry.kernels import level_tangent
    expected = level_tangent(initial_rhs[0], geom.free_mask.unsqueeze(-2), geom.nu.unsqueeze(-2))
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    initial_error = pcg.kernels.norm(expected - initial_rhs[0])
    initial_floor = (128 * torch.finfo(dtype).eps * 3**.5
                     * pcg.kernels.norm(initial_rhs[0]).clamp_min(1))
    assert (initial_error > initial_floor).all()


def test_three_asset_stochastic_quartic_membership(dtype, device, monkeypatch):
    oracle, x = problem(dtype, device)
    observed = observe_rhs(monkeypatch)
    result = solve_yand(oracle, x, options=YANDOptions(logdet_mode='stochastic', max_iter=30),
                        seed=123)
    assert observed, 'The regression must exercise stochastic PCG solves'
    assert result.status in (SolverStatus.CONVERGED_KKT, SolverStatus.STALLED, SolverStatus.MAX_ITER)
    assert torch.isfinite(result.x).all() and torch.isfinite(result.value)
