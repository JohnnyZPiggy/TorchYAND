"""Small local CPU tests for the project-owned scientific adapter only."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('scientific_pilot', ROOT/'experiments/sae_mvsk/driver.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
pilot.torch = torch


class ScientificPilotTests(unittest.TestCase):
    def test_cpu_oracle_and_nontrivial_kkt_convergence(self):
        pilot.verify_package(ROOT)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                result = pilot.numerical_checks('cpu', dtype)
                self.assertEqual(result['status'], 'PASS')
                self.assertGreater(result['quadratic']['accepted_steps'], 0)

    def test_independent_reference_rejects_bad_analytic_gradient(self):
        from torchyand import MVSKOracle
        original = MVSKOracle.value_grad
        def incorrect(oracle, *args, **kwargs):
            value, gradient, cache = original(oracle, *args, **kwargs)
            return value, gradient+1, cache
        with patch.object(MVSKOracle, 'value_grad', incorrect):
            with self.assertRaises(AssertionError):
                pilot.numerical_checks('cpu', torch.float64)

    def test_reference_directional_finite_difference(self):
        rng = torch.Generator().manual_seed(21)
        raw = torch.randn(31, 5, dtype=torch.float64, generator=rng)*.2
        x = torch.randn(5, dtype=torch.float64, generator=rng).requires_grad_()
        direction = torch.randn(5, dtype=torch.float64, generator=rng)
        c = torch.tensor([1., .8, .3, .2], dtype=torch.float64)
        gradient, = torch.autograd.grad(pilot.reference_value(raw, x, c), x)
        step = 1e-5
        difference = (pilot.reference_value(raw, x+step*direction, c)-pilot.reference_value(raw, x-step*direction, c))/(2*step)
        torch.testing.assert_close(difference, gradient.dot(direction), atol=1e-9, rtol=1e-8)

    def test_mismatched_source_is_rejected(self):
        with patch.object(Path, 'read_bytes', return_value=b'changed algorithm'), \
             patch.object(Path, 'is_symlink', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'ALGORITHM_SOURCE_MISMATCH'):
                pilot.verify_package(ROOT)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
