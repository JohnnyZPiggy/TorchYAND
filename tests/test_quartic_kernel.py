"""Strict fullgraph capture and public/unchecked numerical equivalence."""
import pytest
import torch

from torchyand.polynomial import minimize_quartic_interval
from torchyand.polynomial.quartic_line_search import _minimize_quartic_interval_unchecked


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("settings", [{}, {"interval_tol": .01, "value_tol": 0.},
                                      {"interval_tol": 0., "value_tol": .1}])
def test_fullgraph_capture(dtype, settings):
    maximum = torch.finfo(dtype).max
    coeff = torch.tensor([
        [1., 0., -2., 0., 1.], [0., -3., 0., 1., 0.],
        [1., -2., 1., 0., 0.], [3., -2., 0., 0., 0.],
        [4., 0., 0., 0., 0.], [1e20, -2., 1., 0., 0.],
        [0., .002, 1., 0., 0.], [0., 0., 0., 0., maximum],
    ], dtype=dtype).unsqueeze(0).expand(2, -1, -1)
    lower = torch.full((2, 8), -2., dtype=dtype)
    upper = torch.full((2, 8), 3., dtype=dtype)
    lower[1] = 0.
    upper[1, 0] = 0.  # zero-width interval and repeated candidates
    graphs = []
    def capture_backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward
    torch.compiler.reset()
    compiled = torch.compile(_minimize_quartic_interval_unchecked,
                             backend=capture_backend, fullgraph=True, dynamic=False)
    expected = minimize_quartic_interval(coeff, lower, upper, **settings)
    unchecked = _minimize_quartic_interval_unchecked(coeff, lower, upper, **settings)
    result = compiled(coeff, lower, upper, **settings)
    assert len(graphs) == 1  # Actual single-graph capture, not Inductor execution.
    for actual in (unchecked, result):
        assert torch.equal(actual.success, expected.success)
        assert torch.equal(actual.candidate_valid, expected.candidate_valid)
        for name in ("x", "value", "candidate_x", "candidate_value"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                       equal_nan=True, rtol=0, atol=0)
        assert not (actual.success & ((actual.x < lower) | (actual.x > upper))).any()
        assert torch.equal(actual.root_result.degree, expected.root_result.degree)
        assert torch.equal(actual.root_result.valid, expected.root_result.valid)
