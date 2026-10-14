"""Verify unchecked refactor equivalence and strict fullgraph capture."""
import pytest
import torch
from torchyand.polynomial import cubic_real_roots
from torchyand.polynomial.cubic import _cubic_real_roots_unchecked


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree_tol", [0., 1e-5])
@pytest.mark.parametrize("polish_steps", [0, 1])
def test_fullgraph_kernel(dtype, degree_tol, polish_steps):
    coefficients = torch.tensor([
        [1., -6., 11., -6.], [1., -3., 3., -1.], [1., 0., -3., 2.],
        [1., 0., 1., 1.], [0., 1., -3., 2.], [0., 0., 2., -6.],
        [0., 0., 0., 2.], [0., 0., 0., 0.], [1e-7, 1., -3., 2.],
    ], dtype=dtype).unbind(-1)
    graphs = []
    def capture_backend(graph_module, inputs):
        graphs.append(graph_module)
        return graph_module.forward
    compiled = torch.compile(_cubic_real_roots_unchecked, backend=capture_backend,
                             fullgraph=True, dynamic=False)
    settings = dict(degree_tol=degree_tol, polish_steps=polish_steps)
    public = cubic_real_roots(*coefficients, **settings)
    unchecked = _cubic_real_roots_unchecked(*coefficients, **settings)
    actual = compiled(*coefficients, **settings)
    assert len(graphs) == 1  # Strict fullgraph capture, not Inductor performance.
    for result in (unchecked, actual):
        assert torch.equal(result.degree, public.degree)
        assert torch.equal(result.valid, public.valid)
        torch.testing.assert_close(result.roots, public.roots, equal_nan=True, rtol=0, atol=0)
        torch.testing.assert_close(result.residual, public.residual, equal_nan=True, rtol=0, atol=0)
