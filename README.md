# TorchYAND

PyTorch implementation of the YAND-MVSK algorithm for higher-moment portfolio
optimization. Version 0.1.0 provides a matrix-free numerical core and an eager
solver for simplex-constrained portfolios.

## Features

- MVSK objective, gradient, Hessian-vector products, and third-order actions.
- Matrix-free projected Hessian operators and batched PCG.
- Stochastic and small-dimension exact logdet correction.
- Affine-normal directions, active-set simplex constraints, exact quartic line
  search, and KKT diagnostics.
- CPU/CUDA, float32/float64, and leading batch dimensions on portfolio tensors.

See [the design specification](TorchYAND_MVSK_Design_Spec.md) for the mathematical
definitions. Internal unchecked kernels are not the supported user API.

## Installation

From a local checkout:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

Python 3.10 or newer is declared in package metadata. The declared PyTorch
minimum (`torch>=2.0`) still needs validation; it is not a claim that every
allowed version has been tested. This project does not currently claim PyPI
availability. Repository: [JohnnyZPiggy/TorchYAND](https://github.com/JohnnyZPiggy/TorchYAND).

## Quick start

```python
import torch
from torchyand import prepare_mvsk_data, MVSKOracle
from torchyand.solver import (
    YANDOptions,
    solve_yand,
    SolverStatus,
)

rng = torch.Generator().manual_seed(123)

returns = 0.01 * torch.randn(
    256,
    16,
    dtype=torch.float64,
    generator=rng,
)

data = prepare_mvsk_data(returns)

oracle = MVSKOracle(
    data,
    coefficients=[1.0, 1.0, 0.2, 0.1],
)

x0 = torch.full(
    (16,),
    1 / 16,
    dtype=returns.dtype,
)

result = solve_yand(
    oracle,
    x0,
    options=YANDOptions(
        logdet_mode="stochastic",
    ),
    seed=123,
)

print(
    SolverStatus(result.status.item()).name,
    result.converged.item(),
)

print(result.x)
print(result.value.item())
print(result.kkt_residual.item())
```

`prepare_mvsk_data` returns `MVSKData`. The oracle and preprocessing API are
available from `torchyand`; `YANDOptions`, `solve_yand`, `YANDMVSK`, `YANDResult`,
and `SolverStatus` are available from `torchyand.solver`. `YANDMVSK` is a
configuration wrapper around the same solver.

## Solver status

A returned result does **not** imply convergence. Inspect `result.converged`
and `result.status`; `result.reason` provides additional diagnostics.

| Status | Meaning |
| --- | --- |
| `CONVERGED_KKT` | Full simplex KKT stopping condition is satisfied. |
| `STALLED` | The progress or face-update policy terminated the solve. |
| `MAX_ITER` | The outer iteration budget was reached. |
| `DIRECTION_FAILED` | Direction construction failed, including its permitted fallback. |
| `LINE_SEARCH_FAILED` | The feasible line search failed. |
| `INVALID_STATE` | State validity or an orchestration invariant failed. |

Face stationarity alone is not full simplex convergence. Batched result fields
retain the input batch dimensions, and individual items can stop differently.
Convergence is not a guarantee of a global optimum for a nonconvex objective.

## Coefficients

`MVSKOracle(data, coefficients)` accepts a finite, nonnegative sequence or
tensor of shape `[4]`, ordered **`[c1, c2, c3, c4]`**:

```text
mu = mean(returns, dim=0)
A = returns - mu
z = A @ x
f(x) = -c1 * dot(mu, x)
       + c2 * mean(z**2)
       - c3 * mean(z**3)
       + c4 * mean(z**4)
```

These are sample central moments, not standardized skewness or kurtosis.
Coefficients are shared across a portfolio batch; batched coefficient arrays
are not supported in v0.1.

## Precision and devices

Float32 and float64 are supported on CPU and CUDA. Portfolio and direction
tensors must match the oracle data's dtype/device and have shape `[..., n]`.
Only leading batch dimensions broadcast; the asset dimension must agree.
For CUDA, move returns to CUDA before preprocessing and create `x0` on the same
device. Float64 is recommended for strict convergence studies.

Preprocessing centers in float64 by default and preserves the input dtype in
its output. Numerical trajectories can differ across devices and precisions;
bitwise CPU/CUDA reproducibility is not promised. Do not mutate oracle data,
coefficients, or cached tensors in place during use.

## Performance

Eager end-to-end GPU execution can be slower than CPU for small single solves
because the orchestration is launch/synchronization heavy. Compiled fixed PCG
has achieved strong steady-state kernel speedups in separate experiments.

In one validated representative repeated fixed-shape Tesla T4 experiment,
end-to-end solve time decreased from about **22.9 s to 9.6 s (~2.38x)**. Cold
compilation cost about **11 minutes**, with break-even around **51 repeated
solves**. These measurements describe that experiment, not a general speed
guarantee.

Compiled PCG is an **experimental benchmark path**, not the default solver
behavior. The eager solver and its default adaptive PCG remain unchanged.

## Validation

**1269 tests passed on the validated Colab environment: PyTorch 2.11.0+cu130,
Tesla T4.** This does not validate all PyTorch versions. CPU GitHub Actions
targets Python 3.10 and 3.13; tests do not require CUDA or native compile code
generation. Compatibility claims depend on those checks actually passing.

## Benchmarks

Run scripts from the checkout root. Generated benchmark results are ignored
and should not be committed.

| Script in `benchmarks/` | Purpose |
| --- | --- |
| `bench_cubic.py` | Cubic solver latency and throughput, with optional compile modes. |
| `bench_quartic.py` | Quartic interval minimizer baseline and compile experiments. |
| `bench_mvsk.py` | Eager oracle and cache costs. |
| `bench_yand_solver.py` | Full eager solver pipeline. |
| `profile_yand_solver.py` | CPU/CUDA solver operation profiles and optional traces. |
| `bench_mvsk_compile.py` | Optional compiled MVSK tensor kernels. |
| `bench_pcg_compile.py` | Adaptive/fixed eager versus compiled fixed PCG core. |
| `bench_pcg_compile_reuse.py` | Graph reuse across changing numerical states and masks. |
| `bench_yand_compiled_pcg.py` | Optional scoped compiled-PCG end-to-end experiment. |

Compile experiments require an appropriate native compiler/toolchain and
report failures without silent fallback. They are not run in CI.

## Project structure

```text
torchyand/
  data/          # Return preprocessing
  oracle/        # MVSK objective and derivatives
  polynomial/    # Real roots and quartic minimization
  geometry/      # Projectors and affine-normal direction
  operators/     # Projected Hessian
  linalg/        # Batched PCG
  logdet/        # Exact/stochastic corrections
  constraints/   # Feasible step cap and active sets
  diagnostics/   # KKT diagnostics
  line_models/   # MVSK line coefficients
  line_search/   # Exact feasible line search
  solver/        # State machine and solver driver
tests/           # Independent references and correctness tests
benchmarks/      # Performance experiments
```

## License

MIT; see [LICENSE](LICENSE).
