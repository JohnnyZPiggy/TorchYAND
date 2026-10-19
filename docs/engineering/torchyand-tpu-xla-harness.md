# Issue #5: CPU-first TPU/XLA experiment harness

Role: Codex Engineer, SAE V3.3. Baseline:
`087accddf28ee566363ae94c8422b5be60fc23f0`.
Refs [#5](https://github.com/JohnnyZPiggy/TorchYAND/issues/5), approved
[research scope](../research/torchyand-tpu-xla-tensorization.md), and
[offline compatibility candidate](issue-3-tpu-xla-offline.md).
Historical pending-research text in the offline report is superseded by the
Scientist's approval; resource allocation remains unauthorized.
Report follows the [SAE Engineering Report](https://github.com/JohnnyZPiggy/SAE-Framework/blob/5d21248709d454ae7a6d3e2d657c494373a0f085/templates/ENGINEERING_REPORT.md).
Status: CPU_VERIFIED / REVIEW; Architect acceptance pending.

## Build-vs-Reuse and scope

REUSE TorchYAND oracle, geometry, projected Hessian and masked PCG tensor kernels;
ADAPT only experimental orchestration and the separately invoked XLA backend.
No new solver, linalg package, default dependency, production numerical change,
CI workflow, SAE bootstrap or Colab orchestration. The harness contains no
provisioning, GitHub publication, SSH or remote installation commands.

## Numerical contract

`benchmarks/bench_tpu_xla.py` has five matched pure boundaries:
value_grad includes projection on both sides; hvp and third_action consume the
same cached z; projected_hessian preserves masking/damping; fixed_pcg calls
initialize, exactly K=8 masked steps, then finish with the unprojected true
residual. Changing numerical state is tensor arguments. There are no tensor
scalar reads or tensor-dependent Python branches inside these boundaries.
No finite/curvature/breakdown guards were removed.

CPU generates deterministic centered A[T,n], mu[n], coefficients[4], feasible
x[4,n], directions and projected b[4,n] with one zero RHS. Smoke T=256,n=32;
reuse T=1024,n=128 has 10 numerical states with a fixed face then 10 changed
masks with unchanged metadata. Seed defaults to 123. Each input dictionary has
a SHA-256 identity covering tensor names, shapes, dtypes and bytes. Oracle
coefficients are [1,1,.2,.1]; PCG control c3=0 with positive damping .1 is
convex on the projected space. Other problems need not be SPD.

CPU independent checks use torch.func derivatives of a separately written scalar
objective, explicit dense P/H/M and public fixed PCG against that dense operator.
Small-case tests also compare to the independent pseudoinverse solution. Public
MVSKOracle parity, true residual, subspace membership, all flags and K budget
are checked. Tests include scales 1 and .01, empty/singleton/irregular faces,
zero RHS, singular/negative operators and nonfinite RHS. K exhaustion and
curvature breakdown are diagnostics, not claimed convergence.

CPU double comparisons use rtol=2e-8 / atol=2e-9; boolean/integer outputs must
match exactly. TPU thresholds remain **PROVISIONAL**, not Scientist-approved:
5e-5/5e-6 for oracle/operator outputs and 5e-4/5e-5 for PCG. TPU will compare
against CPU float32 control (including exact status), and CPU float64 results
on identically quantized inputs. Float64 status is not equated to float32 status
because their existing stopping tolerances differ. Actual residual and stopping
tolerance are reported separately. No automatic tolerance relaxation.

## Support matrix and guardrails

| Path | Implementation / verification |
| --- | --- |
| CPU reference, float64, precision=default | Implemented; actual results below |
| CPU xla-compile/lazy or precision high/highest | Rejected before backend import |
| TPU reference or invalid counts/missing options | Rejected before backend import |
| TPU lazy / xla-compile | Implemented experimental path; NOT RUN, NOT device validated |
| TPU version/identity rejection tests | MOCK_ONLY; no real XLA import/initialization |
| GPU, full solver, XLA eager, openxla alternative | Outside scope |

TPU invocation requires existing allocated hardware (separate future approval),
PJRT_DEVICE=TPU, exact installed torch/torch_xla 2.8.0 candidate, matching imported
versions, actual TPU real-device strings and device_kind attributes, single
process/non-SPMD, and verified precision getter. Missing API/version/plugin or
identity fails nonzero, never CPU fallback. No claim that candidate 2.8.0 fits an
unobserved Colab VM. Python/ABI/libtpu/op compatibility remains unverified.
Official tagged API sources: [compile/sync/device](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/torch_xla.py),
[precision](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/backends/__init__.py),
[metrics](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/debug/metrics.py).

Future TPU modes run identical functions/shapes/precision on the same TPU.
Lazy already traces/compiles on sync; explicit uses xla.compile(full_graph=True).
Transfers/preparation and CPU checks are outside kernel timing. Cold first-call
includes compile and execution. Warm calls use sync(wait=True), retain output,
and record raw milliseconds, median/p95. CompileTime/counter snapshots are
preserved per state; aten:: fallback fails. Warm compile-sample deltas are
recorded, but graph reuse remains UNVERIFIED_REQUIRES_METRIC_REVIEW (zero samples
alone do not establish cache reuse). No speedup ratios are computed. Use fresh
process/output directories per mode/precision; no cache reset between reuse states.

## Reproduction and evidence

From the checkout root with Python 3.10+ and existing torch/pytest/numpy:

```text
python -m pip install -e ".[test]"
python benchmarks/bench_tpu_xla.py --device cpu --mode reference --precision default --case smoke --output-dir benchmarks/results/issue5-smoke
python benchmarks/bench_tpu_xla.py --device cpu --mode reference --precision default --case reuse --output-dir benchmarks/results/issue5-reuse
python -m pytest tests/test_tpu_xla_harness.py -q
python -m pytest tests/ -q
```

Each output directory must be new. No overwrite of an earlier run. manifest.json
contains source SHA/dirty flag, environment, seed, shapes, K, input hashes and
status. correctness.json contains actual checks/errors/PCG diagnostics.
summary.md labels device/mode and TPU NOT RUN for CPU. CPU emits no latency
files or synthetic TPU measurements. Future actual TPU-only timings.json holds
raw measurements and metrics. Failures return 1 and preserve a failure manifest;
argument errors return 2 before creating output. Git metadata commands have a
10-second timeout. This is a checkout experiment, not an installed console entry.

Local installation used `python -m pip install -e ".[test]" --no-deps` with
already installed dependencies; editable build/install exited 0. This validates
installation in the existing environment, not a fresh dependency resolver.
No CPU compiler or TPU runtime was invoked.

Tested implementation SHA: `e10d5ebe343bb09f063b2376c2be9860a5cb81a5`.
Both smoke and reuse manifests record source_dirty=false. Subsequent evidence
updates change this document only; implementation/test bytes remain identical.
Actual environment: Windows CPU, Python 3.14.6 (Anaconda), torch 2.14.1+cpu,
pytest 9.0.3, numpy 2.4.6. Interpreter: `D:\software\anaconda3\python.exe`.
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 were set for the actual runs.
One TorchScript FutureWarning concerns Python 3.14; no warning was suppressed.
This does not establish compatibility across all declared torch versions.

Actual commands used the interpreter above and the reproduction arguments,
with run-specific output directories `benchmarks/results/issue5-e10d5eb-smoke`
and `benchmarks/results/issue5-e10d5eb-reuse`. Captures are in the ignored
`benchmarks/results/issue5-e10d5eb-evidence/` directory. Genuine stdout excerpts:

```text
python -m pip install -e ".[test]" --no-deps
Successfully built torchyand
Successfully installed torchyand-0.1.0
Process exit code: 0

cpu/reference: PASS; reports: benchmarks\results\issue5-e10d5eb-smoke
Exit code: 0
cpu/reference: PASS; reports: benchmarks\results\issue5-e10d5eb-reuse
Exit code: 0

python -m pytest tests/test_tpu_xla_harness.py -q
24 passed, 1 warning in 5.23s
Exit code: 0

python -m pytest tests/ -q
1250 passed, 43 skipped, 1 warning in 70.73s (0:01:10)
Exit code: 0
```

Smoke: all five comparisons PASS, maximum absolute errors by boundary:
value_grad 1.39e-16, hvp 2.66e-15, third_action 9.77e-15,
projected_hessian 9.77e-15, fixed_pcg 2.22e-15. Smoke PCG iterations
[0,8,8,8], converged [true,false,false,false], no breakdown/negative curvature,
max_iter_reached [false,true,true,true]. True residuals approximately
[0,6.96e-5,6.37e-4,5.01e-4]. This is correct budget reporting, not convergence
of all systems. Reuse: 20/20 CPU states passed; this is numerical state coverage,
not evidence of XLA graph reuse. No CPU/TPU benchmark speedup was measured.

Actual local report SHA-256 identifiers (raw files remain ignored):

| File | SHA-256 |
| --- | --- |
| smoke/manifest.json | 8b457005f63729526936a0cd0755355fe8ec4346572b201098f928e9c8c494ea |
| smoke/correctness.json | 86af29c8b19b918cc2f8eaf17a138ae9314eace489f47ecbe3b63c2d4e114ed0 |
| reuse/manifest.json | 4b8afd2277a73ed2035d9e58a8deb4e5600bd383898333d775883c612d8cd886 |
| reuse/correctness.json | d30be6dc5e0ff33fccd715afd9c274c5a8a8e4c732bf1bf3c57850b610da7f7d |

Only manifest.json, correctness.json and summary.md exist in each CPU run
directory; there are no timing files. Production `torchyand/`, dependencies and
CI configuration match the baseline (local git diff empty). Whitespace check
passed. The 43 skipped tests are not counted as hardware coverage.

| Required state | Actual result |
| --- | --- |
| CPU_HARNESS | PASS |
| CPU_REGRESSION | PASS |
| TPU_DEVICE | NOT RUN |
| XLA_COMPILE | NOT RUN |
| TPU_BENCHMARK | NOT RUN |
| TPU_ALLOCATED | NO |

Scientific scope: RESEARCH_READY; no TPU tolerance approval inferred.
Architect review: pending PR review. No merge performed by Codex.

## Next action

Architect reviews exact implementation/tested SHA and CPU evidence. Any real
TPU smoke requires a new Scientist-approved variant, session/time and numeric
CU budget. This PR does not authorize hardware or establish TPU readiness.
