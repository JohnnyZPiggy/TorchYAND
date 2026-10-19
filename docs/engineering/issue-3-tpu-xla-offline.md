# Issue #3: TPU/XLA offline feasibility and experiment proposal

Inspection date: 2026-10-19 (Europe/Zurich). Role: Codex Engineer, SAE V3.3.
Refs [#3](https://github.com/JohnnyZPiggy/TorchYAND/issues/3);
research gate [#1](https://github.com/JohnnyZPiggy/TorchYAND/issues/1),
scientific draft [PR #2](https://github.com/JohnnyZPiggy/TorchYAND/pull/2).

## Decision and scope

**OFFLINE_PREFLIGHT_PASS** means the requested inspection and proposal are complete,
not that TPU execution is ready or verified. **TPU_ALLOCATED=NO;
TPU_BENCHMARK=NOT RUN; SESSION_COUNT=0** at the actual server checks recorded in
[CLI evidence](issue-3-cli-evidence.txt). No session was created, connected to,
installed on, restarted or stopped. No accelerator quota was requested.

Inspected source: `4d428c0e178c4f0d32e82e8e8ba497805ed66e67` (HEAD and
origin/main revalidated). Future execution must revalidate this algorithm SHA and
record the separately reviewed experiment implementation SHA. SAE governance was
consulted in its repository; no bootstrap or snapshot is adopted here. No solver,
dependencies or Actions configuration changes are part of this work.

| Feasibility item | Actual status | Meaning |
| --- | --- | --- |
| WSL2 / Colab CLI inspection | PASS | Ubuntu 24.04.5, CLI 0.7.4 |
| TPU selection / connection / release interfaces | DOCUMENTED | Help exposes TPU variants and stop; functionality NOT TESTED |
| Colab TPU device availability | NOT TESTED | No allocation attempted |
| PyTorch/XLA compatibility | PARTIAL | Official candidate version pair verified; actual runtime unknown |
| Remote installation / operator coverage | NOT RUN | No remote execution |
| Local CPU regression | PASS | See validation below; no XLA coverage inferred |
| Research / resource gate | PENDING | This report grants neither approval nor compute |

## Local tooling evidence

The adjacent text file contains actual selected command output, not reconstructed
benchmark results. WSL kernel: `6.18.54.1-microsoft-standard-WSL2`; system Python
3.12.3. `colab version` returns 0.7.4. `colab --version` fails (unsupported flag);
the supported subcommand is `version`. System `python3 -m pip show torch
torch-xla colab-cli` failed with `No module named pip`. A subsequent
`importlib.util.find_spec` check returned `torch=False, torch_xla=False` in that
interpreter. This says nothing about unrelated virtual environments.

Observed help interfaces (only help invoked):

- `new --tpu <str>` lists `v5e1`, `v6e1`; entitlement/availability varies.
- `exec --session <name> --file <path> --timeout <seconds> --env KEY=VALUE`
  describes execution in an existing session.
- `ssh --session <name> --tpu <variant> --rm` describes connection and optional
  release, but **may auto-create a missing session**. Do not use SSH as a probe.
- `stop --session <name>` describes explicit shutdown; `sessions` is the read-only
  server check. Actual shutdown is NOT TESTED because nothing was allocated.

`colab sessions` returned `No active sessions found on server` before and after
inspection. No `BLOCKED_TPU_CLI` is indicated by missing selection flags; actual
provisioning/connection may still fail later. No OAuth or credential files read.

## Version compatibility matrix

Use a deliberately pinned documented candidate, not an assertion of the newest
release. Official [v2.8.0 README](https://github.com/pytorch/xla/blob/v2.8.0/README.md)
specifies Python 3.11–3.13 and `torch==2.8.0` with `torch_xla[tpu]==2.8.0`;
[release notes](https://github.com/pytorch/xla/releases/tag/v2.8.0) also discuss
Python 3.12/3.13 builds. The README lists a Linux cp312 TPU VM wheel. This is
documentation-level VERSION_VERIFIED, not a successful resolver/import test.

| Component | Candidate / observed | Evidence status | Installation status |
| --- | --- | --- | --- |
| Candidate Python | Linux x86_64 Python 3.12; documented range 3.11–3.13 | VERSION_VERIFIED for release | Remote UNVERIFIED |
| Candidate torch | 2.8.0 | VERSION_VERIFIED pair in release README | WSL system NOT_INSTALLED; remote UNVERIFIED |
| Candidate torch_xla[tpu] | 2.8.0 | VERSION_VERIFIED pair in release README | Local NOT_INSTALLED; remote UNVERIFIED |
| TPU plugin | libtpu 0.0.17 via 2.8.0 extra | VERSION_VERIFIED in release setup.py | Remote UNVERIFIED |
| PJRT | TPU backend, `PJRT_DEVICE=TPU` | API/source verified, not a separately guessed Python package version | Runtime NOT RUN |
| Colab TPU VM/runtime | OS, architecture, Python ABI, preinstalled packages unknown | UNVERIFIED | NOT TESTED |
| Local WSL system | Python 3.12.3; torch/XLA absent | Observed interpreter only | No install attempted |
| Local Windows CPU reference | Python 3.14.6; torch 2.14.1+cpu; XLA absent | Observed, CPU only | Not the proposed XLA pair |

The [tagged setup.py](https://github.com/pytorch/xla/blob/v2.8.0/setup.py)
pins the TPU extra to libtpu 0.0.17. PJRT is the runtime interface, not a reason
to install an unrelated `pjrt` package. Future authorized setup must inspect
runtime Python/OS/ABI, resolve the exact pair in an isolated environment, record
wheel versions/hashes and `pip check`, then test import. If incompatible, stop
and revise the plan; do not overwrite the Colab environment speculatively.
The repository's `torch>=2.0` is insufficient to establish XLA compatibility.

Official `master` guides are moving API references, not release guarantees.
The [release index](https://github.com/pytorch/xla/releases) includes newer 2.9.0
notes changing libtpu installation; do not mix those instructions with the 2.8
extra. Recheck all pins before a later approved run. No remote installation was
performed and no installed-XLA compatibility claim is made.

## Static source audit

All paths below refer to the inspected SHA. This is static eligibility, not XLA
lowering proof. Matmul, reductions, vector_norm, isfinite, where, boolean/int64
state and broadcasting require an actual coverage test on the selected release.

| Boundary | Audit finding / proposed treatment |
| --- | --- |
| `torchyand/oracle/kernels.py` | `value_grad`, `hvp`, `third_action` are tensor math; no `.item()`, CUDA calls or tensor-dependent Python branch. Include projection in both value_grad modes; HVP/third_action consume identical precomputed z. |
| `torchyand/operators/kernels.py` | Projected HVP uses level_tangent, where and tensor damping; free_count/level_regular stay tensor arguments. No host scalar reads. |
| `torchyand/linalg/kernels.py` | initialize/step/fixed/finish are masked tensor recurrence; fixed uses static Python K. finish recomputes true unprojected residual. Python callback/optional configuration branches are static. |
| `benchmarks/bench_pcg_compile_reuse.py:42` | reusable_fixed_pcg already passes changing b/z/A/c/mask/nu/count/regular/damping explicitly; keep K static. Do not capture numerical state in a long-lived closure or change tensor metadata between reuse calls. |
| `benchmarks/bench_mvsk_compile.py` | CPU/CUDA enumeration, `.cuda()`, CUDA metadata, CUDA-oriented synchronization, default compile backend; public eager validation differs from compiled boundary. Replace orchestration in a separate experiment harness; compare identical pure functions. |
| `benchmarks/bench_pcg_compile.py` | Adaptive branch has `.any().item()`; use fixed loop only. Existing compile backend is Inductor. check/diagnostics contain scalar reads outside core. |
| `benchmarks/bench_pcg_compile_reuse.py` | CountingInductor counts Dynamo/native Inductor events, not XLA compilation. Logging damping `.item()` and correctness scalar reads belong after sync, outside timing/capture. Metadata/geometry assertions are CPU preparation. |
| Shared benchmark helpers / full solver | CPU/CUDA sync helpers do not provide XLA barriers. Solver `.any().item()` host control remains outside scope. No full solver TPU claim. |

Do not remove PCG finite, curvature or breakdown protections for compilation.
Fixed K may unroll to a large graph; begin K=8, then K=32 only within a later
approved budget. Damping is not a proof of positive definiteness on arbitrary
MVSK states. Include negative-curvature and rank-deficient-face cases as
diagnostic tests, not convergence successes.

## Proposal: `benchmarks/bench_tpu_xla.py` (not implemented or executed)

This section is the concrete minimal experiment design requested by #3. There
is intentionally no executable TPU launcher or benchmark CSV in this PR.
Implement the harness only after architecture/research review; resource use
requires separate approval. Reuse frozen kernels without modifying mathematics.

Proposed explicit CLI contract:

```text
--device {cpu,tpu}                   required; no auto/all/fallback
--precision {default,high,highest}   required; float32 matmul precision
--mode {reference,lazy,xla-compile}  required
--case {smoke,reuse} --seed 123 --warmup 3 --repeats 20
--output-dir <run-specific-directory>
```

`cpu` permits `reference` only, performs float64 numerical checks, and emits no
TPU latency/speedup. `tpu` permits lazy or xla-compile only, float32 tensors, one
process/device; no SPMD, BF16 tensor or float64 TPU assumption. Reject conflicting
options and invalid counts before device initialization. Generate the same
seeded float64 data on CPU, quantize once to float32, and transfer identical
inputs to each baseline. Reference both the quantized inputs promoted to
float64 (backend error) and original float64 inputs (quantization error).

Required hard-failure sequence for the future TPU path, with no try/except
fallback: before importing XLA require environment `PJRT_DEVICE=TPU`; import
matched torch/XLA and check exact approved versions; require
`torch_xla.runtime.device_type() == 'TPU'`; obtain device and require XLA type;
require `torch_xla.real_devices()` identifies TPU and runtime device attributes
record the approved hardware class. Missing API, plugin, identity or mismatch
must return nonzero. Generic `xla:0` or an environment variable alone is
insufficient. Then require an arithmetic smoke and `torch_xla.sync(wait=True)`.
No provisioning code belongs in the harness.

The tagged [compile/sync source](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/torch_xla.py)
contains `torch_xla.compile(fn, full_graph=True)` and sync(wait=True);
[runtime source](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/runtime.py)
supports PJRT/device attribute inspection. `torch_xla.backends` precision APIs
must be checked in the installed release and fail if missing. Record the
getter value and relevant XLA environment; use fresh processes for each precision.

Workload table (fixed metadata per case, CPU data preparation untimed):

| Case | T, n, B / RHS q | K | Data / checks |
| --- | --- | --- | --- |
| Smoke kernels | 256, 32, 4 | n/a | Seed 123, coefficients [1,1,0.2,0.1], feasible x; random u,v; value_grad with projection, hvp, third_action |
| Smoke projected PCG | 256, 32, q=4 | 8 | c3=0 convex control, nonnegative c2/c4, positive damping; zero and nonzero RHS |
| Reuse (conditional) | 1024, 128, q=4 | 32 | 10 changing numerical states fixed mask then 10 changing masks; same tensor shapes/strides/dtypes; reuse one callable |

Use normalized synthetic returns (scale=1 for derivative coverage; separately
scale=0.01 for portfolio-like cancellation). CPU geometry preparation verifies
feasible x/mask membership and constructs matching nu/count/regular, RHS and
damping. Transfer all state as tensors. Include empty/singleton faces, irregular
level geometry and an indefinite control; evaluate CPU status as expected,
never demand convergence for the indefinite control.

For small cases independently check derivatives using autograd and projected
Hessian against an explicit CPU basis/dense matrix. Compare all PCG outputs,
true residual `||b-Mx||`, relative residual, face leakage, iterations, converged,
breakdown, nonpositive_curvature and max_iter_reached. A short K budget can
legitimately reach max_iter. Status differences must be reported, not hidden by
a loose solution tolerance.

Provisional numerical acceptance for Scientist/Architect review: at highest,
kernel rtol=5e-5, atol=5e-6; PCG solution/residual rtol=5e-4, atol=5e-5, exact
boolean/status agreement away from stopping thresholds. Record per-output
maximum absolute error and relative norm error against CPU float64. high/default
are exploratory: apply the same thresholds and record failures, do not loosen
after seeing results. Also report observed true residual against the existing
float32 PCG tolerance (1e-5 times RHS norm), independently of comparison tolerance.
Threshold-boundary disagreement blocks speedup acceptance and needs investigation.
These tolerances are proposals, not device-validated scientific acceptance.

### Same-TPU timing and recompilation

Baseline A: default XLA lazy execution of each pure function. It already traces
and compiles at synchronization, so never label it uncompiled eager. Baseline B:
the identical callable wrapped by `torch_xla.compile(..., full_graph=True)` on
the same TPU and precision. `torch.compile(backend='openxla', fullgraph=True,
dynamic=False)` is a separately named optional variant after release verification,
not a fallback. Experimental XLA eager mode is optional and excluded initially.
See the official [mode guide](https://docs.pytorch.org/xla/master/learn/eager.html).

For each mode/case use a fresh process, document persistent-cache settings, sync
transfers before timing, retain output until after the completion barrier:
sync(wait=True), start monotonic wall clock, invoke, sync(wait=True), stop.
First invocation includes tracing/compilation/execution; report as cold first
call, not pure compiler time. Separately collect XLA CompileTime and ExecuteTime
metrics if available; unavailable metrics remain UNKNOWN. Warm up 3 calls, time
20 synchronous calls, report median/p95 and all raw samples. CPU reference,
host preparation, transfer, printing and result copies are outside timing.

Record XLA metrics before/after first call, warmup, each changed state and timed
phase. Inspect CompileTime sample deltas/graph metrics and CPU fallback counters
(aten counters), preserving names and values. Unexpected lowering/fallback fails
the case; there is no CPU fallback performance result. Do not infer recompile
counts from latency or CountingInductor. If trustworthy metrics are absent,
mark reuse UNKNOWN and block a reuse/speedup claim. Distinguish recompiles while
changing state from initial compilation. No cache resets between reuse states.

CPU is a numerical reference only. A speedup denominator must be lazy time on
the same TPU/case, contingent on both modes passing correctness. Report cold
overheads for both; incremental break-even uses their cold-cost difference and
positive warm-time saving, otherwise NO_SPEEDUP/no finite break-even claim.
Precision semantics and APIs are documented in the official
[API guide](https://docs.pytorch.org/xla/master/learn/api-guide.html); precision
is a separate experimental axis, not merely the tensor dtype.

## Future bounded resource proposal — NOT AUTHORIZED

Candidate: one Colab v5e1 session, one process, no other concurrent session,
maximum 20 minutes including installation and compilation; no retries or upgrade
to v6e1. Nominal cap is 1/3 accelerator-session-hour. Colab compute-unit cost is
UNKNOWN offline: future approval must specify a numeric CU cap after a read-only
rate/balance check, and enforce the smaller time/CU limit. This proposal does not
grant spending authority. If budget/identity cannot be verified, do not allocate.

After explicit Scientist resource approval and relevant research review, use
the pinned implementation SHA, run smoke only first, stop on failure/budget
boundary, and release the exact session in success/failure/interruption cleanup.
Verify `colab sessions` after `stop --session <owned-name>`; report any cleanup
failure immediately and do not stop unrelated sessions. A local watchdog must
enforce the session cap, including a stuck remote installation/compile.

Future actual runs must produce manifest.json (source + harness SHAs, package
versions, hardware/PJRT/precision, commands, seed, time/budget), correctness.json,
benchmark.csv and summary.md only from executed cases. Benchmark execution has
no GitHub writes. Download, verify SHA-256 identity and review secret-free
artifacts before separately authorized publication. This offline PR is not that
runtime or publication verification.

## Validation and handoff

Local CPU regression command: PowerShell sets OMP_NUM_THREADS=1 and
MKL_NUM_THREADS=1, then runs `D:\software\anaconda3\python.exe -m pytest tests/ -q`.
Result: **1226 passed, 43 skipped, 1 warning in 71.06s** (exit 0).
The warning reports TorchScript limitations on Python 3.14; the skipped cases
are not counted as TPU coverage. No native compilation or TPU/XLA
compilation was attempted. This documentation proposal introduces no executable
logic and no new dependencies. Source and documentation diff checks are local.

Next action: Architect reviews this offline report and PR #2 with the Scientist;
approve/revise the experiment contract before implementation or requesting a
separate numeric TPU resource budget. Keep #3 open via `Refs #3`, not `Closes`.
