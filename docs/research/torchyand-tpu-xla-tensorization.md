# TorchYAND — TPU/XLA mathematical structure and compilation study (Scientist-approved scope)

> SAE Framework V3.3 research gate · **Scientific Mentor: GPT** · **Scientific reviewer: Human Scientist**  
> Target: `JohnnyZPiggy/TorchYAND` · [Research Issue #1](https://github.com/JohnnyZPiggy/TorchYAND/issues/1)  
> **Status: RESEARCH_READY — Scientist-approved kernel-first TPU/XLA research scope (2026-10-19); TPU allocation not authorized.**  
> Prepared 2026-10-19 · Reviewed TorchYAND source commit: `4d428c0e178c4f0d32e82e8e8ba497805ed66e67`  
> Consulted (but not installed/copied) SAE V3.3 governance source: `5d21248709d454ae7a6d3e2d657c494373a0f085`

## 1. Reference problem and scope

**This is a compiler/backend study of an existing mathematical implementation.** It does not propose a new portfolio objective, YAND search direction, convergence theorem, or default solver behavior. The research question is whether existing, matrix-free tensor kernels can execute correctly on TPU via PyTorch/XLA, and whether XLA compilation improves *same-device* steady-state performance enough to recover cold compilation cost.

[Existing mathematical design](../../TorchYAND_MVSK_Design_Spec.md), [MVSK kernels](../../torchyand/oracle/kernels.py), [projected Hessian](../../torchyand/operators/kernels.py), [fixed PCG](../../torchyand/linalg/kernels.py), and [full YAND driver](../../torchyand/solver/solve.py) are the reference implementation; the original YAND research paper was **not independently inspected** in this cycle, so this document does not certify equivalence to that paper.

For observations `R: [T,n]`, `mu: [n]`, centered `A=R-1 mu^T: [T,n]`, and portfolios `x: [...,n]` with `x>=0, sum_i x_i=1`, let `z=A x: [...,T]`. Sample central-moment objective:

```text
f(x) = -c1 (mu · x) + c2/T sum_t z_t^2
                       - c3/T sum_t z_t^3 + c4/T sum_t z_t^4.
c=[c1,c2,c3,c4] is a shared, finite, nonnegative [4] tensor in v0.1.
```

The existing oracle accepts arbitrary `x` (simplex restrictions apply to the optimization problem, not oracle evaluation). v0.1 public solver uses float32/float64 on CPU/CUDA; **TPU is not a supported/validated platform as of this study**.

### Research baseline and alternatives

- **Reference:** existing CPU float64 oracle, dense small-n finite-difference/autograd/reference checks where appropriate; existing CPU/CUDA tests and eager adaptive solver as regression controls. Historic README GPU numbers are **not TPU evidence**.
- **Candidate:** run exactly the existing unchecked tensor kernels on XLA tensors in an experiment-only adapter, with fixed shapes and fixed masked PCG iteration budget; compare same-region XLA lazy/default execution with explicitly compiled XLA.
- **Fallback:** stop at MVSK/HVP or projected Hessian if fixed PCG fails compilation/numerics. Retain the original production solver unchanged; never hide an eager/CPU fallback inside a purported compiled benchmark.

## 2. Structure inventory and dependencies

| Component | Shapes / algebra | Parallelism and compiler opportunity | Hard boundary |
|---|---|---|---|
| Projection & moments | `x[...,n] @ A.T[n,T] -> z[...,T]`; reductions over `T` | Batched matmul + elementwise polynomial + reduction | precision of TPU matmul/reduction; numerical cancellation |
| Gradient | `-c1 mu + A.T psi1(z)/T`, `[...,n]` | Two matrix products with cached projection | precision and finite checks in public wrapper |
| HVP | `A.T (psi2(z) * (A v))/T`, `[...,n]` | matrix-free, batchable | `psi2(z)` may be negative; no global SPD guarantee |
| Third action | `A.T (psi3(z) * (Au) * (Av))/T` | shared `A`, batched GEMM | stochastic sampling belongs outside compiled region |
| Projected Hessian | `P_S H(x) P_S v + lambda P_S v` | fixed-size masking/projection | depends on stable geometric projector and appropriate damping |
| Fixed PCG | `K` tensor-only masked updates | fixed Python loop bound; potential compiled unrolling/fusion | `K` influences compile size/cost; residual/curvature-sensitive |
| Adaptive PCG & full solver | iterative stopping, active-set and line-search branches | investigate only after kernels verified | `.item()`, Python `while`, host-dependent control flow, changing shapes |

### R2. Tensor-equivalent derivatives (real arithmetic)

Let `z_t=sum_j A[t,j] x_j` and `v: [...,n]`; define

```text
psi1(z) =  2 c2 z - 3 c3 z^2 +  4 c4 z^3
psi2(z) =  2 c2   - 6 c3 z   + 12 c4 z^2
psi3(z) =          - 6 c3     + 24 c4 z
```

By `∂z_t/∂x_i=A[t,i]`, the gradient and Hessian are

```text
(grad f)_i = -c1 mu_i + (1/T) sum_t A[t,i] psi1(z_t)
(H(x) v)_i = (1/T) sum_t A[t,i] psi2(z_t) (sum_j A[t,j] v_j)
(D^3 f(x)[u,v,.])_i = (1/T) sum_t A[t,i] psi3(z_t)
                                           (sum_j A[t,j]u_j)
                                           (sum_k A[t,k]v_k)
```

Thus `H(x)v = A.T [psi2(z) ⊙ (A v)]/T`; no dense `[n,n]` Hessian, `[n,n,n]` coskewness, or order-4 cokurtosis tensor is necessary. Existing `torchyand.oracle.kernels` implements these contractions directly; this study proposes **a different compilation boundary**, not new mathematics.

On an active simplex face, the projectors must preserve subspace membership and satisfy (up to numerical tolerance) `P_S^2=P_S` and `P_S^T=P_S` on regular geometries. The implemented projected operator is

```text
K_lambda v = P_S H(x) P_S v + lambda P_S v
```

on the level-set tangent space, with a masked zero action for empty subspaces. If `psi2(z)` becomes negative, `H(x)` may be indefinite even when all `c_i>=0`; a finite positive `lambda` does **not automatically certify** positive definiteness on every problem instance. PCG is mathematically justified only for an effectively SPD operator on its projected subspace; preserve true-residual, curvature and breakdown diagnostics. Negative or tiny curvature is a correctness event, not something to optimize away.

**Proof obligation for a future changed implementation:** algebraic equivalence of tensor contractions and masked recurrence in exact arithmetic *under stated projector/SPD assumptions*. Floating-point trajectories, termination and statuses may differ across devices: compare numerically with explicit tolerances; do not claim bitwise equivalence.

## 3. Complexity / memory / parallelism

For portfolio batch `B` (product of leading batch dimensions), one projection, HVP or third action is `O(B T n)` arithmetic with repeated `A: [T,n]`; cached `z` has `O(B T)` footprint. Without fusing, pointwise intermediates can add `O(B T)` traffic. Fixed PCG with `K` steps and matrix-free projected Hessian matvec is approximately `O(K B T n)` work, plus `O(K B n)` mask/projection/reductions. Baseline state memory can be `O(Tn+BT+Bn)` plus compiler temporaries. Compile time/graph size may grow materially with `K`; this is a hypothesis, not an observed scaling law.

Large `T,n,B` can amortize host overhead; tiny batches and repeated reductions may not. Moving data CPU<->TPU or testing CPU vs TPU with different execution boundaries invalidates isolated compile-speed comparisons.

## 4. Correctness and numerical study contract

1. **Keep the production math and default adaptive solver untouched** in the first phase. Use existing independent CPU tests for oracle/PCG, small-n dense or autograd checks, and CPU float64 as a scientific reference where meaningful.
2. Compare matched `value/gradient/HVP/third_action` tensors, `P_S` membership/idempotence, fixed PCG `x`, actual **unprojected** true residual `||b-Kx||`, per-item iteration count, breakdown/curvature flags, and status. For full solver only later, compare objective, KKT residual and termination status.
3. Inputs: deterministic CPU-generated returns/probes/masks then one-time transfer to device **outside** timed region; documented `seed` and independent batch shapes `B=1,16`; candidate `(T,n)=(256,16),(1024,128)` initially, scale to `(4096,256)` only after memory/time approval. For fixed PCG consider `K=8` first, `K=32` if safe. Include regular and masked-face cases, nearly singular problems, zero RHS, nonpositive curvature and numerical extremes.
4. Start with float32, with TPU matmul precision setting explicitly recorded (`default/high/highest`, if the installed XLA release supports it). `default` can use BF16 internal products on TPU; `high/highest` trade speed for higher accuracy. **Do not assume TPU float64 or any precision mode is available/equivalent to CPU double; test and report supported dtypes first.** Do not silently downcast or weaken a stopping criterion.
5. **Tolerances are not yet Scientifically approved:** begin by recording absolute/relative errors and `rtol/atol`, then propose case-by-case thresholds based on conditioning, `torch.finfo`, reference and XLA precision. Existing CPU float64 PCG `rtol=1e-10` is not automatically suitable for TPU float32 with BF16 multiplication.
6. A method is **correctness PASS** only for named shapes, dtype, precision mode, operator, seed, device and thresholds; numerical failure is reported separately from compiler/backend failure.

## 5. XLA compilation research and Build-vs-Reuse review

### Candidate backend options (documentation reviewed 2026-10-19)
- **REUSE:** existing TorchYAND pure tensor kernels and PyTorch/XLA maintained backend. No vendor-specific custom op, Triton/CUDA port, or new solver dependency in this phase.
- **XLA lazy baseline:** call the same isolated kernel on XLA tensors, explicitly `torch_xla.sync(wait=True)` to execute and synchronize. The default mode already traces and compiles lazily: **never label this 'uncompiled/eager' or attribute all improvements to compiling rather than to graph boundary/reuse**.
- **Explicit XLA compiler:** in a verified compatible `torch/torch_xla` pair, test `torch.compile(fn, backend="openxla")` and optionally `torch_xla.compile(fn)` with precise version, mode and failure records. `torch_xla.experimental.eager_mode(True)` is experimental/version-dependent; compare it separately if supported.
- **REJECT for first phase:** blanket `torch.compile` with the default Inductor backend on TPU; transplanting CUDA-specific `torch.cuda` synchronization into XLA; introducing new numerical algorithms to game a benchmark.
- Official sources: https://docs.pytorch.org/xla/master/learn/eager.html , https://docs.pytorch.org/xla/master/learn/api-guide.html , https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html .
- **Unknown:** actual Colab TPU availability, CLI TPU allocation, XLA runtime build, op coverage, precision-mode support, graph compilation and compile-cache/recompilation behavior. TorchYAND's existing `torch>=2.0` packaging lower bound does **not** assert compatibility with any `torch_xla` version.

### Measurement questions (future tests only; no runs performed)

- **Stage A: environment gate:** Codex performs local/static dependency and CLI feature inspection without allocating Colab resources. Report executable versions, exact proposed TPU type and a version-locked installation strategy. Actual TPU allocation needs prior Scientist-approved type, max concurrent sessions, time and quota.
- **Stage B: small TPU smoke:** with authorization, first verify `torch_xla` import, *real* device identification (TPU, not generic `xla:0` alone), tensor arithmetic, `sync(wait=True)`, op/dtype support, compiled no-fallback invocation, and cleanup.
- **Stage C: kernel correctness:** compare same source `value_grad` (including projection on both paths), `hvp`, `third_action`, `projected_hessian`; then PCG fixed including finish/true residual with matched argument shape.
- **Stage D: benchmark:** independently record environment, precision, source SHA, compile graph/specializations, cold first-call latency, compiler time if measurable, warmups, medians and p95 synchronized steady latency, per-case recompile count and max memory when available. Separate data prep/transfers from kernel-only timing. Compare **XLA lazy/default versus XLA explicit compiled on the SAME TPU and exact numerical workload**; CPU/GPU baselines are supplementary absolute-performance comparisons, not compiler-speedup denominators.
- **Stage E (conditional):** if kernels pass with repeatable benefit, design a subsequent project PR for full YAND orchestration, addressing `state.active.any().item()` and other tensor-dependent Python branching. That is a **separate mathematical/architecture gate** and may fail legitimately.

Break-even for repeated identical-shape calls exists only if `t_baseline > t_compiled`:

```text
N_break_even ≈ C_compilation / (t_baseline - t_compiled)
```

Where compilation includes all required graphs/specializations and steady-state timings are correctly synchronized. If no positive steady-state improvement or a correctness failure occurs, report `NO_SPEEDUP`/`BLOCKED` rather than fabricate a speedup.

## 6. Acceptance artifacts and evidence classification

Expected eventual outputs, **only if actually run and approved**, follow SAE compute protocol: `manifest.json` (version/SHA/device/precision/time/quota), `correctness.json` (named cases, tolerances, error), `benchmark.csv` (cold/warm/median/p95, compilation/recompilation, methodology), and `summary.md` (comparisons, failures and cleanup). Benchmark first has **no GitHub write privileges**, and safe publication is a separately authorized step after integrity/secrets checking. Reports must not include credentials or raw Colab tokens.

As of this document:
- Repo source inspected: **PASS**.
- Algebraic HVP/projector derivation from repository definitions: **DOCUMENTED, not device-verified**.
- Live TPU allocation: **NOT RUN**.
- XLA compatibility, true TPU identity, float32/float64 operator coverage: **NOT RUN**.
- CPU rerun and independent XLA numerical test: **NOT RUN**.
- Compile benchmark and speedup: **NOT RUN**.
- TPU allocation/compute budget: **NOT AUTHORIZED / LIMITS NOT SPECIFIED**.

## 7. Alternatives, risks, Scientist decision and handoff

- **Research decision:** `RESEARCH_READY` — Human Scientist explicitly confirmed PR #2 on 2026-10-19; [GitHub decision record](https://github.com/JohnnyZPiggy/TorchYAND/issues/1#issuecomment-6275813704). Approval covers the **existing-mathematics, kernel-first TPU/XLA research scope and evidence gates only**; it does **not** authorize TPU resource spending, unvalidated error tolerances, mathematical changes, or performance claims.
- **Mathematically established:** matrix-free derivative identities given the sample-moment objective, and fixed tensor recurrence code exists. Positive definiteness, TPU support and compile acceleration are **not established**.
- **Hypotheses:** larger regular batched contractions could suit TPU; repeated fixed-shape PCG calls may amortize compilation; changing tensor masks without changing metadata could reuse XLA graphs.
- **Open risks:** precision/indefiniteness/PCG breakdown, graph explosion at large `K`, sync and implicit XLA lazy compilation, XLA op/device unsupported cases, Colab CLI TPU allocation.
- **Scientist review:** initial unchanged-math TPU kernel research scope **APPROVED** (2026-10-19). Actual TPU allocation remains **NOT AUTHORIZED** until hardware class, concurrent sessions, time and numeric quota/CU limits are separately approved.
- **Architect handoff:** Codex can inspect CLI / compatibility offline and prepare a minimal non-running preflight independently; any change to numerical mathematics or formal TPU benchmark must await the relevant gate(s).
- **Next research question:** can fixed masked PCG achieve reliable correct true residual and curve checks under supported TPU matmul precision modes without recompiling across changed tensor masks?

*This is an approved research scope, not a validated TPU numerical implementation, experimental report, or authorization to allocate hardware.*
