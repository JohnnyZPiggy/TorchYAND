# Issue 21: bounded MVSK scientific pilot

Algorithm pin: `8ba99c4448e5494290fbfa0f8e12120115dcb58b`. Transport pin: SAE `8081b6e02eee0754903898d5afe5b112f1bb23d0`, CLI 0.7.4. Algorithm files are unchanged. The adapter's new commit is separately recorded as `source_sha`; `algorithm_source_sha` remains the exact original baseline. The self-contained driver verifies all 49 package files plus LICENSE against baseline Git blobs. Remote staging downloads only that exact public source archive (30-second socket timeout, 10 MiB cap), verifies allowed regular files and hashes, and imports it from a temporary directory. It does not pip-install or modify the algorithm. Real remote runs use the existing Colab PyTorch; unavailable dependencies fail closed.

## Numerical design, fixed before live allocation

Seed 20261021; raw returns 31×5, float32 and float64. The independent reference starts with raw returns, computes mean/centered moments and `-c1*mean.dot(x)+c2*E[z²]-c3*E[z³]+c4*E[z⁴]`, with c=(1,.8,.3,.2). Autograd differentiates this scalar. It never calls the analytic oracle, its data preparation or kernel code. Centering uses float64 and casts back, matching the documented data contract without reusing its implementation. Analytic value/gradient must satisfy abs(error)≤atol+rtol*abs(reference): float32 atol=3e-6/rtol=3e-5; float64 atol=2e-12/rtol=2e-10. Raw errors and maximum tolerance ratios are recorded.

For convergence, use the simplex quadratic f(t,1−t)=t²−.5t with unique solution (.25,.75), starting at (.1,.9). This exercises an accepted solver step, not just an initially optimal point. Exact logdet, max_iter=10, seed123, kkt_rtol=0, kkt_atol=2e-5(float32)/2e-10(float64). Require CONVERGED_KKT, at least one accepted step, objective decrease, simplex feasibility, agreement with the known solution and an independent Autograd stationarity residual ≤ that tolerance. Compare with the existing KKT diagnostic. Both solution coordinates are strictly interior, so gamma=mean(gradient) and residual=max|gradient−gamma| is full KKT for this case. This is a global-optimum claim only for this convex quadratic, not for general MVSK.

Regression tests perturb the analytic gradient to ensure the oracle check fails, verify the independent reference by directional finite differences, and reject changed source bytes. Run only the small pilot tests, not the entire historical suite:

```sh
python tests/test_sae_scientific_pilot.py -v
```

## Timing scope

One fixed shape: T=2048, n=256, batch256, float32; seed above. Time the existing public eager `value_grad`, including validation and fresh projection, on the **same Colab host** CPU and CUDA. CPU threads=2; TF32 disabled; float32 matmul precision highest. CPU and GPU inputs derive from the same CPU tensors, with full-batch output parity checked outside timing. No compile, TPU, autocast or algorithm change.

Each scope records its first call for this shape (not process-cold), 2 warmups and 7 raw timed repetitions, median, IQR, min/max. CUDA synchronization brackets all GPU timed calls. CPU resident excludes generation/centering; GPU compute-only excludes staging/transfers; GPU transfer-included copies centered data/coefficients/x, constructs the public oracle, evaluates and returns value/gradient to CPU. One-time resident setup/transfer is separate. CUDA initialization is measured after the availability probe and is not a pristine process-start measurement. Ratios are observations for these scopes only, not general speedup claims or full-solver benchmarks. No legacy compile-tax comparison.

## Execution and resource boundary

Local CPU gate uses an isolated existing Python environment with the project's declared PyTorch dependency, before any provider allocation. Set `SAE_CPU_ONLY=1`, `SAE_LOCAL_SOURCE` to the exact-byte baseline checkout, `SAE_CANDIDATE_SHA` to the adapter commit, and `SAE_ARTIFACT_DIR` to a fresh output directory, then execute driver.py. Keep raw Git bytes: CRLF checkout changes must not bypass hash verification.

Reuse the three adjacent SAE runtime files and MIT LICENSE exported verbatim from the exact framework commit to a separate local tool directory. Verify their hashes; do not edit/vendor controller internals. Run separate `--preflight-only`, then the generic controller with this committed driver/manifest and adapter `--source-sha`. Controller artifact SHA/source validation is additional to scientific checks. The project driver alone owns source staging, math tests and benchmark interpretation.

At most 2 sequential real attempts; each task600/allocation420/cleanup60/kernel30/work240 seconds, original new cap120 seconds. External host interrupt540 seconds plus60 cleanup grace, kill/reap5 and bounded emergency named stop/readback15 seconds each (host ceiling660 seconds). Combined live host processes bounded by1320 seconds; no overlap or frequent status/sessions polling. A second attempt needs previous controller cleanup_confirmed, successful stop, a separate fresh authenticated CLI0.7.4 usage zero count with positive balance, and a fresh run ID/output. Unknown cleanup, quota/permission denial, unsupported CLI or failed local CPU stops dispatch. No new paid commitment, installation on GPU, service, Actions or Publisher.

Real GPU success additionally requires CUDA available, actual device name/capability/memory and a CUDA tensor, both dtype numerical checks, bounded benchmark/parity, downloaded JSON SHA-256 and successful named stop plus authenticated zero readback. No CPU fallback may be labeled GPU. A scientific FAIL remains a failed required artifact and is downloaded before the controller fails its status check. Errors record safe codes/classes only. GPT independently accepts results and any merge; engineer never self-certifies scientific readiness.
