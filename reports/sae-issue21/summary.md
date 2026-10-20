# TorchYAND Issue 21 — CPU/CUDA checks PASS; GPT acceptance pending

Algorithm baseline `8ba99c4448e5494290fbfa0f8e12120115dcb58b` unchanged; adapter/tests/design executed from `c9b58d228de38c2f04443f0495d55ab26797afed`; SAE runtime `8081b6e02eee0754903898d5afe5b112f1bb23d0` and CLI0.7.4. The adapter commit adds only experiments and focused tests. All 49 algorithm Python blobs plus LICENSE match the original baseline and are independently hash-verified when staging. Runtime scripts/license were exported as exact adjacent SAE blobs to a separate local tool directory, without editing or vendoring controller internals. Exact adapter commit is source_sha; algorithm_source_sha remains the requested baseline, not a floating main.

## CPU gate and actual GPU result

Local CPU: Python3.12.3 / PyTorch2.11.0+cpu, two threads. Four focused tests PASS (including gradient corruption rejection, directional finite differences and source mismatch rejection). Raw reports/log are in local-cpu/. Actual driver gradient max absolute error: float32 **3.7253e-9**, float64 **3.46945e-18**; objective errors zero for this seed. Known quadratic converged in one accepted step to (.25,.75), with zero simplex and independent KKT residual in both dtypes. This gate completed before any provider allocation. Only the declared PyTorch dependency and its resolved dependencies were installed into an isolated local environment; no project dependency declaration changed. The optional NumPy warning is retained and no NumPy API was used.

Separate no-allocation preflight PASS at 18:02:01–18:02:04 UTC. First real attempt PASS; no second allocation or retry. Observed **Tesla T4**, CUDA tensor **cuda:0**, compute capability **7.5**, memory **15,637,086,208 bytes**; remote Python **3.13.15**, PyTorch **2.11.0+cu130**, CUDA **13.0**, two CPU threads, TF32 disabled. Mandatory default Kernel marker verified before scientific driver. No mocked provider, CPU-as-GPU fallback, core algorithm refactor, TPU or compile.

| Colab check | float32 | float64 |
|---|---:|---:|
| CPU gradient max abs error vs independent Autograd | 1.86265e-9 | 1.38778e-17 |
| CUDA gradient max abs error vs independent Autograd | 9.31323e-10 | 1.38778e-17 |
| CUDA objective max abs error vs reference | 0 | 0 |
| Independent quadratic KKT / simplex sum residual | 0 / 0 | 0 / 0 |

Tolerances and math assumptions were committed before the live run in experiments/sae_mvsk/README.md. Every recorded scaled error ratio is ≤1. This is a small deterministic test, not proof of general nonconvex optimality or all inputs. Only the convex quadratic has a known global optimum; the benchmark measures the existing MVSK oracle, not full-solver speed.

## Same-Colab-host eager timings

T2048 / n256 / batch256 / float32; public value_grad includes validation and fresh projection. Same data generated on CPU and transferred to GPU. Each scope has first-call-for-shape, two warmups, seven raw repetitions, median/IQR/min/max. CUDA synchronized before/after each GPU measurement. Full-batch CPU/GPU output parity max abs errors: objective **7.27596e-12**, gradient **5.82077e-11**. No timing includes generation or centering.

| Scope | Median ms | IQR ms | First shape call ms |
|---|---:|---:|---:|
| CPU resident (2 threads) | 12.019641 | .439554 | 23.106046 |
| GPU resident compute-only | 1.101796 | .033761 | 38.888449 |
| GPU transfer + oracle construction + compute + result return | 2.206174 | .049415 | 2.533774 |

Observed median CPU/GPU ratios **10.909×** resident and **5.448×** with transfers/construction/results. These apply only to this input, eager public operation, host and measured scopes. GPU setup/transfer separately **1.697ms**, CUDA initialization after availability probe **215.209ms**; first shape call is not pristine process-cold. No universal or full-optimizer speedup claim. CPU model, thermal state and clock frequencies were not recorded; seven repeats and one shape are limited evidence. Raw repetitions remain in the original downloaded artifact.

## Integrity and resource release

Downloaded mvsk-science.json: **7365 bytes**, SHA-256 **bcfc6efcc45e29f99a6d3479471b997de0881dceb810b7ac9e64433eabc6bf20**. Generic controller validated emitted length/hash/source/PASS; local verification additionally rechecked device, baseline/tool pins, numerical status and timing summaries. Controller classification TRANSPORT_ONLY_SCIENCE_UNREVIEWED and host transport label are preserved: neither independently certifies science. The project artifact contains the numerical evidence; GPT decides scientific acceptance.

One real new, named stop exit0, controller authenticated usage zero assignments, and independent fresh authenticated usage zero at **18:03:20.577665 UTC**, with positive remaining existing balance. Zero-state proof is at those timestamps, not continuous monitoring. No status/sessions polling. Host monotonic duration **30.775s**, controller **29.026s**, scientific driver **8.776s**, allocation lifecycle **23.995s**, cleanup **3.207s**. UTC span differs from monotonic by about1.492s during cleanup; cause unverified, original records retained, no fabricated clock correction. Performance samples use perf_counter and do not depend on UTC subtraction.

Bounds per attempt task600/allocation420/cleanup60/kernel30/work240/new120 seconds; external host ceiling660 seconds; maximum two sequential hosts1320 seconds. The second attempt and retry gate were not needed after first success. Unknown cleanup would stop all retries. Existing prepaid capacity only, no new subscription/top-up/billing setting, paid commitment, package installation on GPU, service, Actions or recurring process.

## Preparation failures and limits

Initial temporary /tmp venv was absent on a later WSL invocation (cause unknown); recreated from cached approved wheels in a user-cache environment. Initial Windows CRLF checkout failed source hash verification; restored exact baseline Git bytes, no committed algorithm change. CPU negative tests deliberately fail corrupted gradient/source. These local preparation problems were resolved before provider dispatch, not hidden as scientific passes. This is not a rerun of the historical1269-test suite. Small tolerance-bounded cases and one benchmark shape do not establish broad scientific readiness. No confidential payload, token, raw OAuth/session URL, private root or vendor debug log is published.

Source/report hashes and independent validation receipts are in manifest.json. Execution handoffs0; final Engineer→GPT evidence handoff1. Task began17:53:09 UTC; GitHub PR creation is publication receipt. GPT reviews and merges independently; engineer does not self-approve scientific readiness.
