# Issue #10: offline TPU Safe Adapter

Codex Engineer / SAE V3.3. Refs [#10](https://github.com/JohnnyZPiggy/TorchYAND/issues/10)
and parent [#7](https://github.com/JohnnyZPiggy/TorchYAND/issues/7).
Dispatch baseline `f143e57824647ce95a0ab8db65dcc92f279e376b` was fetched and
pinned. Parent #7 remains OPEN/BLOCKED. This document requests code review,
not resource authorization or a live-enablement decision.

## Build-vs-Reuse and source review

REUSE PR #9's separate-process monitor, atomic local JSON writing and parent
liveness detection; REUSE standard-library subprocess/JSON/dataclasses.
ADAPT only mock transport, approval/ownership gates and endpoint cleanup.
No new solver/dependencies, HTTP client, agent framework, scheduled task,
production algorithm change or benchmark arithmetic change.

Reviewed official **tag v0.7.4**, without invoking any installed CLI:

- [session.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/commands/session.py):
  runtime selection can map a truthy unrecognized TPU selector to v6e1. The
  adapter accepts exactly lowercase v5e1 and no fallback. Named new/stop use
  explicit -s. Assignment listings may expose an orphan as local name ?.
- [run.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/commands/run.py):
  implicit teardown can hide failures; its timeout is not an end-to-end budget.
  This adapter never plans colab run or ssh and does not trust termination text.

No Colab command, account API, credential import/refresh or TPU/XLA operation
was executed in this Issue. Prior private read-only CU evidence is sufficient:
prospective v5e1 CU rate remains UNKNOWN. No fresh balance/rate query occurred.

## Public entry and authority model

`tools/tpu_safe_adapter.py` defaults to a dry-run plan with LIVE_ADAPTER_ENABLED=NO.
`--live` is unconditionally rejected before subprocess creation, even when
records exist. No force flag, environment bypass, stored approval token, real
command dispatcher or hidden live launch mode exists. Child subprocesses use
a constant Python entry into local fake/watch functions, not a real CLI.

Frozen ProbeConfig requires device=v5e1; one owned session; first-probe proposal
at most 5 minutes; positive finite numeric CU cap and rate upper bound; rate
provenance; zero allocation retries; exact reviewed 40-hex source SHA; unique
session, owner and allocation correlation; deadline; approval reference.
Boolean budgets/counts, device typos/mixed case, missing fields, stale/unbounded
deadline, unreviewed SHA, absent approval/risk acknowledgment and insufficient
CU envelope fail before command transport. Approval validation compares the
entire record and matches source/budget/hardware to configuration.

**All accepted approval/rate references in this offline model are synthetic
mock:// records.** Their presence is not authentication of a human decision.
Real HTTPS approval references are deliberately rejected; no real approval
verifier is implemented. A future separate enablement PR must add independently
reviewed human-decision verification, actual rate provenance, private CLI mapping
and runtime probes. Numeric mock CU/rate=1 is fixture data, not a rate quote or
spending authorization. Global account-concurrency/billing safeguards are not
established by the per-owned-session mock count.

## Command seam and lifecycle

Future-shaped argv are allowlisted **data only**:

```text
colab version
colab sessions
colab new -s <exact-owned-name> --tpu v5e1
colab exec -s <exact-owned-name> --file tiny_identity_probe.py --timeout 1
colab exec -s <exact-owned-name> --file tiny_probe.py --timeout 1
colab stop -s <exact-owned-name>
```

Probe filenames describe future transport seams; these remote scripts are NOT
shipped or deployed. The identity/probe mock returns structured fixture data
and a genuine tiny local dot product (11), not TPU computation. Executed argv
always begin with the current Python interpreter and constant local child code.
The planned `colab ...` tuple is serialized into that fake invocation and checked
for exact equality. It is never handed to subprocess as the executable command.
All waits are bounded, shell=False, stdin closed, with no retries/fallback.
Injected runners receive the fake command only. Transport errors are reduced to
types/fixed messages; raw stdout/stderr is not copied into public error logs.

Mock lifecycle:
PRECHECK -> APPROVAL_CHECK -> NAMED_ALLOCATION_PENDING -> ALLOCATED ->
DEVICE_VERIFY -> TINY_PROBE -> STOP_OWNED -> VERIFY_RELEASE -> RELEASED.
BLOCKED, ALLOCATION_UNCERTAIN and CLEANUP_FAILED/MANUAL_ATTENTION_REQUIRED are
explicit alternatives. RELEASED describes owned mock cleanup, not probe success:
the separate outcome must also be MOCK_PROBE_PASS for CLI exit 0.

Journal is stored exclusively in a new local OS temporary directory (outside
synced checkout), mode 0700 directory / 0600 files where supported. It records
schema, immutable configuration, owner/name/correlation, exact source SHA,
wallclock/monotonic start, total deadline, phase/history and returned endpoint.
Before simulated allocation, pending identity is saved with endpoint unset.
Writes flush/fsync before atomic replacement; this requests OS durability,
not proof against power loss. Windows ACL privacy/directory durability remain
unverified. No bearer/proxy/OAuth token or account balance is stored.

An independent watchdog starts and acknowledges readiness before the fake
allocation. The accepted guard's wait_for_trigger loop is reused. Mock total
budget is 4 seconds (not a real 5-minute approval), with 1.5 seconds reserved
for cleanup and each command bounded by remaining time and at most 0.5 seconds.
Allocation/identity/probe cannot consume cleanup reserve. No retries, extensions
or second allocations. Parent finally provides secondary cleanup if the
watchdog exits; that result explicitly marks watchdog FAILED/manual attention.
It does not certify the failed primary safety mechanism as healthy.

## Uncertain allocation and owned cleanup

Returned identity is persisted only after exact mock owner/correlation/source/
name/endpoint validation. Lost response or allocation timeout never causes a
second new call. Reconciliation reads a fresh fake backend list and requires
exact matching provenance. With one provable match, cleanup can proceed; absent,
duplicate, orphaned or conflicting identity fails ALLOCATION_UNCERTAIN/manual
attention without guessing. Names alone and generic first list entries cannot
establish ownership.

**Important current CLI gap:** v0.7.4 new generates its allocation UUID internally;
the documented CLI listing does not expose our mock owner/correlation/source
fields. The fake reconciliation proof is NOT evidence that the real CLI can
provide it. No real text parser or endpoint mapping adapter is claimed here.
Resolving that gap requires separate reviewed work; no hidden HTTP API workaround.

For known endpoints, private journal identity and fresh server-backed identity
must agree. Lost local name/orphan mapping blocks destructive action, even if
the endpoint is recognizable. Stop is one exact named call; fresh endpoint
absence, not exit code or printed success, establishes mock release. Unknown
schema, wrong ownership, forged cached result, endpoint drift, credentials/API
errors, timeout or still-listed endpoint fails closed. Terminal results are
idempotent for validated identity; repeated verification errors cannot start
unbounded retry loops. Other synthetic sessions remain unchanged.

Human-only fallback for a future real orphan: inspect **Colab UI Manage sessions**
and confirm ownership before manually releasing the exact known assignment,
or contact provider support if identity/control remains uncertain. Do not
terminate unrelated work. No such UI/provider action occurred here.

## Coverage and residual risks

Local fake subprocess tests cover:

| Contract | Evidence type |
| --- | --- |
| Exact hardware/positive budgets/reviewed SHA/approval/risk gate | CPU validation, no transport on veto |
| Missing/wrong CLI version; argument errors and --live | Fake version; Popen/run veto tests |
| Pending journal before new; lost/hung allocation; no retries | Local fake list and exact correlation fixtures |
| Wrong/missing device before probe | Fake identity; zero probe calls |
| Worker/compile hang, failed worker, KeyboardInterrupt | Local bounded fake subprocesses |
| Parent termination and wedged living parent | Actual local parent kill/deadline; independent watchdog |
| Missing local name, orphan, corruption, wrong/cached identity | No guessed stop, manual attention |
| Misleading stop text, nonzero/slow stop, 403/5xx/rate limit/credentials/schema errors | Failure fixtures, not live HTTP |
| Watchdog death, repeated verification errors, unrelated sessions | Secondary finally only; no false primary safety claim |

HARD_STOP_GUARANTEE=NOT_PROVEN; HOST_OR_NETWORK_FAILURE_NOT_COVERED. Host sleep,
power loss, Windows/WSL reboot, credential loss, provider/network outages and
delayed billing can defeat local control. Windows-local tests do not validate
WSL process survival; Linux/WSL execution is NOT RUN this cycle. No off-host
release mechanism, scheduled service, provider budget limiter or exact billing
cap was installed. Time limits and polling do not prove hard CU enforcement.
Unknown prospective rate keeps COST_GATE_BLOCKED / CU_CAP_ENFORCEABILITY=BLOCKED.
Architect must assess an off-host/provider-managed mechanism before unattended
live use; a separate enablement change and Scientist budget decision remain
mandatory. Parent #7 stays OPEN/BLOCKED regardless of this offline PR's acceptance.

## Reproduction / validation evidence

The stdlib tool needs no install. pytest is an existing test dependency.
Use a new private OS temp directory for simulations, e.g. PowerShell:

```powershell
python tools/tpu_safe_adapter.py --dry-run
python tools/tpu_safe_adapter.py --dry-run --scenario success --output-dir "$env:TEMP/torchyand-safe-demo"
python -m pytest tests/test_tpu_safe_adapter.py tests/test_colab_tpu_guard.py -q
python -m pytest tests/ -q
```

Exit 0 means mock probe AND owned cleanup succeeded; expected blocked/failure
simulations return 1. CLI errors/live veto return 2 and spawn no subprocesses.
State/commands/results remain in local private temp; compact public evidence
must omit endpoints, actual account values, home paths and approval details.
No actual Colab/TPU operation is necessary to reproduce tests.

Preliminary focused validation: 87 passed in 44.24s, exit 0, before additional
approval/argument/publication checks. Final committed-source results will be
recorded below after verification; preliminary output is not final evidence.

Next action: Architect reviews this exact source/tested SHA and mock evidence.
Only Architect can accept OFFLINE_ADAPTER_ACCEPTED; this report does not assert
TPU_READY or self-approve/merge its own code.

## Final committed-source evidence

Tested implementation: `b7a999e06357c1a8e354a97e30fc8d2fafc8cbc2`, clean
working tree before execution. Subsequent evidence-only commit changes this
report, not code or tests. Windows 11 / Python 3.14.6 (Anaconda), pytest 9.0.3;
full suite uses existing torch 2.14.1+cpu. OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1
were set. Existing Python environment was reused; stdlib tool needs no install.
Windows-local execution PASS; Linux/WSL execution NOT RUN.

Genuine terminal excerpts on that exact implementation:

```text
python -m pytest tests/test_tpu_safe_adapter.py tests/test_colab_tpu_guard.py -q
98 passed in 45.77s
Exit code: 0

python -m pytest tests/ -q
1348 passed, 43 skipped, 1 warning in 107.34s (0:01:47)
Exit code: 0
```

The 98 focused cases comprise 78 new adapter cases and 20 retained guard cases.
The existing warning concerns TorchScript on Python 3.14; skips establish no
hardware coverage. No CPU native compilation, XLA compilation or remote runtime
was involved. `git diff --check` passed; production torchyand/, benchmarks/,
pyproject.toml and .github/ are unchanged against dispatch baseline. Legacy
guard's historical 20-minute full-smoke proposal remains dry-run-only; this
adapter's first-probe gate is at most 5 minutes, also not authorized.

Eight actual **local-only** fault demonstrations ran on the tested commit:

| Synthetic case | Cleanup status / outcome | Allocate / stop / probe calls (all FAKE) |
| --- | --- | --- |
| success | RELEASED / MOCK_PROBE_PASS | 1 / 1 / 1 |
| allocation response lost, no provable identity | ALLOCATION_UNCERTAIN / ALLOCATION_UNCERTAIN | 1 / 0 / 0 |
| allocation response lost, provable mock correlation | RELEASED / ALLOCATION_UNCERTAIN | 1 / 1 / 0 |
| wrong device | RELEASED / DEVICE_VERIFY_FAILED | 1 / 1 / 0 |
| misleading stop success text | CLEANUP_FAILED / MOCK_PROBE_PASS | 1 / 1 / 1 |
| local name lost | CLEANUP_FAILED / MOCK_PROBE_PASS | 1 / 0 / 1 |
| watchdog killed | RELEASED / WATCHDOG_FAILED; manual attention | 1 / 1 / 0 |
| rate evidence missing | BLOCKED / BLOCKED | 0 / 0 / 0 |

These logs contain no real endpoint/account/approval data. Raw test captures
and sanitized synthetic fault summary remain ignored under
`benchmarks/results/issue10-b7a999e-evidence/`. Summary faults.json SHA-256:
`4e8cfc8e9c8970523417ef89eacb64b468cc9bb9eb864dfc989ee03cde631497`.
Private fake journals/command logs are local temporary data, not public artifacts.
No real Colab CLI/account/credential operation occurred this cycle.

| Required status | Actual evidence |
| --- | --- |
| ADAPTER_STATIC_REVIEW | PASS (Engineer source/contract inspection; Architect review pending) |
| DEVICE_ALLOWLIST_MOCK | PASS |
| ALLOCATION_UNCERTAINTY_MOCK | PASS; real CLI correlation proof unavailable |
| OWNED_CLEANUP_MOCK | PASS (including correct fail-closed outcomes) |
| WATCHDOG_MOCK | PASS (including independent parent death/deadline and watchdog-failure handling) |
| CPU_TESTS | PASS, 98 focused cases |
| CPU_REGRESSION | PASS, 1348 passed / 43 skipped / 1 warning |
| LIVE_ADAPTER_ENABLED | NO |
| REAL_SESSION_STOP | NOT RUN |
| TPU_ALLOCATED | NO |
| XLA_COMPILE | NOT RUN |
| PROSPECTIVE_CU_RATE | UNKNOWN |
| CU_CAP_ENFORCEABILITY | BLOCKED |
| HARD_STOP_GUARANTEE | NOT_PROVEN |

No new human approval, off-host guarantee, real reconciliation capability or
resource readiness is established by mock PASS. Parent #7 remains OPEN/BLOCKED.
Next command: Architect reviews this child PR's exact head and CPU evidence,
without enabling live execution or allocating resources.
