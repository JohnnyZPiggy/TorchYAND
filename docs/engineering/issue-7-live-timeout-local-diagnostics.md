# Issue #14 — probe timeout accounting and durable local diagnostics

SAE V3.3 / Codex Engineer. Refs #14 and #7. Baseline:
`0e6ce4a6f66a8790929a715ca4845e79d1288f98` (merged PR #13).
This cycle is CPU/Mock only: no Colab command/account request, allocation,
exec/stop, login/token refresh, package installation or TPU restart.

## Facts and what remains unknown

Reviewed the retained ignored local `benchmarks/results/issue-7-live/report.md`
and orchestration scripts. The original real attempt began 2026-10-20
08:24:27.358623 UTC, ran 43.120s, exit 1 / TimeoutExpired, after one advertised
V5E1/TPU assignment. It reported RELEASED; an independent backend-backed query
at 08:26:13.163396 UTC reported zero sessions. These remain **Codex-reported**
real evidence, not an Architect rerun or a PJRT/numerical PASS.

Original host: WSL2 Ubuntu 24.04, Linux 6.18.54.1-microsoft-standard-WSL2;
existing Colab CLI 0.7.4 interpreter Python 3.15.0. The outer probe subprocess
had only 30s for interpreter startup, three GitHub reads (10s each), CLI imports,
session lookup, runtime setup, remote execution and output return. Its inner
exec timeout was also 30s. This accounting defect is source-evidenced; the first
run's exact slow substage remains **INDETERMINATE**. Original remote output,
substage times and private /tmp directory are unavailable; its disappearance
cause is UNKNOWN. No attempt was made to recreate that evidence or rerun TPU.

## Build vs Reuse / focused fix

Reuse the existing WSL2 Ubuntu and installed CLI, stdlib subprocess/JSON/hashlib,
existing independent watchdog and `guard.write_json` atomic fsynced writer.
No new package, daemon, approval/budget gate, orchestration framework, solver,
CI or pricing work. Existing identity/SHA/device/ownership gates remain intact.

`CLI.call('probe')` now has a finite **up to 180s host envelope**, shortened to
remaining total time minus the existing cleanup reserve (up to 60s). Other
metadata/stop queries remain 30s; allocation remains up to 180s. Total deadline
is still at most 300s and starts before the run's authorization recheck.
No retries or hardware fallback are added.

The child receives the actual absolute host deadline. After authorization and
imports, each official CLI `execute_code` (including initial /content setup)
receives **up to 120s**, further shortened to the lesser of host deadline and
cleanup boundary, minus **15s return margin**. The budget is recomputed before
each call. Missing/nonfinite host limits are rejected. If under 1s remains,
execution is BLOCKED instead of extending the session. A slow startup can still
consume the finite envelope; diagnostics identify the reached stage. This is
not a guarantee of successful XLA startup within the approved session.

The existing CLI 0.7.4 execution.py was read locally, not invoked: session lookup
precedes runtime setup and the actual code block. Two small method wrappers
record session lookup and execute_code stages without recording arguments or
results. Public CLI main/commands remain in use; no custom Colab HTTP path.

## Durable evidence and privacy

Default stable destination: `~/.local/state/torchyand/probes/<random-run-id>/`
on the WSL home filesystem, outside source control, Dropbox and /tmp. The root
is fixed, resolved, checked against checkout/Dropbox/OneDrive; there is no new
configuration layer. Directories use 0700, files 0600. Runtime/session mapping,
approval copies and CLI config remain private disposable OS-temp state.

A separate random diagnostics ID is returned in the summary; it is not the
provider session name/endpoint. Incremental append-only `timeline.jsonl` uses
one O_APPEND write per JSON record, flush via fsync before blocking actions.
Rows contain event, UTC timestamp, monotonic elapsed and remaining milliseconds,
stage, outcome, error class, source SHA and diagnostics run ID. Stages cover
parent/child handoff, authorization and each GitHub read, CLI import/dispatch,
watchdog readiness, session lookup/discovery, remote setup/exec, result parsing,
cleanup and fresh backend release verification. Startup latency is the interval
from handoff to child-enter. A final parse verdict alone is not a timing claim.

Success, error, timeout and interrupt persist output **byte counts only**.
Exception messages, raw stdout/stderr (including TimeoutExpired partial bytes),
code, headers, URLs, OAuth/proxy fields, exact session names/endpoints, approval
references and interpreter/user paths are not copied to durable diagnostics.
A missing probe is NOT VERIFIED; injected lifecycle results are labeled CPU_MOCK
and never labeled real TPU verified. Worker death can leave a final WORKER_LOST
record through the independent watchdog.

Terminal `summary.json` is atomic/fsynced. `manifest.json` records SHA-256 and
byte length of each sealed artifact prefix. A still-exiting child may append
later timeline rows after parent death; the manifest authenticates the observed
prefix, not an assertion that no later tail exists. Normal completion seals the
whole timeline. Existing human-readable console summary remains sanitized.

Diagnostics failure blocks new allocate/probe handoff but does not prevent
cleanup of a known owned session. If the runtime journal vanishes, cleanup is
honestly CLEANUP_FAILED / MANUAL_ATTENTION_REQUIRED; no RELEASED inference is
made from missing files. Stable-disk write failure is reported WRITE_FAILED.

No automatic deletion/rotation is added. Keep private evidence for review, then
remove only the selected completed run directory manually. WSL shutdown or /tmp
cleaners may remove disposable state depending on host configuration; user-home
storage generally persists, but disk loss/manual deletion can still destroy it.
Neither cause is attributed to the first run. Missing original diagnostics
cannot be reconstructed from the new logger.

## Local reproductions and verification

Injected monotonic clocks and fake GitHub/CLI isolate: three 10s GitHub reads,
40s interpreter startup, 40s CLI import, 40s session lookup, genuine simulated
remote timeout above 120s, tiny remaining budget, 17s result parsing, partial
stdout/stderr on timeout and KeyboardInterrupt. Delayed auth/startup/import/
lookup exceed the old outer 30s while succeeding inside the new bounded budget;
a genuine simulated remote timeout remains a failure. No real sleep or remote
kernel is used for these delay cases.

Tests remove disposable state after a fake lifecycle and verify readable,
hash-valid durable evidence with no canary token/endpoint/name; remove runtime
state during a fake run and require manual attention; remove diagnostics and
prove cleanup still happens; exercise parent death and existing admission,
auth/source, corrupted journal and cleanup failure regressions. All GitHub and
Colab operations are dependency-injected fakes.

The existing WSL CLI interpreter has no pytest. Without installing anything,
a stdlib-only self-check vetoed subprocess/network, wrote private CPU_MOCK
metadata, removed a disposable directory, and verified 0700/0600 permissions,
hashes and absence of canary secrets. A **separate WSL invocation** read the
same user-home artifacts and verified the hashes again. This proves the tested
local persistence path, not survival of a host power failure. Windows runs the
pytest CPU suite. Exact SHA/results are recorded below after completion.

| Field | State |
|---|---|
| FIRST_REAL_PROBE | BLOCKED_TIMEOUT (historical) |
| REAL_RELEASE | CODEX_REPORTED_PASS (historical) |
| TPU_PJRT | NOT VERIFIED |
| LOCAL_TIMEOUT_ROOT_CAUSE | INDETERMINATE for real run; accounting defect source-evidenced |
| LOCAL_FAKE_DIAGNOSTICS | See completed test results below |
| DURABLE_LOG_RECOVERY | WSL stdlib self-check PASS; pytest results below |
| RAW_REMOTE_OUTPUT | NOT AVAILABLE |
| SECOND_TPU_ALLOCATION | NOT RUN |

## Completed implementation evidence

Tested implementation SHA: `e3aa00a28c1dfc8b6e121539d2534043d5f9cb13`.
Subsequent evidence-only commits do not change runtime/tests. Windows 11 build
26200, Anaconda Python 3.14.6, torch 2.14.1+cpu, pytest 9.0.3;
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1.

- `python -m pytest tests/test_tpu_live_adapter.py tests/test_tpu_safe_adapter.py tests/test_colab_tpu_guard.py -q`:
  **217 passed in 68.53s**, exit 0 (119 adapter checks +98 reused guard/mock checks).
- Default `python tools/tpu_live_adapter.py`: DRY_RUN, exit 0; no diagnostic
  directory creation, network or Colab subprocess. `git diff --check`: exit 0.
- Existing WSL Python 3.15.0 stdlib-only self-check and separate-invocation
  hash/persistence verification: PASS, exit 0; no cloud/process launch inside
  self-check, no installed packages changed. WSL pytest suite is NOT RUN.
- `python -m pytest tests/ -q`: **1467 passed, 43 skipped, 1 warning in
  167.72s**, exit 0, at the exact implementation SHA above. Warning remains
  the existing torch.jit.script / Python 3.14 FutureWarning in the MVSK oracle.
- Logs (ignored): benchmarks/results/issue-14/scoped.txt and regression.txt;
  WSL self-check script/pointer are also ignored, with no raw remote material.

LOCAL_FAKE_DIAGNOSTICS=PASS; DURABLE_LOG_RECOVERY=PASS for the local synthetic
lifecycle and separate WSL reader. Diff is three files (adapter, its tests,
this report). Next: independent Architect T2 review of the PR. No automatic
second probe, resource allocation, approval/budget expansion or self-merge.
