# Issue #8: read-only CU inspection and local watchdog simulation

SAE V3.3 / Codex Engineer. Refs [#8](https://github.com/JohnnyZPiggy/TorchYAND/issues/8)
and parent [#7](https://github.com/JohnnyZPiggy/TorchYAND/issues/7).
Baseline `0c057c873e3e8ddbab5f94e4b852bd2da5fd5678` re-fetched before work.
This is an offline engineering deliverable, not a TPU execution authorization.

## Decision

READONLY_ACCOUNT_USAGE_PASS: installed CLI/account access returned balance,
current aggregate consumption rate and assignment-count fields. **Prospective
v5e1 rate is UNKNOWN; NUMERIC_CU_GUARD=BLOCKED.** No prospective quote or
defensible upper bound was found in the supported interfaces inspected.
The independent watchdog is a **LOCAL MOCK ONLY** implementation. It cannot
establish real cleanup or cost-cap enforcement. Parent #7 stays
TPU_SMOKE_NOT_READY / WAITING_FOR_SCIENTIST_RESOURCE_APPROVAL.

No TPU allocation, Colab session creation, remote execution, remote install or
real stop occurred. Real commands were limited to CLI version/help and the two
authorized read-only account queries below. No scheduled/background automation.
No assertion of current or historical real SESSION_COUNT=0 is made here; the
session response is retained privately and not published.

## CU inspection and privacy-safe evidence

Actual platform: WSL2 Ubuntu 24.04; executable
`/home/zhangru/.local/bin/colab`, resolving to the uv-managed google-colab-cli
entrypoint. `colab version` returned `Version: 0.7.4`, exit 0;
`colab usage --help` returned the installed usage command help, exit 0.
Native Windows CLI support is not assumed.

Before account access, inspected the installed Python package source, not any
credential file. `commands/usage.py` calls Client.get_consumption_user_info;
`client.py` uses its GET-default request method for the documented consumption
and assignment-list endpoints. The installed authentication code supports cached
OAuth2/normal refresh and an interactive fallback. This inspection grants no
new scope or credential access.

The actual account queries invoked the installed CLI `main` in its own Python
interpreter with explicit `--auth oauth2`, under a fail-closed wrapper:
interactive OAuth fallback and input() were replaced with exceptions; the CLI
request method accepted only GET requests to the two inspected read-only paths.
The wrapper called the official CLI; it did not invoke a hidden endpoint or
replace account responses with mocks. No login prompt, new auth flow, browser,
OAuth setup or scope change was performed. Existing CLI-managed authentication
was reused. Public evidence deliberately contains no credential paths or raw
account output.

Sanitized actual observations (UTC, 2026-10-19):

```text
20:38:21.669551Z  colab --auth oauth2 sessions
exit_code=0; session_query=PASS; raw_retained_privately=true

20:38:22.332969Z  colab --auth oauth2 usage
exit_code=0
balance_field=AVAILABLE
rate_field=AVAILABLE
assignments_field=AVAILABLE
prospective_rate=UNKNOWN
```

Representative redacted display schema (not a numerical account disclosure):

```text
Current balance: <REDACTED_NUMBER> compute units
Usage rate: <REDACTED_NUMBER>/hr
Active assignments: <REDACTED_NUMBER>
```

Output was captured in memory, stored in private WSL temporary files with mode
0600 outside the repository/Dropbox, and only field-presence classifications
were emitted. Raw balances, rates, account identifiers, tokens, credit/billing
details and full responses are not in Git. The successful queries are real;
the watchdog simulations below are entirely synthetic.

Official sources consulted:
[consumption model](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/consumption.py),
[usage/auth design](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md),
[session guide](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md),
[Colab FAQ](https://research.google.com/colaboratory/faq.html).
The account rate is aggregate current consumption, not a quote for a new TPU.
Availability/limits vary. No supported prospective v5e1 quote was established;
do not substitute Cloud TPU prices or allocate hardware to discover a rate.
If an account UI can show a non-allocating prospective rate, the Scientist must
inspect it and provide a dated rate/bound source; that UI capability is not
verified here. No GUI automation or private API scraping was attempted.

The successful read-only interface could support a future approved poll of
balance/rate deltas, but this task performs only one usage query and implements
no real polling. Aggregate account usage may include unrelated workloads;
refresh delays, balance adjustments, query failures, stop latency and billing
granularity prevent an exact hard numeric CU guarantee. Time cap is not CU cap.
Until a defensible rate bound, numeric Scientist cap and safety decision exist,
autonomous live TPU use is BLOCKED.

## Build-vs-Reuse and implementation

REUSE the official CLI's read-only interfaces and Python standard-library
subprocess/time/JSON/process APIs. BUILD only `tools/colab_tpu_guard.py` and
focused tests; no orchestrator framework or new dependency. Production kernels,
benchmark mathematics, dependencies and CI are unchanged.

The tool has no real CLI adapter, no allocation code and no approval bypass.
Default invocation prints DRY_RUN / AWAIT_SCIENTIST_APPROVAL. `--live` always
fails closed before spawning anything, even if someone supplies an approval
record elsewhere. A future live adapter would require a new reviewed change,
recorded Scientist device/session/time/numeric-CU/SHA approval AND a separate
user-invoked execution step. Current code cannot perform a real stop.

State machine: PRECHECK -> AWAIT_SCIENTIST_APPROVAL ->
ALLOCATE_NAMED_SESSION (future only; absent) -> RUN (mock worker only) ->
STOP_OWN_SESSION (local fake only) -> VERIFY_RELEASE (synthetic server only).
Tests create a synthetic local server file, never a remote session. Mock CU=1
is synthetic validation data, never a budget approval. Default plan has MAX_CU
unset. Preflight-veto/missing-budget simulations invoke no fake stop/list command.

Each simulation creates a new private directory and random owner/name marker,
pinned benchmark SHA, device v5e1 and max_sessions=1. A separate Python watchdog
starts before the fake worker and monitors a monotonic deadline, completion,
cancel signal and parent liveness. It does not rely on the benchmark process's
finally block. Windows parent liveness uses a query-only process handle;
POSIX uses signal 0. No process is killed by a raw PID from state metadata.
The controller kills only its own Popen child handle when required.

Simulations use a 2.5-second total budget, including startup and cleanup, with
1.2 seconds reserved for cleanup. Every fake CLI call has a timeout at most
0.35 seconds and no longer than the remaining budget. Each cleanup performs
at most one stop and bounded precheck/verify calls, with no retries/reallocation.
Metadata missing/stale/foreign, server owner mismatch or unknown command argv
veto cleanup. CLI argv are fixed lists, shell=False; only a source-identical
copy of the local fake executable can be injected. No untrusted command template
or PATH-resolved `colab` command is executed by tests/tool.

Result records are idempotent for the same validated ownership; another owner
cannot reuse a cached result. Unrelated synthetic sessions are preserved. Stop
failure, hung stop, invalid metadata or a session still listed produces
CLEANUP_FAILED / MANUAL_ATTENTION_REQUIRED. SESSION_COUNT stays UNKNOWN; success
means only MOCK_RELEASE_VERIFIED for the exact owned synthetic name.

## Future cleanup runbook and limits (not executed)

Candidate envelope only: v5e1, at most one session, 20 minutes TOTAL from before
allocation through cleanup, reserve a cleanup buffer, zero allocation retries,
no hardware escalation; numeric MAX_CU remains UNSET / NOT AUTHORIZED.
Before any future allocation, validate the separate approval, source SHA and
unique owned name/owner record in a private local state file. Start the reviewed
independent watchdog first. Each real subprocess would need a bounded wait.
On completion/cancel/timeout/parent failure, inspect only the owned session,
stop that exact named session once, then query the server to verify release.
If metadata is missing, identity is ambiguous, stop fails or release cannot be
verified, require manual attention and do not report zero sessions. Never stop
an unrelated session or allocate again during cleanup. This is a design runbook,
not evidence that real commands or real session mapping were validated.

HARD_STOP_GUARANTEE=NOT_PROVEN. Host power loss, WSL shutdown/reboot, network/API
outage, process scheduling or parent-PID reuse can defeat local monitoring.
Windows mode bits do not establish private ACLs; caller-controlled private
storage and future platform-specific permission review are necessary for live
metadata. WSL child behavior does not survive Windows/WSL shutdown. No Task
Scheduler job, system service, off-host monitor or provider-enforced spend cap
was installed or verified. A local process test is not an off-host guarantee;
these limitations block autonomous live use rather than being waived.

## Reproduction and actual validation

No installation required for the stdlib tool. pytest is needed for tests only:

```text
python tools/colab_tpu_guard.py --dry-run
python tools/colab_tpu_guard.py --dry-run --scenario timeout --output-dir benchmarks/results/issue8-timeout
python -m pytest tests/test_colab_tpu_guard.py -q
python -m pytest tests/ -q
```

Output directory must be new. Only local files/processes are created. Internal
_fake/_watch/_worker entries are local simulator plumbing, not live operations.
KeyboardInterrupt is injected locally in the cancellation test; parent death
test actually terminates a separate local controller and observes independent
watchdog cleanup. Timeout tests actually wedge a local worker or fake stop.
These are MOCK_ONLY, not CLEANUP_VALIDATED.

Initial scoped validation: `18 passed in 8.00s`, exit 0, local Windows Python
3.14.6. Committed-source evidence and final matrix will be appended after the
bounded final validation. No TPU/XLA device test has been run.

Next action: Architect reviews exact head, read-only evidence and mock safety
coverage. Parent #7 remains open and blocked; no TPU readiness inferred.
