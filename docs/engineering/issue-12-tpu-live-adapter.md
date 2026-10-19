# Issue #12 — minimal live TPU smoke adapter

SAE V3.3 / Codex Engineer. Refs #12 and parent #7. Baseline:
`fcfbf926116dc64836697e9dcfed592cdadba59b` (merged PR #11).
This is implementation evidence and a future runbook, **not execution consent**.
Current Scientist policy removes CU pricing investigation as an engineering gate.
No Colab command, account query, credential refresh or real TPU operation was run.

## Build vs Reuse

Reuse PR #9 atomic journals, parent liveness and watchdog trigger loop; extend its
horizon parameter with the old five-second mock default unchanged. Reuse PR #11's
ownership/cleanup design, but not its fictitious owner/correlation/rate fields.
Build one stdlib-only live transport/entrypoint in `tools/tpu_live_adapter.py`.
The historical mock entrypoint remains offline. Solver, benchmark mathematics,
dependencies, Actions and SAE adoption files are unchanged.

## Static CLI mapping (official v0.7.4)

- [session.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/commands/session.py):
  `new -s NAME --tpu v5e1`, `sessions`, `stop -s NAME`. Only exact lowercase
  v5e1 is accepted locally: upstream other truthy selectors can select v6e1.
  Assignment UUID is generated inside the CLI; no caller correlation field.
  Named session state maps to backend endpoint. Display rows are parsed strictly;
  unknown output, duplicate endpoints and orphan `?` cannot prove ownership.
- [cli.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/cli.py)
  and [state.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/state.py):
  global `--config PRIVATE_FILE --auth oauth2` isolates session mappings from
  other CLI sessions. Uses an existing CLI interpreter, not a new installation.
- [execution.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/commands/execution.py):
  `exec -s NAME --file tiny_probe.py --timeout 30`. CLI exit code alone is not
  scientific evidence: require one nonce-bound structured probe marker.
- [common.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/common.py):
  fresh sessions uses backend assignments. Its auth SystemExit can otherwise
  become empty assignments when local state is empty. Bridge converts that exit
  to failure, vetoes interactive authentication and disables debug logging that
  could include HTTP headers. Background update checks are disabled for the
  exact pinned version. No private HTTP client or credential bypass is added.

These are **STATIC_VERIFIED** mappings, not installed-CLI/TPU runtime proof.

## Safety and future activation

Default `python tools/tpu_live_adapter.py` prints a dry plan without subprocesses.
After independent review/merge AND a separate explicit experiment instruction,
a supervised Engineer may prepare a private approval JSON with:

```json
{
  "source_sha": "<exact merged clean 40-hex source SHA>",
  "approval_ref": "<Scientist resource-decision comment URL on Issue #7>",
  "device": "v5e1", "max_sessions": 1, "max_minutes": 5,
  "zero_retries": true, "decision": "APPROVED", "supervised": true,
  "residual_outage_risk_ack": true,
  "quota_policy": "price_not_a_gate_no_purchase"
}
```

A distinct Issue #7 comment URL records the future explicit run instruction.
References are validated for shape/distinctness, not authenticated over GitHub:
the supervising Engineer must verify actual authors/content, merged SHA and
fresh origin/main locally. These records are deliberate operator inputs, not
cryptographic authority; fixtures are not approvals. No rate/balance field is
required. No purchase, upgrade or unlimited-runtime consent is inferred.

Future invocation only, from the reviewed clean checkout on Linux/WSL/macOS:

```text
python tools/tpu_live_adapter.py --live --approval-file PRIVATE_APPROVAL.json --instruction-ref SEPARATE_ISSUE_7_COMMENT_URL --cli-python ABSOLUTE_EXISTING_CLI_PYTHON --output-dir NEW_PRIVATE_OS_TEMP_DIRECTORY
```

Native Windows live mode is rejected; local Windows CPU/Mock is tested.
WSL/macOS execution and installed v0.7.4 compatibility remain NOT RUN. Use an
existing authenticated supported CLI environment; no interactive OAuth, package
installation or upgrades during the probe. Keep state outside synced Dropbox
and outside Git. Private directory/file permissions use POSIX 0700/0600 and
umask 077; OS temporary storage/ACL confidentiality remains the operator's duty.

The monotonic <=300-second local deadline includes preflight and cleanup.
Subprocess timeouts are bounded by remaining budget; allocation reserves up to
60 seconds for cleanup. The independent watchdog is ready before allocation;
a durable one-shot allocation claim prevents retries even after lost response.
A known unique name + endpoint absent from the preflight snapshot establishes
ownership. If allocation failed before local mapping was saved and only an
orphan remains, the adapter does not guess or allocate again.

The first future probe records Python/torch/XLA, checks compatible versions,
actual TPU PJRT/runtime device identities, runs one tiny float32 matrix product,
synchronizes and checks [[3],[7]], then signals immediate cleanup. Missing XLA
or incompatible runtime is BLOCKED; no optimizer/PCG/performance benchmark.

Cleanup reads fresh backend sessions, stops only the exact owned name once,
and verifies the same endpoint absent in another fresh query. Success-looking
stop text is ignored. Lost local mapping, ambiguous ownership, schema drift,
backend errors, persistent endpoints or journal corruption yield manual attention.
Watchdog loss cannot report a successful run even after secondary cleanup.

**Manual recovery:** remain present for the full future run. Open Colab's
Manage sessions UI; compare the private journal's unique name/known endpoint
with session details, refresh and verify disappearance. If ownership cannot be
positively established, request human reconciliation; never terminate another
session or infer an orphan belongs to this run. Record unresolved allocation as
MANUAL_ATTENTION_REQUIRED. Host power loss, network/provider outage and blocked
OS process termination cannot be made provider-side hard-stop guarantees by a
local watchdog. Local timeout is neither proof of release nor a CU charge cap.

## Local evidence

Implementation SHA and completed test results are appended after execution.
Only local Windows CPU/Mock tests were used; injected transports never invoke
installed Colab. Existing guard tests terminate a real local parent process to
check independent watchdog survival; new tests bind live cleanup to that trigger.
No real sessions are created by imports, pytest or default invocation.

| Field | State |
|---|---|
| LIVE_ADAPTER_CODE | IMPLEMENTED; pending Architect review |
| CPU_TESTS | See verified results below |
| MOCK_CLEANUP | See verified results below |
| REAL_CLI_MAPPING | STATIC_VERIFIED; runtime integration unresolved |
| REAL_SESSION_STOP | NOT RUN |
| TPU_DEVICE | NOT RUN |
| XLA_COMPILE | NOT RUN |
| ACTUAL_CU | NOT MEASURED |

### Verified implementation and commands

Tested implementation SHA: `ed70a40dd36a4b3d37ff264ac8702da474bed3e8`.
Subsequent evidence-only commits do not change runtime/tests.
Environment: Windows 11 build 26200, Anaconda Python 3.14.6,
torch 2.14.1+cpu, pytest 9.0.3. OMP_NUM_THREADS=1, MKL_NUM_THREADS=1.

- `python -m pytest tests/test_tpu_live_adapter.py tests/test_tpu_safe_adapter.py tests/test_colab_tpu_guard.py -q`:
  **144 passed in 51.73s**, exit 0. Includes 46 new adapter checks and 98 reused
  safe-adapter/independent-watchdog checks, all CPU/Mock.
- Default `python tools/tpu_live_adapter.py`: DRY_RUN, exit 0; no Colab process.
- `git diff --check origin/main`: exit 0.

Coverage includes exact hardware/authorization, zero subprocess by default,
no double allocation, realistic named/orphan listings, lost response, wrong
hardware, version mismatch, probe timeout/interruption, corrupted mapping,
misleading stop, persistent assignment, outage, dead watchdog, expired deadline,
exact CLI bridge argv, interactive auth veto and auth-exit false-empty protection.
No inferred real TPU performance, real release or CU measurements.
- `python -m pytest tests/ -q`: **1394 passed, 43 skipped, 1 warning in
  108.62s**, exit 0, at the implementation SHA above. Warning is the existing
  torch.jit.script / Python 3.14 compatibility FutureWarning in the MVSK oracle.
- Final diff: four files, isolated to tools, tests and this note. Local ignored
  command logs: benchmarks/results/issue-12/scoped-final.txt and regression-final.txt.
  The counts and commands above are the sanitized public evidence; private
  session files and credential material are not included.

CPU_TESTS=PASS; MOCK_CLEANUP=PASS. Next: Architect independent T2 review of this
PR. No automatic merge/experiment follows. After acceptance, a separate explicit
bounded experiment command may start the first real probe; price research is
not a prerequisite.
