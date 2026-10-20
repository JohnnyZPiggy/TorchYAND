# Issue #16 — import diagnostics and probe stdout/stderr

SAE V3.3 / Codex Engineer. Refs #16 and #7. Baseline main:
`d26b9b6ac223336897ee5ce907a001a48964bd67` (merged PR #15).
This cycle is local CPU/Mock and read-only existing local artifacts only.
No Colab/account operation, session allocation/exec/stop, token refresh/login,
installation, package upgrade, billing, or third TPU run occurred.

## Observation vs explanation

The second actual run's retained, endpoint/nonce-correlated official CLI history
was read locally again in the existing WSL environment. It still reports remote
Python **3.13.16**, status BLOCKED and ModuleNotFoundError, without error.name or
stage. **MISSING_MODULE=UNKNOWN**: neither the old report nor its durable stage
trace proves whether torch, torch_xla or a transitive dependency is absent.
Existing WSL host CLI uses Python **3.15.0**, google-colab-cli **0.7.4**; host
package inventory cannot establish remote Python inventory or compatibility.

Historical total was 47.299s; remote execution returned in 15.415s. Parent
RuntimeError resulted from the blanket nonempty-stderr veto (stdout 155 bytes,
stderr 191 bytes), not a new TimeoutExpired. The original raw stderr contents
remain unavailable. Prior release is Codex-reported PASS (endpoint absent and
independent server sessions=0); no provider query was repeated in this cycle.

## Small change / Build vs Reuse

Reuse the same adapter, PR #15 staged budgets, private fsynced diagnostics and
owned cleanup. Only embedded PROBE, probe transport acceptance, safe diagnostic
projection, tests and this note change. No dependency metadata, solver, CI,
framework, approval mechanism or budget gate is introduced.

PROBE marks import_torch, import_torch_xla, import_torch_xla_runtime, versions,
pjrt, device, matmul, sync, result and complete. Versions are recorded after
each successful package import. On ModuleNotFoundError it reports the actual
import instruction stage and error.name independently: a transitive failure
inside import_torch_xla can correctly name numpy._core, not necessarily
 torch_xla. Mismatched names are not rewritten into a guessed top-level cause.
Only ASCII dotted identifiers up to 96 characters pass; malformed, nonstring,
path/URL/newline/oversized names become UNKNOWN. The host applies the same
validation again. Common exception classes and stages use fixed vocabularies;
unrecognized classes are generic, with no exception message or traceback.

Only probe + exit code 0 may pass stdout onward despite stderr. Nonzero exit
remains fatal. version/sessions/allocate/stop preserve their existing stderr
rejection. stderr_present and UTF-8 stderr_bytes are persisted without contents.
One JSON object with a matching nonce is required before projecting diagnostics.
Missing/multiple/malformed markers or wrong/missing nonce fail closed. A valid
BLOCKED report remains blocked. Wrong PJRT/device/result cannot become PASS.

Persistent summary now includes bounded probe_diagnostic fields: remote-reported
status, validated stage/error_class and, for ModuleNotFoundError, missing_module.
This is a diagnostic projection, not the host's final success decision; top-level
outcome, cleanup and TPU_PJRT still determine accepted evidence. Legacy BLOCKED
records with error but no stage/module retain UNKNOWN metadata rather than being
invented. Raw stderr, tokens, arbitrary record fields and nonce are not copied.

The <=300s deadline, cleanup reserve, 180s host/120s remote caps, exact v5e1,
zero retry, source/Scientist checks, independent watchdog, endpoint absence check,
0700/0600 user-home evidence and hash manifest remain unchanged. CPU_MOCK does
not claim real TPU verification. Disposable-state loss remains manual attention.

## Diagnosis and conditional future remedy

No dependency can be selected or installed from UNKNOWN. A later independently
reviewed probe must first identify the actual missing module and remote Python,
torch and torch_xla versions/provider runtime. Then check official matching
wheel/Python/torch-XLA compatibility. If the evidence identifies missing
 torch_xla, a single scoped future remote-environment bootstrap may be indicated;
installing into the local WSL host is not a remote fix. No such bootstrap, package
choice or automatic third experiment is authorized or performed by this PR.
The CPU library's torch>=2.0 dependency remains unchanged.

## Tests and evidence

Tests inject imports, tensor/device facades, GitHub and CLI transports. Cover
missing torch/torch_xla/runtime/transitive modules, mismatched import/name,
malformed names, other exceptions, success with stderr, BLOCKED with stderr,
legacy records, nonzero exits with markers, malformed/multiple/array records,
wrong nonce/PJRT/device/matrix, canary redaction, and prior timeout/auth/cleanup/
temp-loss regressions. No actual TPU/XLA module or Colab interface is used.

Existing WSL Python 3.15 ran a stdlib-only self-check: subprocess/network vetoes,
fake import failure, bounded module/stage projection, and private persistent
summary with no canary contents. Read-only historical inspection in the same
script published only the nonce-correlated safe fields above. No pytest was
installed in that environment. Windows pytest evidence is recorded below.

| Field | State |
|---|---|
| SECOND_REAL_PROBE | BLOCKED_IMPORT (historical) |
| REMOTE_PYTHON | 3.13.16 (historical retained report) |
| MISSING_MODULE | UNKNOWN for historical run |
| HOST_CLI | 0.7.4 |
| REAL_RELEASE | CODEX_REPORTED_PASS (historical) |
| TPU_PJRT | NOT VERIFIED |
| LOCAL_CPU_MOCK | See completed results below |
| THIRD_TPU_ALLOCATION | NOT RUN |
