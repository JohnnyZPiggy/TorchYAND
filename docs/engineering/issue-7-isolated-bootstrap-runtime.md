# Issue #7: pip-less CPython 3.12 bootstrap and finite experiment duration

The previous CPU session found `/usr/bin/python3.12` and `venv`, but no
`ensurepip`. The new path creates `venv --without-pip` and uses the **already
installed default interpreter's pip >=22.3**, with `--python <venv>/bin/python`,
to manage that target. No pip upgrade, get-pip, apt, default torch replacement,
or host CLI change is needed or attempted. Missing support is a bounded failure.

Official pip documentation explicitly supports this mechanism:
https://pip.pypa.io/en/stable/topics/python-option/

## Scope and source binding

The adapter still defaults to dry-run. Existing five-minute records remain valid.
New source-bound resource and execution records can select `max_minutes: 60` and
`isolated_cp312: true`. These fields must agree in both authentic Scientist GitHub
JSON comments and the local approval record. The remote main SHA, comment identity,
expiry, and independent instruction checks are unchanged. An old approval for a
different SHA/duration cannot enable this path. No new cost/price approval is added.

The selected integer `max_minutes` is the runtime configuration, rather than an
unbound CLI override. It must be 1..1440; 24 hours guards erroneous configuration,
not provider availability. **60 minutes is an initial engineering choice**, not a
scientific acceptance threshold or a promise of provider session duration.
After review/merge, new records must refer to that actual merged main SHA; this PR
does not manufacture approval comments or launch experiments.

## Same-session remote path

1. Before package downloads, run the absolute `/usr/bin/python3.12` in an isolated
   subprocess. Check standard cp312 ABI, x86_64, glibc >=2.31, `venv`, and an initial
   8 GiB space screen. No inference from the earlier CPU image is used; `ensurepip`
   is intentionally not a prerequisite. Inspect installed manager pip version/help.
2. Create one fresh private directory and a venv with no system site packages.
   Verify the manager can actually dispatch `pip --python <target> --version`.
   There is no bootstrap fallback and no new package selected at runtime.
3. Download only `tools/tpu_wheels.json`: 30 fixed wheels, 452635819 bytes, trusted
   HTTPS hosts `files.pythonhosted.org` and `download-r2.pytorch.org`, including
   redirect validation. Cap lengths, use fresh exclusive files, and never retry.
   Verify all SHA256 values, read ZIP uncompressed sizes, and recheck installation
   space before running pip. The selected roots remain torch 2.9.0+cpu,
   torch_xla[tpu] 2.9.0, libtpu 0.0.21, tpu-info 0.14.2.
4. Use existing manager pip with `--isolated --python <target>` and offline
   `--no-index --find-links --require-hashes --only-binary=:all: --no-cache-dir
   --no-compile`. Resolve dependencies normally; never use `--no-deps` or ignore
   Python/ABI constraints. Then `pip --python <target> check` and verify all 30
   installed versions in a target subprocess.
5. Only a new venv Python process imports torch/XLA, verifies actual PJRT TPU/device
   identity, synchronizes and checks the existing `[[3.], [7.]]` tiny matmul. The
   outer notebook does not import or replace native torch. Bootstrap/package PASS
   and the actual device/probe PASS are separate nonce-checked outcomes.

The manifest is a projection of the reviewed Issue #7 comment 6289390843, retaining
every name/version/file/URL/hash/size unchanged. Generated requirements use LF;
their SHA256 is `fa623ac4cf1cce5d7710464beb5f68e51e18771c30cda1725794706805439078`.
The previously reported CRLF requirements hash remains
`b94adc3e6ac48f0b6091496f6bccd1bd4de1740fa19864b8468ab0b2e3969b3f`.
Line endings change the text digest, not the dependency or wheel hashes.

## Deadline and cleanup audit

| Limit | Behavior |
|---|---|
| Parent allocation/probe child (old 180s) | Remaining selected runtime minus cleanup reserve |
| CLI exec/remote execution (old 120s) | Remaining host probe deadline minus 15s return margin, recomputed for each remote execution |
| Remote bootstrap subprocesses | Remaining finite bootstrap budget; kill their process group on timeout |
| Watchdog horizon (old 300s) | Selected permit seconds; independent detached process |
| Cleanup | `min(60s, total*0.25)` reserved, same as before; 60s for 5/60-minute runs |
| Backend/version/stop calls | Still individually bounded at <=30s and the remaining absolute deadline |
| GitHub verification | Existing <=10s request/read bounds unchanged |
| Wheel HTTP reads | <=30s socket timeout and repeated checks of remaining overall budget |

There is no new fixed 30/120/180/300-second ceiling for builds. Five minutes is
still a valid chosen runtime; a longer selected runtime reaches parent, child,
remote execution and watchdog. The existing monitor still reacts to parent death,
cancel, completion and the cleanup boundary. One named allocation/stop claim,
exact endpoint ownership and fresh backend release verification are preserved.
Unknown/orphan/foreign mappings never authorize stopping unrelated sessions.

Bounded subprocess output is private (1 MiB file limit); durable records contain
only event times, error classes, byte counts and allowlisted diagnostics. Bootstrap
stage times and safe failure codes are added to the existing durable summary.
Credentials, endpoints, nonce and raw pip/provider output are not published there.
Failure to verify release remains MANUAL_ATTENTION_REQUIRED; a client watchdog
cannot guarantee provider release during host/provider/network outages.

## Local verification and next experiment

Tests use mock downloads/processes and fake Colab entrypoints, including malformed
redirects/lengths/hashes, missing manager capability, bootstrap/installation failure,
deadline expiration, five- and sixty-minute permits, and nonce rejection. A real
local empty venv test checks `--without-pip` plus the existing pip `--python check`
without installing dependencies or using network. Its host Python version is not
evidence about Colab CPython 3.12. A real independent local watchdog process checks
a dead parent with a sixty-minute horizon and a file-backed fake backend, preserving
an unrelated session.

No Colab/TPU session, wheel download or dependency installation is part of this PR
implementation. After Architect review and merge, proceed directly to **one real
v5e1 session**, using new compatible records for the merged SHA and finite chosen
duration, without another CPU rehearsal. Recheck prerequisites in that actual TPU
image; missing Python/pip/platform support is one truthful BLOCKED outcome with
cleanup, not a trigger for blind changes or repeated allocations. No new billing.
