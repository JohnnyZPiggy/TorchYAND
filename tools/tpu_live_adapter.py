"""Default dry-run; opt-in single supervised CLI v0.7.4 TPU probe after review.

Never run live from tests. Prices are not a gate; separate resource/run records are.
"""
from dataclasses import dataclass, asdict
import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import colab_tpu_guard as guard

CHILD = ('import sys;sys.path.insert(0,sys.argv[1]);from tools import tpu_live_adapter as a;'
         'raise SystemExit(a.child(sys.argv[2:]))')
REFERENCE = r'https://github\.com/JohnnyZPiggy/TorchYAND/issues/7#issuecomment-[0-9]+'
API = 'https://api.github.com/repos/JohnnyZPiggy/TorchYAND'
SCIENTIST_ID = 158853624
EVIDENCE_ROOT = Path.home()/'.local/state/torchyand/probes'
EMPTY = '[colab] No active sessions found on server.'
PROBE_STAGES = ('import_torch', 'import_torch_xla', 'import_torch_xla_runtime', 'versions',
                'pjrt', 'device', 'matmul', 'sync', 'result', 'complete')
PROBE_ERRORS = ('ModuleNotFoundError', 'ImportError', 'RuntimeError', 'ValueError', 'TypeError',
                'AttributeError', 'AssertionError', 'OSError', 'Exception')
MODULE_NAME = r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'
PROBE = '''import json, os, sys, re
os.environ['PJRT_DEVICE'] = 'TPU'
report = dict(nonce=NONCE, status='BLOCKED', python=sys.version.split()[0], stage='import_torch')
try:
    import torch
    report['torch'] = str(torch.__version__)
    report['stage'] = 'import_torch_xla'
    import torch_xla as xla
    report['torch_xla'] = str(xla.__version__)
    report['stage'] = 'import_torch_xla_runtime'
    import torch_xla.runtime as xr
    report['stage'] = 'versions'
    if torch.__version__.split('+')[0] != xla.__version__.split('+')[0]:
        raise RuntimeError('VERSION_PAIR_MISMATCH')
    report['stage'] = 'pjrt'
    if xr.device_type() != 'TPU':
        raise RuntimeError('NOT_TPU')
    report['stage'] = 'device'
    device = xla.device()
    real = xla.real_devices()
    kinds = [str(a.get('device_kind', '')) for a in xr.global_runtime_device_attributes()]
    if device.type != 'xla' or not real or any(not d.startswith('TPU:') for d in real) or not kinds or any('TPU' not in k.upper() for k in kinds):
        raise RuntimeError('TPU_IDENTITY_MISSING')
    report['stage'] = 'matmul'
    value = torch.tensor([[1., 2.], [3., 4.]], device=device) @ torch.ones(2, 1, device=device)
    report['stage'] = 'sync'
    xla.sync(wait=True)
    report['stage'] = 'result'
    actual = value.cpu().tolist()
    if actual != [[3.], [7.]]:
        raise RuntimeError('NUMERICAL_MISMATCH')
    report.update(status='PASS', stage='complete', PJRT='TPU', real_devices=real, device_kinds=kinds, result=actual)
except Exception as error:
    kind = 'ModuleNotFoundError' if isinstance(error, ModuleNotFoundError) else type(error).__name__
    report['error_class'] = kind if kind in ('ModuleNotFoundError', 'ImportError', 'RuntimeError',
        'ValueError', 'TypeError', 'AttributeError', 'AssertionError', 'OSError') else 'Exception'
    if isinstance(error, ModuleNotFoundError):
        name = error.name
        report['missing_module'] = name if isinstance(name, str) and len(name) <= 96 and re.fullmatch(
            r'[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*)*', name) else 'UNKNOWN'
print('TORCHYAND_TPU_PROBE:' + json.dumps(report))
'''



def probe_diagnostic(probe):
    """Only nonce-validated records reach this bounded diagnostic projection."""
    stage = probe.get('stage')
    kind = probe.get('error_class', probe.get('error'))
    result = dict(status=probe.get('status') if probe.get('status') in ('PASS', 'BLOCKED') else 'UNKNOWN',
                  stage=stage if stage in PROBE_STAGES else 'UNKNOWN',
                  error_class=None if kind is None else kind if kind in PROBE_ERRORS else 'UNKNOWN')
    if kind == 'ModuleNotFoundError':
        name = probe.get('missing_module')
        result['missing_module'] = name if isinstance(name, str) and len(name) <= 96 and re.fullmatch(MODULE_NAME, name) else 'UNKNOWN'
    return result


@dataclass(frozen=True)
class Permit:
    source_sha: str
    approval_ref: str
    instruction_ref: str
    cli_python: str
    device: str = 'v5e1'
    seconds: int = 300


def new_evidence(sha, seconds):
    root = EVIDENCE_ROOT.resolve()
    if root.is_relative_to(ROOT.resolve()) or any(x.lower() in ('dropbox', 'onedrive') for x in root.parts):
        raise ValueError('UNSYNCED_EVIDENCE_REQUIRED')
    root.mkdir(parents=True, exist_ok=True); root.chmod(0o700)
    path = root/uuid.uuid4().hex
    path.mkdir(mode=0o700)
    guard.write_json(path/'context.json', dict(origin_sha=sha, run_id=path.name,
                     started=time.monotonic(), deadline=time.monotonic()+seconds))
    return path


def event(evidence, stage, name, outcome='PENDING', error=None, stdout=None, stderr=None):
    if evidence is None:
        return
    try:
        evidence = Path(evidence)
        context = json.loads((evidence/'context.json').read_text())
        now = time.monotonic()
        row = dict(event=name, utc_ts=datetime.now(timezone.utc).isoformat(),
                   elapsed_monotonic_ms=round((now-context['started'])*1000), stage=stage,
                   outcome=outcome, error_class=type(error).__name__ if error is not None else None,
                   deadline_remaining_ms=max(0, round((context['deadline']-now)*1000)),
                   origin_sha=context['origin_sha'], run_id=context['run_id'])
        # Untrusted output contents/exception messages are never persisted.
        for key, value in (('stdout_bytes', stdout), ('stderr_bytes', stderr)):
            if value is not None:
                row[key] = len(value if isinstance(value, bytes) else value.encode('utf-8'))
                if key == 'stderr_bytes':
                    row['stderr_present'] = bool(value)
        fd = os.open(evidence/'timeline.jsonl', os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(row)+'\n').encode()); os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except (OSError, ValueError, KeyError):
        return False  # Lost diagnostics must never disable owned-session cleanup.


def timed(evidence, stage, function, *args, **kwargs):
    event(evidence, stage, 'start')
    try:
        value = function(*args, **kwargs)
    except SystemExit as error:
        event(evidence, stage, 'finish' if error.code in (None, 0) else 'error',
              'PASS' if error.code in (None, 0) else 'FAILED', None if error.code in (None, 0) else error)
        raise
    except BaseException as error:
        event(evidence, stage, 'error', 'FAILED', error)
        raise
    event(evidence, stage, 'finish', 'PASS')
    return value


def finish_evidence(evidence, result):
    safe = dict(result, TPU_PJRT='VERIFIED' if result['outcome']=='TPU_PROBE_PASS'
                and result.get('execution_mode')=='LIVE' else 'NOT VERIFIED')
    guard.write_json(evidence/'summary.json', safe)
    event(evidence, 'lifecycle', 'terminal', result['outcome'])
    manifest = {}
    for name in ('context.json', 'timeline.jsonl', 'summary.json'):
        payload = (evidence/name).read_bytes()
        manifest[name] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    # A dying worker's child may append later: seal the exact observed prefix.
    guard.write_json(evidence/'manifest.json', manifest)


def validate_record(record, instruction, sha, merged_sha, dirty, cli_python, platform=sys.platform):
    if (not isinstance(record, dict) or record.get('source_sha') != sha or sha != merged_sha or dirty
            or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha)
            or record.get('device') != 'v5e1' or type(record.get('max_sessions')) is not int
            or record['max_sessions'] != 1 or type(record.get('max_minutes')) is not int
            or not 1 <= record['max_minutes'] <= 5 or record.get('zero_retries') is not True
            or record.get('decision') != 'APPROVED' or record.get('supervised') is not True
            or record.get('residual_outage_risk_ack') is not True
            or record.get('quota_policy') != 'price_not_a_gate_no_purchase'
            or not re.fullmatch(REFERENCE, str(record.get('approval_ref', '')))
            or not re.fullmatch(REFERENCE, str(instruction)) or instruction == record['approval_ref']):
        raise ValueError('SEPARATE_RESOURCE_AND_RUN_APPROVAL_REQUIRED')
    if platform not in ('linux', 'darwin') or not Path(cli_python).is_absolute():
        raise ValueError('USE_EXISTING_LINUX_WSL_OR_MACOS_CLI_PYTHON')
    return Permit(sha, record['approval_ref'], instruction, str(cli_python), seconds=record['max_minutes']*60)


def github_json(url):
    """Fixed public GitHub API only; bounded read, no credentials or redirects."""
    if not url.startswith(API + '/'):
        raise ValueError('UNTRUSTED_GITHUB_API')
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json',
                                                 'User-Agent': 'TorchYAND-approval-verifier'})
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.geturl() != url:
            raise ValueError('GITHUB_REDIRECT_REJECTED')
        raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('GITHUB_RESPONSE_TOO_LARGE')
        return json.loads(raw)


def authorize(record, instruction, sha, merged_sha, dirty, cli_python, platform=sys.platform, evidence=None):
    permit = validate_record(record, instruction, sha, merged_sha, dirty, cli_python, platform)
    remote = timed(evidence, 'github_main', github_json, API + '/git/ref/heads/main')
    if (remote.get('ref') != 'refs/heads/main' or remote.get('object', {}).get('type') != 'commit'
            or remote['object'].get('sha') != sha):
        raise ValueError('LIVE_REMOTE_MAIN_MISMATCH')
    now = datetime.now(timezone.utc)
    expected = {key: value for key, value in record.items() if key != 'approval_ref'}
    previous_update = None
    expiries = []
    for reference, kind in ((permit.approval_ref, 'TPU_RESOURCE_APPROVAL'),
                            (instruction, 'TPU_SMOKE_EXECUTION')):
        comment_id = int(reference.rsplit('-', 1)[1])
        comment = timed(evidence, 'github_approval' if kind=='TPU_RESOURCE_APPROVAL' else 'github_instruction',
                        github_json, API + '/issues/comments/' + str(comment_id))
        user = comment.get('user', {})
        if (type(comment.get('id')) is not int or comment['id'] != comment_id
                or comment.get('html_url') != reference or comment.get('issue_url') != API + '/issues/7'
                or user.get('login') != 'JohnnyZPiggy' or user.get('id') != SCIENTIST_ID
                or user.get('type') != 'User'):
            raise ValueError('SCIENTIST_COMMENT_IDENTITY_MISMATCH')
        body = json.loads(comment['body'])
        created = datetime.fromisoformat(comment['created_at'].replace('Z', '+00:00'))
        updated = datetime.fromisoformat(comment['updated_at'].replace('Z', '+00:00'))
        expires = datetime.fromisoformat(body['expires_at'].replace('Z', '+00:00'))
        if (created.tzinfo is None or updated.tzinfo is None or expires.tzinfo is None
                or not created <= updated <= now < expires
                or (now-created).total_seconds() > 86400
                or (expires-created).total_seconds() > 86400
                or previous_update is not None and created < previous_update):
            raise ValueError('SCIENTIST_COMMENT_EXPIRED_OR_OUT_OF_ORDER')
        scope = dict(expected, kind=kind, expires_at=body['expires_at'])
        if kind == 'TPU_SMOKE_EXECUTION':
            scope['approval_ref'] = permit.approval_ref
        # Canonical JSON forbids ambiguous prose/negated approval or extra scope.
        if (body != scope or any(type(body.get(k)) is not type(v) for k, v in scope.items())):
            raise ValueError('SCIENTIST_COMMENT_SCOPE_MISMATCH')
        previous_update = updated
        expiries.append(expires)
    if datetime.now(timezone.utc) >= min(expiries):
        raise ValueError('SCIENTIST_COMMENT_EXPIRED_DURING_VERIFICATION')
    return permit


def sessions(text):
    """Parse exact tagged CLI display, not a synthetic backend JSON schema."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    lines = [x for x in lines if not re.fullmatch(r'\[colab\] Pruned [0-9]+ stale local session\(s\)\.', x)]
    if lines == [EMPTY]:
        return []
    rows = []
    pattern = r'\[([^\]\s]+)\] ([^\s|]+) \| Hardware: ([A-Z0-9]+) \| Shape: (Standard|High-RAM) \| Variant: (TPU|GPU|DEFAULT)'
    for line in lines:
        if re.fullmatch(r'\[colab\] Pruned [0-9]+ stale local session\(s\)\.', line):
            continue
        match = re.fullmatch(pattern, line)
        if not match:
            raise ValueError('CLI_SESSION_SCHEMA_UNRESOLVED')
        name, endpoint, hardware, _, variant = match.groups()
        rows.append(dict(name=name, endpoint=endpoint, hardware=hardware, variant=variant))
    if not rows or len({r['endpoint'] for r in rows}) != len(rows):
        raise ValueError('CLI_SESSION_SCHEMA_UNRESOLVED')
    return rows


def state(directory):
    data = json.loads((directory/'live-journal.json').read_text())
    if (data.get('schema') != 1 or not re.fullmatch(r'torchyand-live-[0-9a-f]{32}', data.get('name', ''))
            or data.get('permit', {}).get('device') != 'v5e1'
            or not re.fullmatch(r'[0-9a-f]{40}', data['permit'].get('source_sha', ''))):
        raise ValueError('OWNERSHIP_JOURNAL_INVALID')
    return data


def save(directory, data):
    guard.write_json(directory/'live-journal.json', data)


class CLI:
    """Actual official CLI entrypoint via its existing interpreter; runner injectable."""
    def __init__(self, directory, runner=None):
        self.directory = directory
        self.runner = subprocess.run if runner is None else runner

    def call(self, operation):
        data = state(self.directory)
        remaining = data['deadline']-time.monotonic()
        if operation in ('allocate', 'probe'):
            remaining -= data['cleanup_reserve']
        if remaining <= 0:
            raise TimeoutError('LOCAL_DEADLINE')
        evidence = data.get('evidence')
        limit = min(180 if operation in ('allocate', 'probe') else 30, remaining)
        argv = [data['permit']['cli_python'], '-c', CHILD, str(ROOT), 'bridge', str(self.directory), operation]
        if operation == 'probe':
            argv.append(str(time.monotonic()+limit))
        recorded = event(evidence, operation, 'parent_child_handoff')
        if evidence and not recorded and operation in ('allocate', 'probe'):
            raise OSError('DURABLE_EVIDENCE_UNAVAILABLE')
        try:
            reply = self.runner(argv, shell=False, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=limit)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            event(evidence, operation, 'subprocess_interrupted', 'FAILED', error,
                  getattr(error, 'stdout', None), getattr(error, 'stderr', None))
            raise
        failed = reply.returncode or (operation != 'probe' and reply.stderr.strip())
        event(evidence, operation, 'subprocess_return', 'FAILED' if failed else 'PASS',
              stdout=reply.stdout, stderr=reply.stderr)
        if failed:
            raise RuntimeError('CLI_FAILED_PRIVATE_DIAGNOSTICS_REQUIRED')
        return reply.stdout



def bridge(directory, operation, probe_deadline=None):
    """No direct HTTP: guarded official v0.7.4 CLI main, per-run private config."""
    data = state(directory)
    permit = Permit(**data['permit'])
    # Child validates the recorded two-step decision; never accepts mock:// records.
    record = json.loads((directory/'resource-approval.json').read_text())
    if validate_record(record, permit.instruction_ref, permit.source_sha, permit.source_sha, False,
                       permit.cli_python) != permit or time.monotonic() >= data['deadline']:
        raise ValueError('CHILD_PERMIT_INVALID')
    evidence = data.get('evidence')
    recorded = event(evidence, operation, 'child_enter')
    if evidence and not recorded and operation in ('allocate', 'probe'):
        raise OSError('DURABLE_EVIDENCE_UNAVAILABLE')
    name = data['name']
    arguments = {'version': ['version'], 'sessions': ['sessions'],
                 'allocate': ['new', '-s', name, '--tpu', 'v5e1'],
                 'probe': ['exec', '-s', name, '--file', str(directory/'tiny_probe.py'), '--timeout', '120'],
                 'stop': ['stop', '-s', name]}
    if operation not in arguments:
        raise ValueError('COMMAND_NOT_ALLOWLISTED')
    if operation in ('allocate', 'probe'):
        timed(evidence, 'authorization', lambda: authorize(record, permit.instruction_ref,
              permit.source_sha, permit.source_sha, False, permit.cli_python, evidence=evidence))
    if operation == 'probe' and (probe_deadline is None or not math.isfinite(probe_deadline)):
        raise ValueError('FINITE_PROBE_HOST_DEADLINE_REQUIRED')
    if operation == 'allocate':
        if data['phase'] != 'NAMED_ALLOCATION_PENDING' or not (directory/'watch-ready').exists():
            raise ValueError('WATCHDOG_OR_PENDING_JOURNAL_MISSING')
        # Durable one-shot claim, including after timeout/lost response.
        with (directory/'allocation-attempted').open('x') as stream:
            stream.write('one allocation only\n'); stream.flush(); os.fsync(stream.fileno())
    if operation in ('probe', 'stop') and not data['endpoint']:
        raise ValueError('OWNED_ENDPOINT_MISSING')
    if operation == 'stop':
        with (directory/'stop-attempted').open('x') as stream:
            stream.write('one stop only\n'); stream.flush(); os.fsync(stream.fileno())
    if importlib.metadata.version('google-colab-cli') != '0.7.4':
        raise ValueError('CLI_VERSION_MISMATCH')
    def load_cli():
        import colab_cli.auth as auth
        import colab_cli.common as common
        from colab_cli import auto_update
        from colab_cli.cli import main as cli_main
        return auth, common, auto_update, cli_main
    auth, common, auto_update, cli_main = timed(evidence, 'cli_import', load_cli)
    def no_interaction(*args, **kwargs):
        raise RuntimeError('INTERACTIVE_AUTH_DISABLED')
    auth._run_remote_flow = no_interaction
    builtins.input = no_interaction
    credentials = common.get_credentials
    def fail_auth_exit(*args, **kwargs):
        try:
            return credentials(*args, **kwargs)
        except SystemExit:
            # common.State.sync_sessions otherwise converts this into empty assignments.
            raise RuntimeError('AUTH_FAILED_NOT_EMPTY_SERVER') from None
    common.get_credentials = fail_auth_exit
    auto_update.run_background_check = lambda: None
    logging.disable(logging.CRITICAL)  # Upstream debug logging can include HTTP headers.
    os.umask(0o077)
    if operation == 'probe':
        from colab_cli.runtime import ColabRuntime
        lookup = common.state.get_session
        common.state.get_session = lambda *args, **kwargs: timed(evidence, 'session_lookup', lookup, *args, **kwargs)
        execute = ColabRuntime.execute_code
        def bounded_execute(runtime, code, *args, **kwargs):
            remaining = min(probe_deadline, data['deadline']-data['cleanup_reserve'])-time.monotonic()-15
            if remaining < 1:
                raise TimeoutError('PROBE_BUDGET_EXHAUSTED')
            kwargs['timeout'] = min(120, remaining, kwargs.get('timeout', 120))
            stage = 'remote_exec' if 'TORCHYAND_TPU_PROBE:' in code else 'remote_setup'
            return timed(evidence, stage, execute, runtime, code, *args, **kwargs)
        ColabRuntime.execute_code = bounded_execute
        arguments['probe'][-1] = '120'
    sys.argv = ['colab', '--config', str(directory/'cli-sessions.json'), '--auth', 'oauth2', *arguments[operation]]
    timed(evidence, 'cli_dispatch', cli_main)
    return 0


def discover(data, rows):
    matches = [r for r in rows if r['name'] == data['name']]
    if len(matches) != 1 or matches[0]['endpoint'] in data['before']:
        raise ValueError('ALLOCATION_UNCERTAIN_MANUAL_ATTENTION_REQUIRED')
    return matches[0]


def release(directory, cli, trigger, evidence=None):
    event(evidence, 'cleanup', 'start')
    result = dict(status='CLEANUP_FAILED', attention='MANUAL_ATTENTION_REQUIRED', trigger=trigger)
    try:
        data = state(directory)
        rows = sessions(cli.call('sessions'))
        endpoint = data['endpoint']
        if endpoint is None:
            endpoint = discover(data, rows)['endpoint']
            data.update(endpoint=endpoint, phase='OWNERSHIP_RECONCILED'); save(directory, data)
        if (not isinstance(endpoint, str) or not re.fullmatch(r'[^\s|]+', endpoint)
                or endpoint in data['before']
                or any(r['name'] == data['name'] and r['endpoint'] != endpoint for r in rows)):
            raise ValueError('OWNERSHIP_ENDPOINT_MISMATCH')
        present = [r for r in rows if r['endpoint'] == endpoint]
        if present:
            if len(present) != 1 or present[0]['name'] != data['name'] or endpoint in data['before']:
                raise ValueError('ORPHAN_OR_FOREIGN_MAPPING')
            data['phase'] = 'STOP_OWNED'; save(directory, data)
            cli.call('stop')
            if any(r['endpoint'] == endpoint for r in sessions(cli.call('sessions'))):
                raise RuntimeError('ENDPOINT_STILL_PRESENT')
        data['phase'] = 'RELEASED'; save(directory, data)
        event(evidence, 'cleanup', 'backend_release_verified', 'PASS')
        result.update(status='RELEASED', attention='NONE')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result['error'] = type(error).__name__
    event(evidence, 'cleanup', 'finish', result['status'])
    if directory.exists():
        guard.write_json(directory/'release.json', result)
    return result


def watch(directory, cli=None):
    data = state(directory)
    evidence = data.get('evidence')
    (directory/'watch-ready').touch()
    event(evidence, 'watchdog', 'ready', 'PASS')
    reason = guard.wait_for_trigger(directory, data['deadline'], data['cleanup_reserve'], data['parent'], 300)
    result = release(directory, CLI(directory) if cli is None else cli, reason, evidence)
    if reason == 'PARENT_TERMINATED' and evidence:
        finish_evidence(Path(evidence), dict(outcome='WORKER_LOST', cleanup=result, error='ParentTerminated',
                        source_sha=data['permit']['source_sha'], execution_mode=data.get('execution_mode', 'UNKNOWN')))
    return 0 if result['status'] == 'RELEASED' else 1


def start_watch(directory):
    return subprocess.Popen([sys.executable, '-c', CHILD, str(ROOT), 'watch', str(directory)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(permit, record, directory, cli_factory=CLI, watch_factory=start_watch):
    evidence = new_evidence(permit.source_sha, permit.seconds)
    event(evidence, 'lifecycle', 'parent_start')
    mode = 'LIVE' if cli_factory is CLI and watch_factory is start_watch else 'CPU_MOCK'
    try:
        if authorize(record, permit.instruction_ref, permit.source_sha, permit.source_sha, False,
                     permit.cli_python, platform='linux', evidence=evidence) != permit:
            raise ValueError('PERMIT_RECORD_MISMATCH')
        if not directory.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError('PRIVATE_OS_TEMP_DIRECTORY_REQUIRED')
    except (Exception, KeyboardInterrupt) as error:
        event(evidence, 'preflight', 'error', 'BLOCKED', error)
        finish_evidence(evidence, dict(outcome='BLOCKED', cleanup=dict(status='NOT_NEEDED'),
                        error=type(error).__name__, source_sha=permit.source_sha, execution_mode=mode))
        raise
    directory.mkdir(parents=True, exist_ok=False); directory.chmod(0o700)
    os.umask(0o077) if os.name == 'posix' else None
    started = json.loads((evidence/'context.json').read_text())['started']
    data = dict(schema=1, execution_mode=mode, permit=asdict(permit), name='torchyand-live-'+uuid.uuid4().hex,
                started_utc=datetime.now(timezone.utc).isoformat(), deadline=started+permit.seconds, evidence=str(evidence),
                cleanup_reserve=min(60, permit.seconds*.25), parent=os.getpid(),
                before=[], endpoint=None, phase='PRECHECK')
    save(directory, data); guard.write_json(directory/'resource-approval.json', record)
    (directory/'tiny_probe.py').write_text(PROBE.replace('NONCE', repr(data['name'])), encoding='utf-8')
    cli = cli_factory(directory)
    guardian = None
    outcome, error_kind = 'BLOCKED', None
    diagnostic = None
    try:
        if cli.call('version').strip() != 'Version: 0.7.4':
            raise ValueError('CLI_VERSION_MISMATCH')
        data['before'] = [r['endpoint'] for r in sessions(cli.call('sessions'))]
        guardian = watch_factory(directory)
        while not (directory/'watch-ready').exists():
            if guardian.poll() is not None or time.monotonic() >= data['deadline']-data['cleanup_reserve']:
                raise RuntimeError('WATCHDOG_NOT_READY')
            time.sleep(.02)
        data['phase'] = 'NAMED_ALLOCATION_PENDING'; save(directory, data)
        if guardian.poll() is not None:
            raise RuntimeError('WATCHDOG_DIED_BEFORE_ALLOCATION')
        outcome = 'ALLOCATION_UNCERTAIN'
        cli.call('allocate')
        row = timed(evidence, 'session_discovery', discover, data, sessions(cli.call('sessions')))
        data.update(endpoint=row['endpoint'], phase='ALLOCATED'); save(directory, data)
        outcome = 'DEVICE_OR_PROBE_BLOCKED'
        if row['hardware'] != 'V5E1' or row['variant'] != 'TPU':
            raise ValueError('WRONG_ALLOCATED_HARDWARE')
        if guardian.poll() is not None:
            raise RuntimeError('WATCHDOG_DIED_BEFORE_PROBE')
        output = cli.call('probe')
        markers = [line.removeprefix('TORCHYAND_TPU_PROBE:') for line in output.splitlines()
                   if line.startswith('TORCHYAND_TPU_PROBE:')]
        if len(markers) != 1:
            raise ValueError('NO_UNIQUE_PROBE_EVIDENCE')
        probe = timed(evidence, 'result_parse', json.loads, markers[0])
        if not isinstance(probe, dict) or probe.get('nonce') != data['name']:
            raise ValueError('PROBE_NONCE_MISMATCH')
        diagnostic = probe_diagnostic(probe)
        if (probe.get('status') != 'PASS'
                or probe.get('PJRT') != 'TPU' or probe.get('result') != [[3.], [7.]]
                or not probe.get('real_devices') or any(not x.startswith('TPU:') for x in probe['real_devices'])):
            raise ValueError('PROBE_FAILED')
        guard.write_json(directory/'probe.json', probe)
        outcome = 'TPU_PROBE_PASS'
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        error_kind = type(error).__name__
        event(evidence, 'lifecycle', 'worker_error', 'FAILED', error)
    finally:
        cleanup = dict(status='NOT_NEEDED', attention='NONE')
        if guardian is not None:
            if directory.exists():
                (directory/'done').touch()
            try:
                guardian.wait(timeout=max(.1, data['deadline']-time.monotonic()))
            except subprocess.TimeoutExpired:
                guardian.kill(); guardian.wait(timeout=2)
            if (directory/'release.json').exists():
                cleanup = json.loads((directory/'release.json').read_text())
            else:
                cleanup = release(directory, cli, 'WATCHDOG_UNAVAILABLE', evidence)
                outcome, error_kind = 'WATCHDOG_FAILED', 'IndependentCleanupUnavailable'
        result = dict(outcome=outcome, cleanup=cleanup, error=error_kind, source_sha=permit.source_sha, run_id=evidence.name, execution_mode=mode,
                      XLA_COMPILE='NOT BENCHMARKED', ACTUAL_CU='NOT MEASURED', HARD_STOP_GUARANTEE='NOT_PROVEN')
        if diagnostic is not None:
            result['probe_diagnostic'] = diagnostic
        try:
            finish_evidence(evidence, result)
        except OSError:
            result['evidence_status'] = 'WRITE_FAILED'
        if directory.exists():
            guard.write_json(directory/'summary.json', result)
    return result


def child(argv):
    if len(argv) in (3, 4) and argv[0] == 'bridge':
        return bridge(Path(argv[1]), argv[2], float(argv[3]) if len(argv)==4 else None)
    if len(argv) == 2 and argv[0] == 'watch':
        return watch(Path(argv[1]))
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--approval-file', type=Path)
    parser.add_argument('--instruction-ref')
    parser.add_argument('--cli-python')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.live:
        if any((args.approval_file, args.instruction_ref, args.cli_python, args.output_dir)):
            parser.error('Live arguments require explicit --live; no commands executed')
        print('DRY_RUN: one v5e1, <=5 minutes, no retries; separate approval and run instruction required; CU price is not a gate')
        return 0
    if not all((args.approval_file, args.instruction_ref, args.cli_python, args.output_dir)):
        parser.error('Missing separate resource approval / execution instruction / CLI interpreter / private directory')
    record = json.loads(args.approval_file.read_text())
    def git(*parts):
        return subprocess.check_output(['git', *parts], cwd=ROOT, text=True, timeout=10).strip()
    permit = authorize(record, args.instruction_ref, git('rev-parse', 'HEAD'),
                       git('rev-parse', 'origin/main'), bool(git('status', '--porcelain')), args.cli_python)
    result = run(permit, record, args.output_dir)
    print(json.dumps(result))  # Sanitized summary; raw session mapping remains private.
    return 0 if result['outcome'] == 'TPU_PROBE_PASS' and result['cleanup']['status'] == 'RELEASED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
