"""Default dry-run; opt-in single supervised CLI v0.7.4 TPU probe after review.

Never run live from tests. Prices are not a gate; separate resource/run records are.
"""
from dataclasses import dataclass, asdict
import argparse
import builtins
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
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
EMPTY = '[colab] No active sessions found on server.'
PROBE = '''import json, os, sys
os.environ['PJRT_DEVICE'] = 'TPU'
report = dict(nonce=NONCE, status='BLOCKED', python=sys.version.split()[0])
try:
    import torch
    import torch_xla as xla
    import torch_xla.runtime as xr
    report.update(torch=str(torch.__version__), torch_xla=str(xla.__version__))
    if torch.__version__.split('+')[0] != xla.__version__.split('+')[0]:
        raise RuntimeError('VERSION_PAIR_MISMATCH')
    if xr.device_type() != 'TPU':
        raise RuntimeError('NOT_TPU')
    device = xla.device()
    real = xla.real_devices()
    kinds = [str(a.get('device_kind', '')) for a in xr.global_runtime_device_attributes()]
    if device.type != 'xla' or not real or any(not d.startswith('TPU:') for d in real) or not kinds or any('TPU' not in k.upper() for k in kinds):
        raise RuntimeError('TPU_IDENTITY_MISSING')
    value = torch.tensor([[1., 2.], [3., 4.]], device=device) @ torch.ones(2, 1, device=device)
    xla.sync(wait=True)
    actual = value.cpu().tolist()
    if actual != [[3.], [7.]]:
        raise RuntimeError('NUMERICAL_MISMATCH')
    report.update(status='PASS', PJRT='TPU', real_devices=real, device_kinds=kinds, result=actual)
except Exception as error:
    report['error'] = type(error).__name__
print('TORCHYAND_TPU_PROBE:' + json.dumps(report))
'''


@dataclass(frozen=True)
class Permit:
    source_sha: str
    approval_ref: str
    instruction_ref: str
    cli_python: str
    device: str = 'v5e1'
    seconds: int = 300


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


def authorize(record, instruction, sha, merged_sha, dirty, cli_python, platform=sys.platform):
    permit = validate_record(record, instruction, sha, merged_sha, dirty, cli_python, platform)
    remote = github_json(API + '/git/ref/heads/main')
    if (remote.get('ref') != 'refs/heads/main' or remote.get('object', {}).get('type') != 'commit'
            or remote['object'].get('sha') != sha):
        raise ValueError('LIVE_REMOTE_MAIN_MISMATCH')
    now = datetime.now(timezone.utc)
    expected = {key: value for key, value in record.items() if key != 'approval_ref'}
    previous_update = None
    for reference, kind in ((permit.approval_ref, 'TPU_RESOURCE_APPROVAL'),
                            (instruction, 'TPU_SMOKE_EXECUTION')):
        comment_id = int(reference.rsplit('-', 1)[1])
        comment = github_json(API + '/issues/comments/' + str(comment_id))
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
        reply = self.runner([data['permit']['cli_python'], '-c', CHILD, str(ROOT),
                             'bridge', str(self.directory), operation], shell=False,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=min(180 if operation == 'allocate' else 30, remaining))
        if reply.returncode or reply.stderr.strip():
            raise RuntimeError('CLI_FAILED_PRIVATE_DIAGNOSTICS_REQUIRED')
        return reply.stdout


def bridge(directory, operation):
    """No direct HTTP: guarded official v0.7.4 CLI main, per-run private config."""
    data = state(directory)
    permit = Permit(**data['permit'])
    # Child validates the recorded two-step decision; never accepts mock:// records.
    record = json.loads((directory/'resource-approval.json').read_text())
    if validate_record(record, permit.instruction_ref, permit.source_sha, permit.source_sha, False,
                       permit.cli_python) != permit or time.monotonic() >= data['deadline']:
        raise ValueError('CHILD_PERMIT_INVALID')
    name = data['name']
    arguments = {'version': ['version'], 'sessions': ['sessions'],
                 'allocate': ['new', '-s', name, '--tpu', 'v5e1'],
                 'probe': ['exec', '-s', name, '--file', str(directory/'tiny_probe.py'), '--timeout', '30'] ,
                 'stop': ['stop', '-s', name]}
    if operation not in arguments:
        raise ValueError('COMMAND_NOT_ALLOWLISTED')
    if operation in ('allocate', 'probe'):
        authorize(record, permit.instruction_ref, permit.source_sha, permit.source_sha, False, permit.cli_python)
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
    import colab_cli.auth as auth
    import colab_cli.common as common
    from colab_cli import auto_update
    from colab_cli.cli import main as cli_main
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
    sys.argv = ['colab', '--config', str(directory/'cli-sessions.json'), '--auth', 'oauth2', *arguments[operation]]
    cli_main()
    return 0


def discover(data, rows):
    matches = [r for r in rows if r['name'] == data['name']]
    if len(matches) != 1 or matches[0]['endpoint'] in data['before']:
        raise ValueError('ALLOCATION_UNCERTAIN_MANUAL_ATTENTION_REQUIRED')
    return matches[0]


def release(directory, cli, trigger):
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
        result.update(status='RELEASED', attention='NONE')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result['error'] = type(error).__name__
    guard.write_json(directory/'release.json', result)
    return result


def watch(directory, cli=None):
    data = state(directory)
    (directory/'watch-ready').touch()
    reason = guard.wait_for_trigger(directory, data['deadline'], data['cleanup_reserve'], data['parent'], 300)
    result = release(directory, CLI(directory) if cli is None else cli, reason)
    return 0 if result['status'] == 'RELEASED' else 1


def start_watch(directory):
    return subprocess.Popen([sys.executable, '-c', CHILD, str(ROOT), 'watch', str(directory)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(permit, record, directory, cli_factory=CLI, watch_factory=start_watch):
    if authorize(record, permit.instruction_ref, permit.source_sha, permit.source_sha, False,
                 permit.cli_python, platform='linux') != permit:
        raise ValueError('PERMIT_RECORD_MISMATCH')
    if not directory.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError('PRIVATE_OS_TEMP_DIRECTORY_REQUIRED')
    directory.mkdir(parents=True, exist_ok=False); directory.chmod(0o700)
    os.umask(0o077) if os.name == 'posix' else None
    started = time.monotonic()
    data = dict(schema=1, permit=asdict(permit), name='torchyand-live-'+uuid.uuid4().hex,
                started_utc=datetime.now(timezone.utc).isoformat(), deadline=started+permit.seconds,
                cleanup_reserve=min(60, permit.seconds*.25), parent=os.getpid(),
                before=[], endpoint=None, phase='PRECHECK')
    save(directory, data); guard.write_json(directory/'resource-approval.json', record)
    (directory/'tiny_probe.py').write_text(PROBE.replace('NONCE', repr(data['name'])), encoding='utf-8')
    cli = cli_factory(directory)
    guardian = None
    outcome, error_kind = 'BLOCKED', None
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
        row = discover(data, sessions(cli.call('sessions')))
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
        probe = json.loads(markers[0])
        if (probe.get('nonce') != data['name'] or probe.get('status') != 'PASS'
                or probe.get('PJRT') != 'TPU' or probe.get('result') != [[3.], [7.]]
                or not probe.get('real_devices') or any(not x.startswith('TPU:') for x in probe['real_devices'])):
            raise ValueError('PROBE_FAILED')
        guard.write_json(directory/'probe.json', probe)
        outcome = 'TPU_PROBE_PASS'
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        error_kind = type(error).__name__
    finally:
        cleanup = dict(status='NOT_NEEDED', attention='NONE')
        if guardian is not None:
            (directory/'done').touch()
            try:
                guardian.wait(timeout=max(.1, data['deadline']-time.monotonic()))
            except subprocess.TimeoutExpired:
                guardian.kill(); guardian.wait(timeout=2)
            if (directory/'release.json').exists():
                cleanup = json.loads((directory/'release.json').read_text())
            else:
                cleanup = release(directory, cli, 'WATCHDOG_UNAVAILABLE')
                outcome, error_kind = 'WATCHDOG_FAILED', 'IndependentCleanupUnavailable'
        result = dict(outcome=outcome, cleanup=cleanup, error=error_kind, source_sha=permit.source_sha,
                      XLA_COMPILE='NOT BENCHMARKED', ACTUAL_CU='NOT MEASURED', HARD_STOP_GUARANTEE='NOT_PROVEN')
        guard.write_json(directory/'summary.json', result)
    return result


def child(argv):
    if len(argv) == 3 and argv[0] == 'bridge':
        return bridge(Path(argv[1]), argv[2])
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
