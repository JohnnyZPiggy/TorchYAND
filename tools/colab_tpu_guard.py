"""Offline-only host watchdog simulator. No live Colab adapter is provided.

Default is dry-run. All subprocesses are this file's local fake CLI or watchdog.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

SOURCE = Path(__file__).resolve()
BASE_SHA = '0c057c873e3e8ddbab5f94e4b852bd2da5fd5678'
SCENARIOS = ('success', 'timeout', 'crash', 'cancel', 'interrupt', 'stop-failure',
             'verify-failure', 'hung-stop', 'missing-metadata', 'stale-metadata',
             'preflight-veto', 'missing-budget')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.chmod(0o600)
    temporary.replace(path)


def alive(pid):
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.restype = ctypes.c_void_p
        api.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        api.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        api.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(api.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def validate_state(state, expected_owner, expected_session, deadline):
    if (state.get('mock_only') is not True or state.get('owner') != expected_owner
            or state.get('session') != expected_session
            or not re.fullmatch(r'torchyand-mock-[0-9a-f]{32}', expected_session)
            or not re.fullmatch(r'[0-9a-f]{32}', expected_owner)
            or not re.fullmatch(r'[0-9a-f]{40}', state.get('benchmark_sha', ''))
            or state.get('deadline') != deadline or state.get('device') != 'v5e1'
            or state.get('max_sessions') != 1
            or not isinstance(state.get('max_cu'), (int, float))
            or isinstance(state.get('max_cu'), bool)
            or not math.isfinite(state['max_cu']) or state['max_cu'] <= 0
            or not deadline-5 <= state.get('created', float('-inf')) < deadline):
        raise ValueError('Missing/stale/foreign ownership or budget metadata')


class FakeCLI:
    """Dependency-injected local executable; only an identical simulator is allowed."""
    def __init__(self, directory, executable=SOURCE, runner=subprocess.run):
        executable = Path(executable).resolve()
        if hashlib.sha256(executable.read_bytes()).digest() != hashlib.sha256(SOURCE.read_bytes()).digest():
            raise ValueError('Only the identical local fake executable is allowed')
        self.prefix = [sys.executable, str(executable), '_fake', str(directory)]
        self.runner = runner

    def call(self, verb, session, owner, timeout):
        if verb not in ('sessions', 'stop') or not re.fullmatch(r'torchyand-mock-[0-9a-f]{32}', session):
            raise ValueError('Unexpected command argv')
        if not re.fullmatch(r'[0-9a-f]{32}', owner) or timeout <= 0:
            raise ValueError('Invalid owner or command budget')
        return self.runner([*self.prefix, verb, '--session', session, '--owner', owner],
                           shell=False, stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=timeout)


def cleanup(directory, state, owner, session, deadline, cli, trigger=None):
    """One stop + one verify, never reallocate; idempotent terminal result."""
    result_path = directory/'result.json'
    result = dict(mock_only=True, status='CLEANUP_FAILED', attention='MANUAL_ATTENTION_REQUIRED',
                  owner=owner, session=session, real_session_stop='NOT RUN',
                  session_count='UNKNOWN', owned_released=False, trigger=trigger)
    try:
        validate_state(state, owner, session, deadline)
        if result_path.exists():
            cached = json.loads(result_path.read_text())
            if cached.get('owner') != owner or cached.get('session') != session or cached.get('mock_only') is not True:
                raise ValueError('Foreign cleanup result')
            return cached
        for stage in ('precheck', 'stop', 'verify'):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Cleanup deadline exhausted')
            reply = cli.call('stop' if stage == 'stop' else 'sessions', session, owner, min(.35, remaining))
            if reply.returncode != 0:
                raise RuntimeError(f'{stage} nonzero ({reply.returncode}); MOCK stderr: {reply.stderr[-1500:]}')
            if stage != 'stop':
                rows = json.loads(reply.stdout)
                if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get('name'), str)
                                                    or not isinstance(r.get('owner'), str) for r in rows):
                    raise ValueError('Invalid server list schema')
                matches = [r for r in rows if r['name'] == session]
                if any(r['owner'] != owner for r in matches) or len(matches) > 1:
                    raise ValueError('Server ownership mismatch')
                if stage == 'verify' and matches:
                    raise RuntimeError('Owned session still listed')
                if stage == 'precheck' and not matches:
                    result.update(status='MOCK_RELEASE_VERIFIED', attention='NONE', owned_released=True)
                    break
        else:
            result.update(status='MOCK_RELEASE_VERIFIED', attention='NONE', owned_released=True)
    except (OSError, ValueError, TypeError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result['error'] = type(error).__name__
        result['detail'] = str(error)
    write_json(result_path, result)
    return result


def watchdog(directory, owner, session, deadline, buffer, parent_pid, executable=SOURCE):
    """Independent process; a wedged benchmark cannot prevent this loop running."""
    if not math.isfinite(deadline) or not math.isfinite(buffer) or not 0 < buffer < 5 or not 0 < deadline-time.monotonic() <= 5 or parent_pid <= 0:
        raise ValueError('Invalid bounded watchdog configuration')
    reason = 'DEADLINE'
    while time.monotonic() < deadline-buffer:
        if not alive(parent_pid):
            reason = 'PARENT_TERMINATED'; break
        if (directory/'cancel').exists():
            reason = 'CANCELLED'; break
        if (directory/'done').exists():
            reason = 'WORKER_FINISHED'; break
        time.sleep(.02)
    try:
        state = json.loads((directory/'ownership.json').read_text())
    except (OSError, ValueError):
        state = {}
    result = cleanup(directory, state, owner, session, deadline, FakeCLI(directory, executable), trigger=reason)
    return 0 if result['owned_released'] else 1


def prepare(directory, scenario, seconds):
    if scenario not in SCENARIOS or not math.isfinite(seconds) or not 1.5 <= seconds <= 5:
        raise ValueError('Known mock scenario and deadline 1.5..5 seconds required')
    if not directory.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError('Mock state must be inside local OS temporary storage, outside the synced checkout')
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    owner, session_id = uuid.uuid4().hex, uuid.uuid4().hex
    session = 'torchyand-mock-'+session_id
    start = time.monotonic()  # Before even simulated allocation.
    state = dict(mock_only=True, owner=owner, session=session, created=start,
                 deadline=start+seconds, device='v5e1', max_sessions=1, max_cu=1.,
                 benchmark_sha=BASE_SHA, scenario=scenario)
    write_json(directory/'ownership.json', state)
    # Synthetic server state only, no Colab allocation command exists.
    write_json(directory/'server.json', [dict(name=session, owner=owner),
                                         dict(name='unrelated-mock-session', owner='other')])
    return state


def start_watchdog(directory, state):
    return subprocess.Popen([sys.executable, str(SOURCE), '_watch', str(directory),
                             state['owner'], state['session'], str(state['deadline']),
                             '1.2', str(os.getpid())], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def simulate(directory, scenario='success', seconds=2.5):
    state = prepare(directory, scenario, seconds)
    if scenario in ('preflight-veto', 'missing-budget'):
        result = dict(mock_only=True, status='BLOCKED', reason=scenario,
                      real_session_stop='NOT RUN', session_count='UNKNOWN')
        write_json(directory/'result.json', result)
        return result
    if scenario == 'missing-metadata':
        (directory/'ownership.json').unlink()
    if scenario == 'stale-metadata':
        stale = dict(state, created=state['created']-3600)
        write_json(directory/'ownership.json', stale)
    watch = start_watchdog(directory, state)
    worker = None
    try:
        worker = subprocess.Popen([sys.executable, str(SOURCE), '_worker', str(directory), scenario],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if scenario in ('cancel', 'interrupt'):
            if scenario == 'interrupt':
                raise KeyboardInterrupt
            (directory/'cancel').touch()
        try:
            worker.wait(timeout=max(.05, state['deadline']-time.monotonic()-1.2))
            (directory/'done').touch()
        except subprocess.TimeoutExpired:
            pass  # Watchdog enforces its independent deadline.
    except KeyboardInterrupt:
        (directory/'cancel').touch()
    finally:
        try:
            watch.wait(timeout=max(.1, state['deadline']-time.monotonic()+.8))
        except subprocess.TimeoutExpired:
            watch.kill(); watch.wait(timeout=1)
            if not (directory/'result.json').exists():
                write_json(directory/'result.json', dict(mock_only=True, status='CLEANUP_FAILED',
                           attention='MANUAL_ATTENTION_REQUIRED', error='WATCHDOG_STALLED', session_count='UNKNOWN'))
        if worker is not None and worker.poll() is None:
            worker.kill(); worker.wait(timeout=1)
    result = json.loads((directory/'result.json').read_text())
    result['worker_exit_code'] = worker.returncode if worker is not None else None
    write_json(directory/'result.json', result)
    return result


def fake_cli(argv):
    directory, verb, flag1, session, flag2, owner = argv
    if flag1 != '--session' or flag2 != '--owner' or verb not in ('sessions', 'stop'):
        return 2
    state = json.loads((Path(directory)/'server.json').read_text())
    with (Path(directory)/'mock-commands.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(mock_only=True, verb=verb, session=session))+'\n')
    metadata_path = Path(directory)/'ownership.json'
    scenario = json.loads(metadata_path.read_text()).get('scenario') if metadata_path.exists() else ''
    if verb == 'stop':
        if scenario == 'hung-stop':
            time.sleep(60)
        if scenario == 'stop-failure':
            return 1
        if scenario != 'verify-failure':
            state = [r for r in state if not (r['name'] == session and r['owner'] == owner)]
            write_json(Path(directory)/'server.json', state)
    else:
        print(json.dumps(state))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_fake':
        return fake_cli(argv[1:])
    if argv and argv[0] == '_watch':
        directory, owner, session, deadline, buffer, parent_pid = argv[1:]
        return watchdog(Path(directory), owner, session, float(deadline), float(buffer), int(parent_pid))
    if argv and argv[0] == '_worker':
        _, _, scenario = argv
        if scenario in ('timeout', 'cancel', 'interrupt'):
            time.sleep(60)
        return 1 if scenario == 'crash' else 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.live:
        parser.error('LIVE_DISABLED: no approval/execution adapter; HARD_STOP_GUARANTEE=NOT_PROVEN')
    if args.scenario:
        if args.output_dir is None:
            parser.error('--output-dir required for local simulation')
        result = simulate(args.output_dir, args.scenario)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] in ('MOCK_RELEASE_VERIFIED', 'BLOCKED') else 1
    print(json.dumps(dict(mode='DRY_RUN', state='AWAIT_SCIENTIST_APPROVAL', device_proposal='v5e1',
                         max_sessions=1, max_seconds_proposal=1200, max_cu=None,
                         state_machine=['PRECHECK', 'AWAIT_SCIENTIST_APPROVAL',
                         'ALLOCATE_NAMED_SESSION (FUTURE ONLY; NOT IMPLEMENTED)', 'RUN (MOCK ONLY)',
                         'STOP_OWN_SESSION (MOCK ONLY)', 'VERIFY_RELEASE (MOCK ONLY)'],
                         HARD_STOP_GUARANTEE='NOT_PROVEN', LIVE='DISABLED'), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
