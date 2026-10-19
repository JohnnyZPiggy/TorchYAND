"""Offline-only TPU lifecycle model. All commands go to local fake subprocesses.

No live transport exists; --live is always rejected. Mock records grant no authority.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import colab_tpu_guard as guard

BASE_SHA = 'f143e57824647ce95a0ab8db65dcc92f279e376b'
SCENARIOS = ('success', 'allocation-lost', 'allocation-provable', 'allocation-hang',
             'wrong-device', 'no-device', 'worker-failure', 'worker-hang', 'compile-hang',
             'interrupt', 'name-lost', 'orphan', 'stale-credentials', 'stop-nonzero',
             'stop-lie', 'stop-slow', 'server-403', 'server-500', 'rate-limit',
             'verify-errors', 'schema-drift', 'journal-corrupt', 'watchdog-death',
             'invalid-approval', 'missing-rate', 'missing-budget', 'missing-cli', 'wrong-version')
CHILD_CODE = ('import sys; sys.path.insert(0,sys.argv[1]); '
              'from tools.tpu_safe_adapter import child; raise SystemExit(child(sys.argv[2:]))')


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class ProbeConfig:
    device: str
    max_sessions: int
    max_minutes: float
    max_cu: float
    rate_upper_bound: float
    zero_allocation_retries: bool
    source_sha: str
    session: str
    owner: str
    correlation: str
    approval_ref: str
    rate_ref: str
    deadline: float

    def validate(self):
        if self.device != 'v5e1' or type(self.max_sessions) is not int or self.max_sessions != 1:
            raise ValueError('EXACT_V5E1_SINGLE_SESSION_REQUIRED')
        if not positive(self.max_minutes) or self.max_minutes > 5 or not positive(self.deadline):
            raise ValueError('FIRST_PROBE_DEADLINE_INVALID')
        if not positive(self.max_cu) or not positive(self.rate_upper_bound):
            raise ValueError('COST_GATE_BLOCKED')
        if self.max_minutes*self.rate_upper_bound/60 > self.max_cu:
            raise ValueError('COST_GATE_BLOCKED')
        if self.zero_allocation_retries is not True:
            raise ValueError('ZERO_RETRIES_REQUIRED')
        for value, pattern in ((self.source_sha, r'[0-9a-f]{40}'),
                               (self.session, r'torchyand-mock-[0-9a-f]{32}'),
                               (self.owner, r'[0-9a-f]{32}'), (self.correlation, r'[0-9a-f]{32}')):
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError('SOURCE_OR_OWNERSHIP_INVALID')
        if not isinstance(self.approval_ref, str) or not re.fullmatch(r'mock://scientist/[0-9a-f]{32}', self.approval_ref):
            raise ValueError('ONLY_SYNTHETIC_APPROVAL_RECORDS_ACCEPTED')
        if not isinstance(self.rate_ref, str) or not re.fullmatch(r'mock://rate-bound/[0-9a-f]{32}', self.rate_ref):
            raise ValueError('COST_GATE_BLOCKED')


def approval_check(config, record, reviewed_sha):
    config.validate()
    expected = dict(reference=config.approval_ref, decision='APPROVED', device='v5e1',
                    source_sha=config.source_sha, max_sessions=1, max_minutes=config.max_minutes,
                    max_cu=config.max_cu, rate_upper_bound=config.rate_upper_bound,
                    rate_ref=config.rate_ref,
                    residual_soft_cap_risk_ack=True, mock_only=True)
    if (not isinstance(record, dict) or record != expected or config.source_sha != reviewed_sha
            or record.get('residual_soft_cap_risk_ack') is not True or record.get('mock_only') is not True
            or type(record.get('max_sessions')) is not int
            or any(not positive(record.get(key)) for key in ('max_cu', 'rate_upper_bound', 'max_minutes'))
            or not 0 < config.deadline-time.monotonic() <= config.max_minutes*60):
        raise ValueError('APPROVAL_CHECK_BLOCKED')
    # Validation of synthetic fields is not authentication of any human approval.


def argv_for(operation, config):
    config.validate()
    plans = {
        'version': ('colab', 'version'),
        'sessions': ('colab', 'sessions'),
        'allocate': ('colab', 'new', '-s', config.session, '--tpu', 'v5e1'),
        'identity': ('colab', 'exec', '-s', config.session, '--file', 'tiny_identity_probe.py', '--timeout', '1'),
        'probe': ('colab', 'exec', '-s', config.session, '--file', 'tiny_probe.py', '--timeout', '1'),
        'stop': ('colab', 'stop', '-s', config.session),
    }
    if operation not in plans:
        raise ValueError('COMMAND_NOT_ALLOWLISTED')
    return plans[operation]


class FakeTransport:
    """Injected runner sees only a fixed local Python fake, never plan argv as a command."""
    def __init__(self, directory, config, runner=None):
        self.directory, self.config = directory, config
        self.runner = subprocess.run if runner is None else runner

    def call(self, operation, timeout):
        plan = argv_for(operation, self.config)
        remaining = self.config.deadline-time.monotonic()
        if operation in ('allocate', 'identity', 'probe'):
            remaining -= 1.5
        if not positive(timeout):
            raise TimeoutError('COMMAND_DEADLINE_EXHAUSTED')
        timeout = min(timeout, remaining)
        if timeout <= 0:
            raise TimeoutError('COMMAND_DEADLINE_EXHAUSTED')
        command = [sys.executable, '-c', CHILD_CODE, str(ROOT), 'fake', str(self.directory),
                   operation, json.dumps(plan)]
        reply = self.runner(command, shell=False, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=timeout)
        if reply.returncode:
            raise RuntimeError('MOCK_TRANSPORT_NONZERO')
        # Never propagate backend stdout/stderr (possibly secret-bearing) into public errors.
        return json.loads(reply.stdout)


def journal_read(directory, config):
    journal = json.loads((directory/'journal.json').read_text())
    if (journal.get('mock_only') is not True or journal.get('config') != asdict(config)
            or journal.get('schema') != 1):
        raise ValueError('JOURNAL_IDENTITY_INVALID')
    return journal


def phase(directory, config, name, endpoint=None):
    journal = journal_read(directory, config)
    journal.update(phase=name, endpoint=endpoint)
    journal['history'].append(name)
    guard.write_json(directory/'journal.json', journal)


def assignments(reply):
    if not isinstance(reply, dict) or set(reply) != {'assignments'} or not isinstance(reply['assignments'], list):
        raise ValueError('ASSIGNMENT_SCHEMA_DRIFT')
    for row in reply['assignments']:
        if not isinstance(row, dict) or not isinstance(row.get('endpoint'), str) or not isinstance(row.get('name'), str):
            raise ValueError('ASSIGNMENT_SCHEMA_DRIFT')
    endpoints = [r['endpoint'] for r in reply['assignments']]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('DUPLICATE_ENDPOINT')
    return reply['assignments']


def owned(row, config):
    return (row.get('owner') == config.owner and row.get('correlation') == config.correlation
            and row.get('source_sha') == config.source_sha
            and isinstance(row.get('endpoint'), str)
            and re.fullmatch(r'mock-endpoint-[0-9a-f]{32}', row['endpoint']))


def clean_owned(directory, config, transport, trigger):
    """At most one named stop; fresh backend endpoint absence is release evidence."""
    result_path = directory/'adapter-result.json'
    result = dict(mock_only=True, status='CLEANUP_FAILED', trigger=trigger,
                  attention='MANUAL_ATTENTION_REQUIRED', REAL_SESSION_STOP='NOT RUN',
                  TPU_ALLOCATED='NO', session_count='UNKNOWN', owned_released=False,
                  identity=dict(owner=config.owner, session=config.session,
                                correlation=config.correlation, source_sha=config.source_sha))
    try:
        config.validate()
        journal = journal_read(directory, config)
        if result_path.exists():
            cached = json.loads(result_path.read_text())
            if (cached.get('identity') != result['identity'] or cached.get('mock_only') is not True
                    or cached.get('status') == 'RELEASED' and cached.get('released_endpoint') != journal['endpoint']):
                raise ValueError('FOREIGN_CLEANUP_RESULT')
            return cached
        def request(operation):
            remaining = config.deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('CLEANUP_DEADLINE')
            return transport.call(operation, min(.5, remaining))
        rows = assignments(request('sessions'))
        endpoint = journal['endpoint']
        if endpoint is None:
            matches = [r for r in rows if owned(r, config)]
            if len(matches) != 1:
                result['status'] = 'ALLOCATION_UNCERTAIN'
                raise ValueError('ALLOCATION_OWNERSHIP_UNPROVABLE')
            endpoint = matches[0]['endpoint']
        else:
            if (not isinstance(endpoint, str) or not re.fullmatch(r'mock-endpoint-[0-9a-f]{32}', endpoint)
                    or any(owned(r, config) and r['endpoint'] != endpoint for r in rows)):
                raise ValueError('JOURNAL_ENDPOINT_MISMATCH')
            matches = [r for r in rows if r['endpoint'] == endpoint]
        if matches:
            if len(matches) != 1 or not owned(matches[0], config) or matches[0]['name'] != config.session:
                raise ValueError('ORPHAN_OR_LOCAL_MAPPING_UNPROVABLE')
            phase(directory, config, 'STOP_OWNED', endpoint)
            request('stop')  # Ignore success-looking text; verify backend independently.
            phase(directory, config, 'VERIFY_RELEASE', endpoint)
            if any(r['endpoint'] == endpoint for r in assignments(request('sessions'))):
                raise RuntimeError('ENDPOINT_STILL_PRESENT')
        phase(directory, config, 'RELEASED', endpoint)
        result.update(status='RELEASED', owned_released=True, attention='NONE', released_endpoint=endpoint)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result['error'] = type(error).__name__  # No raw endpoint/transport/approval content in output.
    guard.write_json(result_path, result)
    return result


def watch(directory, config, parent_pid):
    (directory/'watch-ready').touch()
    trigger = guard.wait_for_trigger(directory, config.deadline, 1.5, parent_pid)
    result = clean_owned(directory, config, FakeTransport(directory, config), trigger)
    return 0 if result['owned_released'] else 1


def mock_setup(directory, scenario, seconds=4.):
    if scenario not in SCENARIOS or not positive(seconds) or not 2.5 <= seconds <= 5:
        raise ValueError('INVALID_MOCK_SCENARIO_OR_DEADLINE')
    if not directory.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError('PRIVATE_LOCAL_TEMP_REQUIRED')
    directory.mkdir(parents=True, exist_ok=False); directory.chmod(0o700)
    start = time.monotonic()
    config = ProbeConfig('v5e1', 1, 5., 1., 1., True, BASE_SHA,
                         'torchyand-mock-'+uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex,
                         'mock://scientist/'+uuid.uuid4().hex, 'mock://rate-bound/'+uuid.uuid4().hex, start+seconds)
    record = dict(reference=config.approval_ref, decision='APPROVED', device='v5e1',
                  source_sha=config.source_sha, max_sessions=1, max_minutes=5., max_cu=1.,
                  rate_upper_bound=1., rate_ref=config.rate_ref, residual_soft_cap_risk_ack=True, mock_only=True)
    guard.write_json(directory/'journal.json', dict(schema=1, mock_only=True, config=asdict(config),
        phase='PRECHECK', endpoint=None, started_utc=datetime.now(timezone.utc).isoformat(),
        started_monotonic=start, history=['PRECHECK']))
    guard.write_json(directory/'server.json', dict(scenario=scenario, rows=[dict(name='unrelated', endpoint='other-endpoint')]))
    return config, record


def launch_watchdog(directory, config):
    return subprocess.Popen([sys.executable, '-c', CHILD_CODE, str(ROOT), 'watch',
                             str(directory), json.dumps(asdict(config)), str(os.getpid())],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def simulate(directory, scenario='success'):
    config, record = mock_setup(directory, scenario)
    watch_process = None
    outcome = 'BLOCKED'
    try:
        if scenario == 'invalid-approval':
            record['decision'] = 'PENDING'
        if scenario in ('missing-rate', 'missing-budget'):
            record.pop('rate_upper_bound' if scenario == 'missing-rate' else 'max_cu')
        approval_check(config, record, BASE_SHA)
        phase(directory, config, 'APPROVAL_CHECK')
        transport = FakeTransport(directory, config)
        if transport.call('version', .5) != {'version': '0.7.4'}:
            raise ValueError('CLI_VERSION_BLOCKED')
        # Named ID must not already exist; no first-entry or broad session inference.
        if any(r['name'] == config.session for r in assignments(transport.call('sessions', .5))):
            raise ValueError('SESSION_NAME_COLLISION')
        watch_process = launch_watchdog(directory, config)
        while not (directory/'watch-ready').exists():
            if watch_process.poll() is not None or time.monotonic() >= config.deadline-1.5:
                raise RuntimeError('WATCHDOG_NOT_READY')
            time.sleep(.02)
        phase(directory, config, 'NAMED_ALLOCATION_PENDING')
        outcome = 'ALLOCATION_UNCERTAIN'
        response = transport.call('allocate', .5)
        if not owned(response, config) or response.get('name') != config.session:
            raise ValueError('ALLOCATION_RESPONSE_UNPROVABLE')
        phase(directory, config, 'ALLOCATED', response['endpoint'])
        outcome = 'DEVICE_VERIFY_FAILED'
        phase(directory, config, 'DEVICE_VERIFY', response['endpoint'])
        identity = transport.call('identity', .5)
        if identity != dict(accelerator='v5e1', PJRT='TPU', real_device='TPU:0'):
            raise ValueError('DEVICE_IDENTITY_BLOCKED')
        if scenario == 'journal-corrupt':
            (directory/'journal.json').write_text('{broken', encoding='utf-8')
        if scenario == 'watchdog-death':
            outcome = 'WATCHDOG_FAILED'
            watch_process.kill(); watch_process.wait(timeout=1)
            raise RuntimeError('WATCHDOG_DIED')
        if scenario == 'interrupt':
            outcome = 'INTERRUPTED'
            raise KeyboardInterrupt
        phase(directory, config, 'TINY_PROBE', response['endpoint'])
        outcome = 'PROBE_FAILED'
        if transport.call('probe', min(.5, max(.01, config.deadline-time.monotonic()-1.5))) != {'value': 11}:
            raise ValueError('CPU_MOCK_PROBE_MISMATCH')
        outcome = 'MOCK_PROBE_PASS'
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        error_kind = type(error).__name__
    finally:
        if watch_process is not None:
            (directory/'done').touch()
            try:
                watch_process.wait(timeout=max(.1, config.deadline-time.monotonic()+.5))
            except subprocess.TimeoutExpired:
                watch_process.kill(); watch_process.wait(timeout=1)
            if not (directory/'adapter-result.json').exists():
                # Secondary finally guard if watchdog died; never reports watchdog as healthy.
                result = clean_owned(directory, config, FakeTransport(directory, config), 'WATCHDOG_UNAVAILABLE')
            else:
                result = json.loads((directory/'adapter-result.json').read_text())
        else:
            result = dict(mock_only=True, status='BLOCKED', owned_released=False,
                          REAL_SESSION_STOP='NOT RUN', TPU_ALLOCATED='NO', session_count='UNKNOWN')
        result.update(outcome=outcome, error=locals().get('error_kind'), LIVE_ADAPTER_ENABLED='NO',
                      WATCHDOG_MOCK='FAILED' if result.get('trigger') == 'WATCHDOG_UNAVAILABLE' else 'OBSERVED',
                      approval_authority='MOCK_ONLY_NOT_HUMAN_AUTHORIZATION',
                      HARD_STOP_GUARANTEE='NOT_PROVEN', HOST_OR_NETWORK_FAILURE_NOT_COVERED=True)
        if result['WATCHDOG_MOCK'] == 'FAILED':
            result['attention'] = 'MANUAL_ATTENTION_REQUIRED'
        guard.write_json(directory/'summary.json', result)
    return result


def fake(directory, operation, plan):
    journal = json.loads((directory/'journal.json').read_text())
    config = ProbeConfig(**journal['config'])
    if tuple(plan) != argv_for(operation, config):
        return 2
    server = json.loads((directory/'server.json').read_text())
    scenario, rows = server['scenario'], server['rows']
    with (directory/'commands.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(mock_only=True, operation=operation, argv=plan))+'\n')
    if operation == 'version':
        if scenario == 'missing-cli': return 127
        reply = {'version': '0.0.0' if scenario == 'wrong-version' else '0.7.4'}
    elif operation == 'allocate':
        row = dict(endpoint='mock-endpoint-'+uuid.uuid4().hex, name=config.session,
                   owner=config.owner, correlation=config.correlation, source_sha=config.source_sha)
        if scenario in ('allocation-lost', 'orphan'):
            row.pop('correlation'); row['name'] = '?'
        rows.append(row); guard.write_json(directory/'server.json', server)
        if scenario in ('allocation-lost', 'allocation-provable'):
            return 1  # Lost response; allocation might have succeeded, no retry.
        if scenario == 'allocation-hang': time.sleep(60)
        reply = row
    elif operation == 'sessions':
        allocated = len(rows) > 1 or (directory/'stop-attempted').exists()
        if allocated and (scenario in ('server-403', 'server-500', 'rate-limit', 'stale-credentials')
                          or scenario == 'verify-errors' and (directory/'stop-attempted').exists()):
            return 1
        if scenario == 'schema-drift':
            reply = {'changed': rows}
        else:
            reply = {'assignments': rows}
    elif operation == 'identity':
        reply = dict(accelerator='v6e1' if scenario == 'wrong-device' else 'v5e1',
                     PJRT='CPU' if scenario == 'no-device' else 'TPU', real_device='TPU:0')
    elif operation == 'probe':
        if scenario in ('worker-hang', 'compile-hang'): time.sleep(60)
        if scenario == 'worker-failure': return 1
        if scenario == 'name-lost':
            rows[-1]['name'] = '?'; guard.write_json(directory/'server.json', server)
        if scenario == 'orphan': return 1
        reply = {'value': sum(a*b for a, b in zip((1, 2), (3, 4)))}
    elif operation == 'stop':
        (directory/'stop-attempted').touch()
        if scenario == 'stop-nonzero': return 1
        if scenario == 'stop-slow': time.sleep(60)
        if scenario != 'stop-lie':
            server['rows'] = [r for r in rows if not (r.get('name') == config.session and owned(r, config))]
            guard.write_json(directory/'server.json', server)
        reply = {'text': 'Session terminated.'}
    else:
        return 2
    print(json.dumps(reply))
    return 0


def child(argv):
    if argv[0] == 'fake':
        return fake(Path(argv[1]), argv[2], json.loads(argv[3]))
    if argv[0] == 'watch':
        return watch(Path(argv[1]), ProbeConfig(**json.loads(argv[2])), int(argv[3]))
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.live:
        parser.error('LIVE_ADAPTER_DISABLED: separate reviewed enablement and human approval required')
    if args.scenario:
        if args.output_dir is None:
            parser.error('local OS temp --output-dir required')
        if (not args.output_dir.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
                or args.output_dir.exists()):
            parser.error('new private local OS temp directory required')
        result = simulate(args.output_dir, args.scenario)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'RELEASED' and result['outcome'] == 'MOCK_PROBE_PASS' else 1
    if args.output_dir is not None:
        parser.error('--output-dir requires --scenario')
    print(json.dumps(dict(mode='DRY_RUN', LIVE_ADAPTER_ENABLED='NO', device='v5e1',
                         first_probe_max_minutes=5, max_cu=None, PROSPECTIVE_CU_RATE='UNKNOWN',
                         CU_CAP_ENFORCEABILITY='BLOCKED', HARD_STOP_GUARANTEE='NOT_PROVEN')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
