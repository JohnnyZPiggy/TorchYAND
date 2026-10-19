"""OFFLINE MOCK ONLY. Real Colab commands/credentials/TPU are never used."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from tools import tpu_safe_adapter as a


def log(directory):
    path = directory/'commands.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('scenario,expected', [
    ('success', 'RELEASED'), ('allocation-lost', 'ALLOCATION_UNCERTAIN'),
    ('allocation-provable', 'RELEASED'), ('allocation-hang', 'RELEASED'),
    ('wrong-device', 'RELEASED'), ('no-device', 'RELEASED'),
    ('worker-failure', 'RELEASED'), ('worker-hang', 'RELEASED'), ('compile-hang', 'RELEASED'),
    ('interrupt', 'RELEASED'), ('name-lost', 'CLEANUP_FAILED'), ('orphan', 'ALLOCATION_UNCERTAIN'),
    ('stale-credentials', 'CLEANUP_FAILED'), ('stop-nonzero', 'CLEANUP_FAILED'),
    ('stop-lie', 'CLEANUP_FAILED'), ('stop-slow', 'CLEANUP_FAILED'),
    ('server-403', 'CLEANUP_FAILED'), ('server-500', 'CLEANUP_FAILED'),
    ('rate-limit', 'CLEANUP_FAILED'), ('verify-errors', 'CLEANUP_FAILED'),
    ('schema-drift', 'BLOCKED'), ('journal-corrupt', 'CLEANUP_FAILED'),
    ('watchdog-death', 'RELEASED'), ('invalid-approval', 'BLOCKED'),
    ('missing-rate', 'BLOCKED'), ('missing-budget', 'BLOCKED'),
    ('missing-cli', 'BLOCKED'), ('wrong-version', 'BLOCKED')])
def test_mock_lifecycle(tmp_path, scenario, expected):
    directory = tmp_path/scenario
    result = a.simulate(directory, scenario)
    assert result['status'] == expected
    assert result['mock_only'] and result['LIVE_ADAPTER_ENABLED'] == 'NO'
    assert result['TPU_ALLOCATED'] == 'NO' and result['REAL_SESSION_STOP'] == 'NOT RUN'
    assert result['HARD_STOP_GUARANTEE'] == 'NOT_PROVEN'
    server = json.loads((directory/'server.json').read_text())
    assert dict(name='unrelated', endpoint='other-endpoint') in server['rows']
    calls = log(directory)
    assert all(row['mock_only'] for row in calls)
    assert sum(r['operation'] == 'allocate' for r in calls) <= 1
    assert sum(r['operation'] == 'stop' for r in calls) <= 1
    if scenario in ('invalid-approval', 'missing-rate', 'missing-budget'):
        assert not calls
    if scenario in ('wrong-device', 'no-device', 'allocation-lost', 'allocation-provable', 'allocation-hang'):
        assert not any(r['operation'] == 'probe' for r in calls)
    if scenario in ('wrong-device', 'no-device'):
        assert result['outcome'] == 'DEVICE_VERIFY_FAILED'
    if scenario in ('allocation-lost', 'orphan', 'name-lost', 'journal-corrupt'):
        assert not any(r['operation'] == 'stop' for r in calls)
    if scenario == 'success':
        assert result['outcome'] == 'MOCK_PROBE_PASS'
        history = json.loads((directory/'journal.json').read_text())['history']
        assert history == ['PRECHECK', 'APPROVAL_CHECK', 'NAMED_ALLOCATION_PENDING', 'ALLOCATED',
                           'DEVICE_VERIFY', 'TINY_PROBE', 'STOP_OWNED', 'VERIFY_RELEASE', 'RELEASED']
    if scenario == 'watchdog-death':
        assert result['WATCHDOG_MOCK'] == 'FAILED'
        assert result['outcome'] != 'MOCK_PROBE_PASS'
    if scenario == 'verify-errors':
        assert any(r['operation'] == 'stop' for r in calls)
        assert sum(r['operation'] == 'sessions' for r in calls) == 3  # preflight, pre-stop, failed verify


@pytest.mark.parametrize('device', ['v6e1', 'V5E1', '', 'v5e-1', 'v5el', 'cpu', 'gpu', None])
def test_exact_device_allowlist_before_transport(tmp_path, device):
    config, record = a.mock_setup(tmp_path/'device', 'success')
    with pytest.raises(ValueError):
        a.approval_check(replace(config, device=device), record, a.BASE_SHA)


@pytest.mark.parametrize('key,value', [('max_cu', 0), ('max_cu', float('nan')), ('max_cu', True),
    ('rate_upper_bound', None), ('rate_upper_bound', float('inf')), ('max_sessions', 2),
    ('max_sessions', True), ('max_minutes', 20), ('max_minutes', 0),
    ('zero_allocation_retries', False), ('source_sha', 'main'), ('owner', ''),
    ('session', 'x; echo bad'), ('correlation', ''), ('approval_ref', 'https://example.org/approval')])
def test_config_fail_closed(tmp_path, key, value):
    config, record = a.mock_setup(tmp_path/'invalid', 'success')
    with pytest.raises(ValueError):
        a.approval_check(replace(config, **{key: value}), record, a.BASE_SHA)


def test_approval_review_risk_and_immutability(tmp_path):
    config, record = a.mock_setup(tmp_path/'approval', 'success')
    with pytest.raises(FrozenInstanceError):
        config.device = 'v6e1'
    for changed in (dict(record, residual_soft_cap_risk_ack=False),
                    dict(record, decision='PENDING'), dict(record, mock_only=False),
                    dict(record, max_cu=True)):
        with pytest.raises(ValueError):
            a.approval_check(config, changed, a.BASE_SHA)
    with pytest.raises(ValueError):
        a.approval_check(config, record, '0'*40)
    with pytest.raises(ValueError):
        a.approval_check(replace(config, max_cu=.0001), record, a.BASE_SHA)
    for changed in (replace(config, rate_ref=''), replace(config, deadline=time.monotonic()+3600)):
        with pytest.raises(ValueError):
            a.approval_check(changed, record, a.BASE_SHA)


@pytest.mark.parametrize('argv', [['--live'], ['--live', '--scenario', 'success'], ['--force'],
    ['--device', 'v6e1'], ['--scenario', 'unknown'], ['--scenario', 'success'],
    ['--dry-run', '--output-dir'], ['_watch', 'bad'], ['--scenario']])
def test_malformed_or_live_spawn_zero_processes(monkeypatch, argv):
    def veto(*args, **kwargs):
        pytest.fail('CLI validation may not spawn any process')
    monkeypatch.setattr(subprocess, 'Popen', veto)
    monkeypatch.setattr(subprocess, 'run', veto)
    with pytest.raises(SystemExit) as error:
        a.main(argv)
    assert error.value.code == 2


def test_default_plan_zero_processes(monkeypatch):
    def veto(*args, **kwargs):
        pytest.fail('dry-run plan must not start transport')
    monkeypatch.setattr(subprocess, 'Popen', veto)
    monkeypatch.setattr(subprocess, 'run', veto)
    assert a.main([]) == 0


@pytest.mark.parametrize('module', [a, a.guard])
@pytest.mark.parametrize('argv', [['--live'], ['--force'], ['--scenario', 'bad'], ['--scenario', 'success']])
def test_both_public_clis_reject_before_dispatch(monkeypatch, module, argv):
    def veto(*args, **kwargs):
        pytest.fail('No real or fake dispatch during argument rejection')
    monkeypatch.setattr(subprocess, 'Popen', veto)
    monkeypatch.setattr(subprocess, 'run', veto)
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2


def test_bad_output_path_never_starts_process(monkeypatch):
    def veto(*args, **kwargs):
        pytest.fail('Bad path started transport')
    monkeypatch.setattr(subprocess, 'Popen', veto)
    monkeypatch.setattr(subprocess, 'run', veto)
    with pytest.raises(SystemExit) as error:
        a.main(['--scenario', 'success', '--output-dir', str(a.ROOT/'forbidden-state')])
    assert error.value.code == 2


def test_fake_dispatch_never_executes_planned_colab_argv(tmp_path):
    config, _ = a.mock_setup(tmp_path/'argv', 'success')
    seen = []
    def runner(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, '{"version":"0.7.4"}', '')
    a.FakeTransport(tmp_path/'argv', config, runner).call('allocate', .2)
    argv, kwargs = seen[0]
    assert argv[:2] == [sys.executable, '-c'] and argv[2] == a.CHILD_CODE
    assert argv[4] == 'fake'
    assert kwargs['shell'] is False and kwargs['timeout'] <= .2
    assert json.loads(argv[-1]) == ['colab', 'new', '-s', config.session, '--tpu', 'v5e1']
    with pytest.raises(ValueError):
        a.argv_for('ssh', config)


def test_private_journal_persisted_before_allocation(tmp_path):
    directory = tmp_path/'journal'
    config, _ = a.mock_setup(directory, 'success')
    seen = []
    def runner(argv, **kwargs):
        state = a.journal_read(directory, config)
        seen.append(state)
        return subprocess.CompletedProcess(argv, 1, '', 'synthetic secret must not escape')
    a.phase(directory, config, 'NAMED_ALLOCATION_PENDING')
    with pytest.raises(RuntimeError) as error:
        a.FakeTransport(directory, config, runner).call('allocate', .2)
    assert 'synthetic secret' not in str(error.value)
    assert seen[0]['phase'] == 'NAMED_ALLOCATION_PENDING'
    assert seen[0]['endpoint'] is None
    if os.name != 'nt':
        assert (directory/'journal.json').stat().st_mode & 0o777 == 0o600


def test_corrupt_endpoint_does_not_report_release(tmp_path):
    directory = tmp_path/'tamper'
    config, _ = a.mock_setup(directory, 'success')
    a.phase(directory, config, 'NAMED_ALLOCATION_PENDING')
    transport = a.FakeTransport(directory, config)
    row = transport.call('allocate', .4)
    a.phase(directory, config, 'ALLOCATED', 'mock-endpoint-'+'0'*32)
    result = a.clean_owned(directory, config, transport, 'TEST')
    assert result['status'] == 'CLEANUP_FAILED'
    assert not any(r['operation'] == 'stop' for r in log(directory))


def test_cleanup_idempotent(tmp_path):
    directory = tmp_path/'idempotent'
    result = a.simulate(directory)
    config = a.ProbeConfig(**json.loads((directory/'journal.json').read_text())['config'])
    before = log(directory)
    cached = a.clean_owned(directory, config, a.FakeTransport(directory, config), 'AGAIN')
    assert cached['owned_released'] and log(directory) == before
    journal = a.journal_read(directory, config)
    journal['endpoint'] = 'mock-endpoint-'+'f'*32
    a.guard.write_json(directory/'journal.json', journal)
    assert a.clean_owned(directory, config, a.FakeTransport(directory, config), 'AGAIN')['status'] == 'CLEANUP_FAILED'
    assert log(directory) == before
    foreign = dict(cached, identity={})
    a.guard.write_json(directory/'adapter-result.json', foreign)
    assert a.clean_owned(directory, config, a.FakeTransport(directory, config), 'AGAIN')['status'] == 'CLEANUP_FAILED'
    assert log(directory) == before


def test_repeated_verification_failure_never_retries(tmp_path):
    directory = tmp_path/'repeat-error'
    assert a.simulate(directory, 'verify-errors')['status'] == 'CLEANUP_FAILED'
    config = a.ProbeConfig(**json.loads((directory/'journal.json').read_text())['config'])
    before = log(directory)
    for _ in range(2):
        assert a.clean_owned(directory, config, a.FakeTransport(directory, config), 'AGAIN')['status'] == 'CLEANUP_FAILED'
    assert log(directory) == before


def test_independent_adapter_watchdog_after_parent_death(tmp_path):
    directory = tmp_path/'parent'
    code = (
        'import sys,pathlib,time; sys.path.insert(0,sys.argv[1]); '
        'from tools import tpu_safe_adapter as a; d=pathlib.Path(sys.argv[2]); '
        'c,r=a.mock_setup(d,"success",5); a.approval_check(c,r,a.BASE_SHA); '
        'p=a.launch_watchdog(d,c); '
        'a.phase(d,c,"NAMED_ALLOCATION_PENDING"); '
        'response=a.FakeTransport(d,c).call("allocate",.5); '
        'a.phase(d,c,"ALLOCATED",response["endpoint"]); '
        '(d/"parent-ready").touch(); time.sleep(60)'
    )
    process = subprocess.Popen([sys.executable, '-c', code, str(a.ROOT), str(directory)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.monotonic()+3
        while not (directory/'parent-ready').exists() and time.monotonic() < end:
            time.sleep(.02)
        assert (directory/'parent-ready').exists()
        process.kill(); process.wait(timeout=1)
        end = time.monotonic()+4
        while not (directory/'adapter-result.json').exists() and time.monotonic() < end:
            time.sleep(.02)
        result = json.loads((directory/'adapter-result.json').read_text())
        assert result['status'] == 'RELEASED' and result['trigger'] == 'PARENT_TERMINATED'
        assert result['REAL_SESSION_STOP'] == 'NOT RUN'
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=1)


def test_adapter_deadline_while_parent_is_wedged(tmp_path):
    directory = tmp_path/'wedged-parent'
    code = (
        'import sys,pathlib,time; sys.path.insert(0,sys.argv[1]); '
        'from tools import tpu_safe_adapter as a; d=pathlib.Path(sys.argv[2]); '
        'c,r=a.mock_setup(d,"success",4); a.approval_check(c,r,a.BASE_SHA); '
        'p=a.launch_watchdog(d,c); a.phase(d,c,"NAMED_ALLOCATION_PENDING"); '
        'response=a.FakeTransport(d,c).call("allocate",.5); '
        'a.phase(d,c,"ALLOCATED",response["endpoint"]); time.sleep(60)'
    )
    parent = subprocess.Popen([sys.executable, '-c', code, str(a.ROOT), str(directory)],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.monotonic()+5
        while not (directory/'adapter-result.json').exists() and time.monotonic() < end:
            time.sleep(.02)
        assert parent.poll() is None  # No parent finally ran; independent process did cleanup.
        result = json.loads((directory/'adapter-result.json').read_text())
        assert result['status'] == 'RELEASED' and result['trigger'] == 'DEADLINE'
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=1)
