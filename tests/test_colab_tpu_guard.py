"""LOCAL MOCKS ONLY: no real colab executable or resource is ever invoked."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

SOURCE = Path(__file__).resolve().parents[1]/'tools/colab_tpu_guard.py'
spec = importlib.util.spec_from_file_location('offline_guard', SOURCE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


@pytest.mark.parametrize('scenario,expected', [
    ('success', 'MOCK_RELEASE_VERIFIED'), ('timeout', 'MOCK_RELEASE_VERIFIED'),
    ('crash', 'MOCK_RELEASE_VERIFIED'), ('cancel', 'MOCK_RELEASE_VERIFIED'),
    ('interrupt', 'MOCK_RELEASE_VERIFIED'), ('stop-failure', 'CLEANUP_FAILED'),
    ('verify-failure', 'CLEANUP_FAILED'), ('hung-stop', 'CLEANUP_FAILED'),
    ('missing-metadata', 'CLEANUP_FAILED'), ('stale-metadata', 'CLEANUP_FAILED'),
    ('preflight-veto', 'BLOCKED'), ('missing-budget', 'BLOCKED')])
def test_mock_scenarios(tmp_path, scenario, expected):
    directory = tmp_path/scenario
    start = time.monotonic()
    result = g.simulate(directory, scenario)
    assert time.monotonic()-start < 4
    assert result['mock_only'] and result['status'] == expected
    assert result['real_session_stop'] == 'NOT RUN'
    assert result['session_count'] == 'UNKNOWN'
    if expected == 'CLEANUP_FAILED':
        assert result['attention'] == 'MANUAL_ATTENTION_REQUIRED'
    server = json.loads((directory/'server.json').read_text())
    assert dict(name='unrelated-mock-session', owner='other') in server
    commands = directory/'mock-commands.jsonl'
    rows = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
    assert all(row['mock_only'] and row['verb'] in ('sessions', 'stop')
               and row['session'].startswith('torchyand-mock-') for row in rows)
    assert sum(r['verb'] == 'stop' for r in rows) <= 1
    if scenario in ('missing-metadata', 'stale-metadata', 'preflight-veto', 'missing-budget'):
        assert rows == []
    if scenario == 'timeout':
        assert result['trigger'] == 'DEADLINE'


def test_default_plan_and_live_veto_no_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No process may run for plan/live rejection')
    monkeypatch.setattr(g.subprocess, 'Popen', forbidden)
    assert g.main([]) == 0
    with pytest.raises(SystemExit) as error:
        g.main(['--live'])
    assert error.value.code == 2


def test_command_argv_injected_fake_executable(tmp_path):
    copied = tmp_path/'injected fake.py'
    copied.write_bytes(SOURCE.read_bytes())
    captured = []
    def runner(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, '[]', '')
    cli = g.FakeCLI(tmp_path, copied, runner=runner)
    session, owner = 'torchyand-mock-'+'a'*32, 'b'*32
    cli.call('stop', session, owner, .1)
    argv, kwargs = captured[0]
    assert argv == [sys.executable, str(copied), '_fake', str(tmp_path), 'stop',
                    '--session', session, '--owner', owner]
    assert kwargs['shell'] is False and kwargs['timeout'] == .1
    assert 'colab' not in argv  # only local Python and fake CLI code
    for verb, name in [('new', session), ('exec', session), ('stop', 'x; echo bad')]:
        with pytest.raises(ValueError):
            cli.call(verb, name, owner, .1)
    copied.write_text('print("untrusted")')
    with pytest.raises(ValueError):
        g.FakeCLI(tmp_path, copied)


def test_mock_idempotency_and_owner_veto(tmp_path):
    directory = tmp_path/'idempotent'
    state = g.prepare(directory, 'success', 3)
    cli = g.FakeCLI(directory)
    a = g.cleanup(directory, state, state['owner'], state['session'], state['deadline'], cli)
    log = (directory/'mock-commands.jsonl').read_text()
    b = g.cleanup(directory, state, state['owner'], state['session'], state['deadline'], cli)
    assert a == b and (directory/'mock-commands.jsonl').read_text() == log
    foreign = dict(state, owner='f'*32)
    c = g.cleanup(directory, foreign, state['owner'], state['session'], state['deadline'], cli)
    assert c['status'] == 'CLEANUP_FAILED'
    assert (directory/'mock-commands.jsonl').read_text() == log


def test_mock_server_owner_mismatch_never_stops(tmp_path):
    state = g.prepare(tmp_path/'mismatch', 'success', 3)
    directory = tmp_path/'mismatch'
    g.write_json(directory/'server.json', [dict(name=state['session'], owner='foreign')])
    result = g.cleanup(directory, state, state['owner'], state['session'], state['deadline'], g.FakeCLI(directory))
    assert result['status'] == 'CLEANUP_FAILED'
    assert '"stop"' not in (directory/'mock-commands.jsonl').read_text()


def test_independent_watchdog_survives_real_parent_termination(tmp_path):
    directory = tmp_path/'parent-death'
    code = (
        'import importlib.util,pathlib,time; '
        f's=importlib.util.spec_from_file_location("guard",{str(SOURCE)!r}); '
        'g=importlib.util.module_from_spec(s); s.loader.exec_module(g); '
        f'd=pathlib.Path({str(directory)!r}); '
        'state=g.prepare(d,"success",4); g.start_watchdog(d,state); '
        '(d/"ready").touch(); time.sleep(60)'
    )
    parent = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.monotonic()+3
        while not (directory/'ready').exists() and time.monotonic() < end:
            time.sleep(.02)
        assert (directory/'ready').exists()
        parent.kill(); parent.wait(timeout=1)
        end = time.monotonic()+4
        while not (directory/'result.json').exists() and time.monotonic() < end:
            time.sleep(.02)
        result = json.loads((directory/'result.json').read_text())
        assert result['status'] == 'MOCK_RELEASE_VERIFIED'
        assert result['trigger'] == 'PARENT_TERMINATED'
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=1)


def test_budget_and_deadline_validation(tmp_path):
    for seconds in (0., float('inf'), 10.):
        with pytest.raises(ValueError):
            g.prepare(tmp_path/'invalid', 'success', seconds)
    state = g.prepare(tmp_path/'budget', 'success', 3)
    state.pop('max_cu')
    result = g.cleanup(tmp_path/'budget', state, state['owner'], state['session'], state['deadline'], g.FakeCLI(tmp_path/'budget'))
    assert result['status'] == 'CLEANUP_FAILED'
    assert not (tmp_path/'budget/mock-commands.jsonl').exists()
