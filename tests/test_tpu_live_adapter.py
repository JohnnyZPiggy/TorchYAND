"""CPU/MOCK ONLY. No installed Colab entrypoint, account or TPU is invoked."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from tools import tpu_live_adapter as a

SHA = 'a'*40
APPROVAL = 'https://github.com/JohnnyZPiggy/TorchYAND/issues/7#issuecomment-1'
INSTRUCTION = 'https://github.com/JohnnyZPiggy/TorchYAND/issues/7#issuecomment-2'


def record():
    return dict(source_sha=SHA, device='v5e1', max_sessions=1, max_minutes=5,
                zero_retries=True, supervised=True, decision='APPROVED',
                residual_outage_risk_ack=True, quota_policy='price_not_a_gate_no_purchase', approval_ref=APPROVAL)


def permit():
    return a.authorize(record(), INSTRUCTION, SHA, SHA, False, str(Path(sys.executable).resolve()), platform='linux')


def line(name, endpoint, hardware='V5E1'):
    return f'[{name}] {endpoint} | Hardware: {hardware} | Shape: Standard | Variant: TPU'


class FakeCLI:
    def __init__(self, directory, scenario='success'):
        self.directory, self.scenario = directory, scenario
        self.rows = [line('?', 'unrelated')]
        self.calls = []
    def call(self, op):
        self.calls.append(op)
        data = a.state(self.directory)
        if op == 'version': return 'Version: 0.0.0' if self.scenario == 'version' else 'Version: 0.7.4'
        if op == 'sessions':
            if self.scenario == 'outage' and 'allocate' in self.calls: raise RuntimeError('mock backend outage')
            return '\n'.join(self.rows) if self.rows else a.EMPTY
        if op == 'allocate':
            assert data['phase'] == 'NAMED_ALLOCATION_PENDING'
            assert (self.directory/'watch-ready').exists()
            assert self.calls.count('allocate') == 1
            hardware = 'V6E1' if self.scenario == 'wrong-device' else 'V5E1'
            name = '?' if self.scenario == 'orphan' else data['name']
            self.rows.append(line(name, 'owned-endpoint', hardware))
            if self.scenario in ('lost', 'orphan'): raise subprocess.TimeoutExpired('fake', 1)
            return '[colab] Session READY.'
        if op == 'probe':
            if self.scenario == 'hang': raise subprocess.TimeoutExpired('fake', 1)
            if self.scenario == 'interrupt': raise KeyboardInterrupt
            if self.scenario == 'probe-failed': return 'Traceback: missing XLA'
            if self.scenario == 'name-lost': self.rows[-1] = line('?', 'owned-endpoint')
            if self.scenario == 'journal-corrupt': (self.directory/'live-journal.json').write_text('{broken')
            return 'TORCHYAND_TPU_PROBE:'+json.dumps(dict(nonce=data['name'],status='PASS',PJRT='TPU',
                 real_devices=['TPU:0'],result=[[3.],[7.]],torch='2.8.0',torch_xla='2.8.0'))
        if op == 'stop':
            assert data['endpoint'] == 'owned-endpoint'
            if self.scenario == 'stop-failed': raise RuntimeError('fake failed stop')
            if self.scenario != 'stop-lie': self.rows = self.rows[:1]
            return 'Session terminated.'
        raise AssertionError('forbidden op')


class FakeWatch:
    def __init__(self, directory, cli):
        self.directory, self.cli = directory, cli
        (directory/'watch-ready').touch()
    def poll(self): return None
    def wait(self, timeout): a.release(self.directory, self.cli, 'MOCK_FINALLY'); return 0
    def kill(self): raise AssertionError('not needed')


@pytest.mark.parametrize('scenario,outcome,cleanup', [
    ('success','TPU_PROBE_PASS','RELEASED'),('lost','ALLOCATION_UNCERTAIN','RELEASED'),
    ('orphan','ALLOCATION_UNCERTAIN','CLEANUP_FAILED'),('wrong-device','DEVICE_OR_PROBE_BLOCKED','RELEASED'),
    ('probe-failed','DEVICE_OR_PROBE_BLOCKED','RELEASED'),('hang','DEVICE_OR_PROBE_BLOCKED','RELEASED'),
    ('interrupt','DEVICE_OR_PROBE_BLOCKED','RELEASED'),('name-lost','TPU_PROBE_PASS','CLEANUP_FAILED'),
    ('stop-lie','TPU_PROBE_PASS','CLEANUP_FAILED'),('stop-failed','TPU_PROBE_PASS','CLEANUP_FAILED'),
    ('outage','ALLOCATION_UNCERTAIN','CLEANUP_FAILED'),('version','BLOCKED','NOT_NEEDED'),
    ('journal-corrupt','TPU_PROBE_PASS','CLEANUP_FAILED')])
def test_lifecycle_with_realistic_cli_text_only(tmp_path, scenario, outcome, cleanup):
    directory = tmp_path/scenario
    cli = FakeCLI(directory, scenario)
    result = a.run(permit(), record(), directory, cli_factory=lambda _: cli,
                   watch_factory=lambda d: FakeWatch(d, cli))
    assert result['outcome'] == outcome and result['cleanup']['status'] == cleanup
    assert cli.rows[0] == line('?', 'unrelated')
    assert cli.calls.count('allocate') <= 1 and cli.calls.count('stop') <= 1
    if scenario in ('orphan','name-lost','journal-corrupt'): assert 'stop' not in cli.calls
    if scenario in ('lost','orphan','wrong-device'): assert 'probe' not in cli.calls


@pytest.mark.parametrize('field,value', [('device','V5E1'),('device','v6e1'),('device',''),
    ('max_sessions',True),('max_sessions',2),('max_minutes',6),('max_minutes',0),
    ('zero_retries',False),('source_sha','main'),('decision','PENDING'),
    ('quota_policy','unlimited'),('residual_outage_risk_ack',False)])
def test_resource_gate(field,value):
    data = record(); data[field]=value
    with pytest.raises(ValueError): a.authorize(data,INSTRUCTION,SHA,SHA,False,'/venv/bin/python',platform='linux')


def test_second_step_merged_source_and_platform():
    for instruction,merged,dirty,platform in [(APPROVAL,SHA,False,'linux'),(None,SHA,False,'linux'),
                                            (INSTRUCTION,'b'*40,False,'linux'),(INSTRUCTION,SHA,True,'linux'),
                                            (INSTRUCTION,SHA,False,'win32')]:
        with pytest.raises(ValueError): a.authorize(record(),instruction,SHA,merged,dirty,'/venv/bin/python',platform=platform)
    assert permit().seconds == 300  # No price, balance or rate input required.


@pytest.mark.parametrize('argv',[[],['--live'],['--live','--instruction-ref',INSTRUCTION],
                                ['--force'],['--approval-file','missing'],['--live','--device','v6e1']])
def test_default_and_incomplete_activation_zero_subprocess(monkeypatch,argv):
    def veto(*args,**kwargs): pytest.fail('No process permitted')
    monkeypatch.setattr(subprocess,'Popen',veto); monkeypatch.setattr(subprocess,'run',veto)
    if not argv: assert a.main(argv)==0
    else:
        with pytest.raises(SystemExit) as error: a.main(argv)
        assert error.value.code==2


@pytest.mark.parametrize('output',["",'garbage',a.EMPTY+'\nunknown',
    line('x','ep')+'\n'+line('y','ep')])
def test_schema_fail_closed(output):
    with pytest.raises(ValueError): a.sessions(output)


def test_tagged_session_format_and_orphan():
    assert a.sessions(a.EMPTY)==[]
    assert a.sessions('[colab] Pruned 1 stale local session(s).\n'+a.EMPTY)==[]
    assert a.sessions(line('?','orphan'))[0]['name']=='?'


def test_real_transport_dispatch_is_injected_and_bounded(tmp_path):
    directory = tmp_path/'transport'; cli = FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    seen=[]
    def runner(argv,**kwargs): seen.append((argv,kwargs));return subprocess.CompletedProcess(argv,0,a.EMPTY,'')
    assert a.CLI(directory,runner).call('sessions')==a.EMPTY
    argv,kwargs=seen[0]
    assert argv[1:3]==['-c',a.CHILD] and argv[-1]=='sessions'
    assert kwargs['shell'] is False and kwargs['stdin']==subprocess.DEVNULL and kwargs['timeout']<=30
    assert a.child(['invalid']) == 2


def test_wrong_journal_endpoint_cannot_fake_release(tmp_path):
    directory=tmp_path/'tamper';cli=FakeCLI(directory,'stop-lie')
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    data=a.state(directory);data['endpoint']='absent-foreign';a.save(directory,data)
    before=cli.calls.count('stop')
    assert a.release(directory,cli,'TEST')['status']=='CLEANUP_FAILED'
    assert cli.calls.count('stop')==before


def test_probe_code_cpu_failure_and_version_metadata(monkeypatch,capsys):
    # No XLA installation/import: local mock import rejects it, no TPU initialization.
    original=__import__('builtins').__import__
    def blocked(name,*args,**kwargs):
        if name=='torch_xla': raise ImportError('mock missing XLA')
        return original(name,*args,**kwargs)
    monkeypatch.setattr('builtins.__import__',blocked)
    monkeypatch.setenv('PJRT_DEVICE', 'CPU')
    exec(a.PROBE.replace('NONCE',repr('test-nonce')), {})
    payload=json.loads(capsys.readouterr().out.split('TORCHYAND_TPU_PROBE:')[1])
    assert payload['status']=='BLOCKED' and payload['nonce']=='test-nonce'


def test_bridge_rejects_mock_or_missing_journal_before_cli_import(tmp_path,monkeypatch):
    def veto(*args,**kwargs): pytest.fail('No process permitted')
    monkeypatch.setattr(subprocess,'Popen',veto);monkeypatch.setattr(subprocess,'run',veto)
    with pytest.raises(OSError): a.bridge(tmp_path,'allocate')


def test_watchdog_loss_cannot_report_success(tmp_path):
    directory = tmp_path/'dead-watch'; cli = FakeCLI(directory)
    class DeadWatch(FakeWatch):
        def wait(self, timeout): return 1
    result = a.run(permit(), record(), directory, cli_factory=lambda _:cli,
                   watch_factory=lambda d:DeadWatch(d,cli))
    assert result['outcome'] == 'WATCHDOG_FAILED'
    assert result['cleanup']['status'] == 'RELEASED'


def test_watch_parent_death_uses_owned_cleanup(tmp_path, monkeypatch):
    directory = tmp_path/'parent-death'; cli = FakeCLI(directory)
    a.run(permit(), record(), directory, cli_factory=lambda _:cli,
          watch_factory=lambda d:FakeWatch(d,cli))
    monkeypatch.setattr(a.guard, 'wait_for_trigger', lambda *args:'PARENT_DEAD')
    assert a.watch(directory, cli) == 0
    assert json.loads((directory/'release.json').read_text())['trigger'] == 'PARENT_DEAD'


def test_expired_transport_never_spawns(tmp_path):
    directory = tmp_path/'expired'; cli = FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    data=a.state(directory);data['deadline']=time.monotonic()-1;a.save(directory,data)
    def veto(*args,**kwargs): pytest.fail('expired process')
    with pytest.raises(TimeoutError): a.CLI(directory,veto).call('allocate')


def test_bridge_exact_dispatch_auth_guard_and_no_double_allocate(tmp_path, monkeypatch):
    from types import ModuleType
    directory=tmp_path/'bridge';cli=FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    data=a.state(directory);data['phase']='NAMED_ALLOCATION_PENDING';a.save(directory,data)
    real_authorize=a.authorize
    monkeypatch.setattr(a,'authorize',lambda *args:real_authorize(*args,platform='linux'))
    monkeypatch.setattr(a.importlib.metadata,'version',lambda _: '0.7.4')
    monkeypatch.setattr(a.os,'umask',lambda _:0)
    monkeypatch.setattr(a.logging,'disable',lambda _:None)
    monkeypatch.setattr(a.builtins,'input',lambda:None)
    monkeypatch.setattr(sys,'argv',[])
    package=ModuleType('colab_cli');package.__path__=[]
    auth=ModuleType('colab_cli.auth');auth._run_remote_flow=lambda:None
    common=ModuleType('colab_cli.common')
    def denied(): raise SystemExit(1)
    common.get_credentials=denied
    update=ModuleType('colab_cli.auto_update');update.run_background_check=lambda:None
    entry=ModuleType('colab_cli.cli');seen=[]
    def main():
        seen.append(sys.argv.copy())
        with pytest.raises(RuntimeError,match='AUTH_FAILED_NOT_EMPTY_SERVER'): common.get_credentials()
        with pytest.raises(RuntimeError,match='INTERACTIVE_AUTH_DISABLED'): auth._run_remote_flow()
    entry.main=main
    for mod in (package,auth,common,update,entry): monkeypatch.setitem(sys.modules,mod.__name__,mod)
    assert a.bridge(directory,'allocate')==0
    assert seen[0][-5:]==['new','-s',data['name'],'--tpu','v5e1']
    assert seen[0][1:3]==['--config',str(directory/'cli-sessions.json')]
    with pytest.raises(FileExistsError): a.bridge(directory,'allocate')
    with pytest.raises(ValueError,match='ALLOWLISTED'): a.bridge(directory,'ssh')
