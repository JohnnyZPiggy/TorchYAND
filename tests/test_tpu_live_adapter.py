"""CPU/MOCK ONLY. No installed Colab entrypoint, account or TPU is invoked."""
from datetime import datetime, timedelta, timezone
from copy import deepcopy
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


def github_records():
    now = datetime.now(timezone.utc)
    created = (now-timedelta(minutes=5)).isoformat()
    expiry = (now+timedelta(minutes=30)).isoformat()
    items = {a.API+'/git/ref/heads/main': dict(ref='refs/heads/main',object=dict(type='commit',sha=SHA))}
    for identifier, reference, kind in [(1,APPROVAL,'TPU_RESOURCE_APPROVAL'),(2,INSTRUCTION,'TPU_SMOKE_EXECUTION')]:
        body = {k:v for k,v in record().items() if k!='approval_ref'}
        body.update(kind=kind,expires_at=expiry)
        if identifier==2: body['approval_ref']=APPROVAL
        items[a.API+'/issues/comments/'+str(identifier)] = dict(id=identifier,html_url=reference,
            issue_url=a.API+'/issues/7',user=dict(login='JohnnyZPiggy',id=a.SCIENTIST_ID,type='User'),
            body=json.dumps(body),created_at=created,updated_at=created)
    return items


@pytest.fixture(autouse=True)
def offline_github(monkeypatch, tmp_path):
    # Every positive record is served by an explicit offline fake, never live GitHub.
    monkeypatch.setattr(a, 'EVIDENCE_ROOT', tmp_path/'durable')
    items=github_records()
    monkeypatch.setattr(a,'github_json',lambda url:deepcopy(items[url]))
    return items


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
        self.evidence = a.state(directory).get('evidence')
        (directory/'watch-ready').touch()
    def poll(self): return None
    def wait(self, timeout): a.release(self.directory, self.cli, 'MOCK_FINALLY', self.evidence); return 0
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
    if not argv:
        assert a.main(argv)==0
        assert not a.EVIDENCE_ROOT.exists()
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
    real_validate=a.validate_record
    monkeypatch.setattr(a,'validate_record',lambda *args:real_validate(*args[:6],platform='linux'))
    real_authorize=a.authorize
    monkeypatch.setattr(a,'authorize',lambda *args,**kwargs:real_authorize(*args,platform='linux',**kwargs))
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


def test_dead_watchdog_blocks_allocation(tmp_path):
    directory=tmp_path/'dead-before';cli=FakeCLI(directory)
    class DeadBefore(FakeWatch):
        def poll(self): return 1
    result=a.run(permit(),record(),directory,cli_factory=lambda _:cli,
                 watch_factory=lambda d:DeadBefore(d,cli))
    assert result['outcome']=='BLOCKED'
    assert 'allocate' not in cli.calls and 'probe' not in cli.calls and 'stop' not in cli.calls


@pytest.mark.parametrize('defect',[
    'nonexistent','wrong-id','wrong-url','wrong-issue','wrong-author','wrong-user-id','bot',
    'stale','expired','future','unbounded','edited-after-execution','wrong-sha','wrong-device',
    'wrong-count','wrong-time','wrong-quota','missing-risk','negated','wrong-kind','wrong-approval',
    'remote-ahead','remote-unavailable','remote-not-commit'])
@pytest.mark.parametrize('which',[1,2])
def test_live_gate_negative_zero_colab(tmp_path,monkeypatch,offline_github,defect,which):
    items=offline_github
    key=a.API+'/issues/comments/'+str(which);comment=items[key]
    body=json.loads(comment['body'])
    now=datetime.now(timezone.utc)
    if defect=='nonexistent': del items[key]
    elif defect=='wrong-id': comment['id']=999
    elif defect=='wrong-url': comment['html_url']=APPROVAL+'0'
    elif defect=='wrong-issue': comment['issue_url']=a.API+'/issues/12'
    elif defect=='wrong-author': comment['user']['login']='attacker'
    elif defect=='wrong-user-id': comment['user']['id']=1
    elif defect=='bot': comment['user']['type']='Bot'
    elif defect=='stale': comment['created_at']=(now-timedelta(days=2)).isoformat()
    elif defect=='expired': body['expires_at']=(now-timedelta(seconds=1)).isoformat()
    elif defect=='future': comment['created_at']=(now+timedelta(seconds=1)).isoformat()
    elif defect=='unbounded': body['expires_at']=(now+timedelta(days=2)).isoformat()
    elif defect=='edited-after-execution':
        items[a.API+'/issues/comments/1']['updated_at']=(now-timedelta(seconds=1)).isoformat()
    elif defect=='wrong-sha': body['source_sha']='b'*40
    elif defect=='wrong-device': body['device']='v6e1'
    elif defect=='wrong-count': body['max_sessions']=True
    elif defect=='wrong-time': body['max_minutes']=6
    elif defect=='wrong-quota': body['quota_policy']='unlimited'
    elif defect=='missing-risk': del body['residual_outage_risk_ack']
    elif defect=='negated': body['decision']='NOT APPROVED'
    elif defect=='wrong-kind': body['kind']='ARCHITECT_REVIEW'
    elif defect=='wrong-approval': body['approval_ref']=INSTRUCTION
    elif defect=='remote-ahead': items[a.API+'/git/ref/heads/main']['object']['sha']='b'*40
    elif defect=='remote-unavailable': del items[a.API+'/git/ref/heads/main']
    elif defect=='remote-not-commit': items[a.API+'/git/ref/heads/main']['object']['type']='tag'
    comment['body']=json.dumps(body)
    approval=tmp_path/'approval.json';approval.write_text(json.dumps(record()))
    seen=[]
    def git(argv,**kwargs):
        seen.append(argv)
        if argv[1:]==['status','--porcelain']: return ''
        return SHA  # Stale/manipulated origin/main still agrees with local HEAD.
    monkeypatch.setattr(a.subprocess,'check_output',git)
    def veto(*args,**kwargs): pytest.fail('NO COLAB PROCESS OR RUN PERMITTED')
    monkeypatch.setattr(a.subprocess,'run',veto);monkeypatch.setattr(a.subprocess,'Popen',veto)
    monkeypatch.setattr(a,'run',veto)
    validate=a.validate_record
    monkeypatch.setattr(a,'validate_record',lambda *args:validate(*args[:6],platform='linux'))
    with pytest.raises((ValueError,KeyError)):
        a.main(['--live','--approval-file',str(approval),'--instruction-ref',INSTRUCTION,
                '--cli-python',str(Path(sys.executable).resolve()),'--output-dir',str(tmp_path/'never')])
    assert not (tmp_path/'never').exists()


def test_local_approved_json_alone_fails(monkeypatch):
    def missing(url): raise OSError('offline / 404')
    monkeypatch.setattr(a,'github_json',missing)
    with pytest.raises(OSError): permit()


def test_github_reader_bounded_fixed_origin(monkeypatch):
    # Exercise the actual stdlib transport against fake response objects only.
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def geturl(self): return target
        def read(self,count): assert count==65537;return b'{}'
    target=a.API+'/git/ref/heads/main'
    def opening(request,timeout):
        assert request.full_url==target and timeout==10
        assert not request.has_header('Authorization')
        return Response()
    monkeypatch.setattr(a.urllib.request,'urlopen',opening)
    # autouse fixture replaced the helper; call saved original below.
    assert REAL_GITHUB_JSON(target)=={}
    with pytest.raises(ValueError): REAL_GITHUB_JSON('https://evil.example/main')


REAL_GITHUB_JSON = __import__('tools.tpu_live_adapter',fromlist=['github_json']).github_json


def test_stale_tracking_ref_rejected_even_if_live_main_matches(tmp_path,monkeypatch):
    approval=tmp_path/'approval.json';approval.write_text(json.dumps(record()))
    def git(argv,**kwargs):
        if argv[-1]=='origin/main': return 'b'*40
        if argv[1]=='status': return ''
        return SHA
    monkeypatch.setattr(a.subprocess,'check_output',git)
    def veto(*args,**kwargs): pytest.fail('No Colab')
    monkeypatch.setattr(a.subprocess,'Popen',veto);monkeypatch.setattr(a,'run',veto)
    validate=a.validate_record
    monkeypatch.setattr(a,'validate_record',lambda *args:validate(*args[:6],platform='linux'))
    with pytest.raises(ValueError):
        a.main(['--live','--approval-file',str(approval),'--instruction-ref',INSTRUCTION,
                '--cli-python',str(Path(sys.executable).resolve()),'--output-dir',str(tmp_path/'never')])


def test_direct_bridge_allocation_rechecks_github_before_cli(tmp_path,monkeypatch,offline_github):
    directory=tmp_path/'child-auth';cli=FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    offline_github[a.API+'/issues/comments/2']['user']['id']=1
    validate=a.validate_record
    monkeypatch.setattr(a,'validate_record',lambda *args:validate(*args[:6],platform='linux'))
    monkeypatch.setattr(a.importlib.metadata,'version',lambda *args:pytest.fail('No CLI import'))
    with pytest.raises(ValueError,match='IDENTITY'): a.bridge(directory,'allocate')
    assert not (directory/'allocation-attempted').exists()


@pytest.mark.parametrize('failure',['timeout','redirect','oversize','invalid-json'])
def test_github_transport_failures(monkeypatch,failure):
    target=a.API+'/git/ref/heads/main'
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def geturl(self): return 'https://evil.example' if failure=='redirect' else target
        def read(self,n): return b'x'*65537 if failure=='oversize' else b'not-json'
    def opening(*args,**kwargs):
        if failure=='timeout': raise TimeoutError('mock timeout')
        return Response()
    monkeypatch.setattr(a.urllib.request,'urlopen',opening)
    with pytest.raises((ValueError,TimeoutError)): REAL_GITHUB_JSON(target)


def test_expiry_during_github_reads_is_rejected(monkeypatch,offline_github):
    start=datetime.now(timezone.utc)
    class Clock:
        calls=0
        @classmethod
        def now(cls,tz):
            cls.calls+=1
            return start if cls.calls==1 else start+timedelta(hours=1)
        fromisoformat=staticmethod(datetime.fromisoformat)
    monkeypatch.setattr(a,'datetime',Clock)
    with pytest.raises(ValueError,match='EXPIRED_DURING_VERIFICATION'): permit()


def evidence_for(directory):
    return Path(a.state(directory)['evidence'])


def timeline(evidence):
    return [json.loads(row) for row in (evidence/'timeline.jsonl').read_text().splitlines()]


def test_durable_evidence_survives_temp_loss_and_excludes_secrets(tmp_path):
    import shutil, hashlib
    directory=tmp_path/'disposable';cli=FakeCLI(directory)
    result=a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    evidence=evidence_for(directory)
    secret=a.state(directory)['name']
    shutil.rmtree(directory)
    saved=json.loads((evidence/'summary.json').read_text())
    assert saved['outcome']=='TPU_PROBE_PASS' and saved['TPU_PJRT']=='NOT VERIFIED' and saved['execution_mode']=='CPU_MOCK'
    assert result['run_id']==evidence.name
    content=''.join(p.read_text() for p in evidence.iterdir())
    for canary in (secret,'owned-endpoint','unrelated','mock://','approval_ref','cli_python'):
        assert canary not in content
    manifest=json.loads((evidence/'manifest.json').read_text())
    for name,entry in manifest.items():
        assert hashlib.sha256((evidence/name).read_bytes()[:entry['bytes']]).hexdigest()==entry['sha256']
    rows=timeline(evidence)
    assert any(r['event']=='backend_release_verified' for r in rows)
    assert all(set(('event','utc_ts','elapsed_monotonic_ms','stage','outcome','error_class',
                   'deadline_remaining_ms','origin_sha','run_id'))<=r.keys() for r in rows)
    if os.name=='posix':
        assert evidence.stat().st_mode & 0o777==0o700
        assert all(p.stat().st_mode & 0o777==0o600 for p in evidence.iterdir())


@pytest.mark.parametrize('failure',['timeout','interrupt'])
def test_partial_output_metadata_never_persists_canaries(tmp_path,failure):
    directory=tmp_path/'partial';cli=FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    evidence=evidence_for(directory)
    secret=b'Authorization: Bearer CANARY_TOKEN https://proxy.invalid/?token=CANARY'
    def runner(argv,**kwargs):
        assert 30 < kwargs['timeout'] <= 180
        assert kwargs['shell'] is False
        if failure=='interrupt': raise KeyboardInterrupt('CANARY_TOKEN')
        raise subprocess.TimeoutExpired(argv,kwargs['timeout'],output=secret,stderr=secret)
    with pytest.raises((subprocess.TimeoutExpired,KeyboardInterrupt)): a.CLI(directory,runner).call('probe')
    row=timeline(evidence)[-1]
    assert row['event']=='subprocess_interrupted' and row['stage']=='probe'
    if failure=='timeout': assert row['stdout_bytes']==row['stderr_bytes']==len(secret)
    assert 'CANARY' not in ''.join(p.read_text() for p in evidence.iterdir())


def test_lost_runtime_journal_reports_manual_attention(tmp_path):
    import shutil
    directory=tmp_path/'lost';cli=FakeCLI(directory)
    original=cli.call
    def call(op):
        if op=='probe':
            shutil.rmtree(directory)
            raise subprocess.TimeoutExpired('MOCK',1)
        return original(op)
    cli.call=call
    result=a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    assert result['cleanup']['status']=='CLEANUP_FAILED'
    assert result['cleanup']['attention']=='MANUAL_ATTENTION_REQUIRED'
    assert not directory.exists()
    saved=json.loads((a.EVIDENCE_ROOT/result['run_id']/'summary.json').read_text())
    assert saved['TPU_PJRT']=='NOT VERIFIED'
    assert not any(r['event']=='backend_release_verified' for r in timeline(a.EVIDENCE_ROOT/result['run_id']))


@pytest.mark.parametrize('slow_stage,delay,expected',[
    ('github',10,'PASS'),('startup',40,'PASS'),('import',40,'PASS'),
    ('lookup',40,'PASS'),('remote',121,'TIMEOUT'),('remaining',5,'TIMEOUT')])
def test_injected_stage_delays_and_remote_budget(tmp_path,monkeypatch,offline_github,slow_stage,delay,expected):
    import types, contextlib, io, builtins
    # Reuse the fully fake official CLI fixture: no installed Colab or network.
    test_bridge_exact_dispatch_auth_guard_and_no_double_allocate(tmp_path,monkeypatch)
    directory=tmp_path/'bridge';evidence=evidence_for(directory)
    clock=SimpleNamespace(value=time.monotonic())
    monkeypatch.setattr(a.time,'monotonic',lambda:clock.value)
    read=a.github_json
    def github(url):
        if slow_stage=='github': clock.value+=delay
        return read(url)
    monkeypatch.setattr(a,'github_json',github)
    common=sys.modules['colab_cli.common']
    def lookup(*args,**kwargs):
        if slow_stage=='lookup': clock.value+=delay
        return SimpleNamespace()
    common.state=SimpleNamespace(get_session=lookup)
    runtime=types.ModuleType('colab_cli.runtime');executions=[]
    class Runtime:
        def execute_code(self,code,**kwargs):
            executions.append(kwargs['timeout'])
            cost=delay if slow_stage=='remote' else 2
            clock.value+=min(cost,kwargs['timeout'])
            if cost>kwargs['timeout']: raise TimeoutError('CANARY_REMOTE_ERROR')
    runtime.ColabRuntime=Runtime
    monkeypatch.setitem(sys.modules,'colab_cli.runtime',runtime)
    def main():
        common.state.get_session('CANARY_SESSION')
        Runtime().execute_code('TORCHYAND_TPU_PROBE:')
    sys.modules['colab_cli.cli'].main=main
    importing=builtins.__import__;seen_import=False
    def imported(name,*args,**kwargs):
        nonlocal seen_import
        if name=='colab_cli.auth' and slow_stage=='import' and not seen_import:
            seen_import=True;clock.value+=delay
        return importing(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',imported)
    if slow_stage=='remaining':
        data=a.state(directory);data['deadline']=clock.value+data['cleanup_reserve']+delay;a.save(directory,data)
    started=clock.value
    def runner(argv,**kwargs):
        if slow_stage=='startup': clock.value+=delay
        with contextlib.redirect_stdout(io.StringIO()): a.bridge(directory,'probe',float(argv[-1]))
        return subprocess.CompletedProcess(argv,0,'MOCK_RESULT','')
    if expected=='PASS':
        assert a.CLI(directory,runner).call('probe')=='MOCK_RESULT'
        assert clock.value-started>30  # Each case would exceed the old host cutoff.
        assert executions and 0<executions[0]<=120
    else:
        with pytest.raises(TimeoutError): a.CLI(directory,runner).call('probe')
        if slow_stage=='remaining': assert executions==[]
    rows=timeline(evidence)
    assert any(r['stage']=='github_approval' for r in rows)
    assert not any('CANARY' in json.dumps(r) for r in rows)


def test_parent_death_persists_terminal_evidence(tmp_path,monkeypatch):
    directory=tmp_path/'worker-death';cli=FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    evidence=evidence_for(directory)
    monkeypatch.setattr(a.guard,'wait_for_trigger',lambda *args:'PARENT_TERMINATED')
    assert a.watch(directory,cli)==0
    assert json.loads((evidence/'summary.json').read_text())['outcome']=='WORKER_LOST'


def test_no_probe_budget_leaves_cleanup_reserve(tmp_path):
    directory=tmp_path/'budget';cli=FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    data=a.state(directory);data['deadline']=time.monotonic()+data['cleanup_reserve']-.1;a.save(directory,data)
    def veto(*args,**kwargs): pytest.fail('No process when reserve reached')
    with pytest.raises(TimeoutError): a.CLI(directory,veto).call('probe')


def test_lost_diagnostics_cannot_prevent_owned_cleanup(tmp_path):
    import shutil
    directory=tmp_path/'diagnostics-loss';cli=FakeCLI(directory);original=cli.call
    def call(op):
        if op=='probe':
            shutil.rmtree(evidence_for(directory))
            raise subprocess.TimeoutExpired('MOCK',1)
        return original(op)
    cli.call=call
    result=a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    assert result['cleanup']['status']=='RELEASED'
    assert result['evidence_status']=='WRITE_FAILED' and cli.calls.count('stop')==1


def test_result_parse_stage_has_its_own_elapsed_time(tmp_path,monkeypatch):
    clock=SimpleNamespace(value=time.monotonic())
    monkeypatch.setattr(a.time,'monotonic',lambda:clock.value)
    loads=a.json.loads
    def parse(value,*args,**kwargs):
        if isinstance(value,str) and value.startswith('{"nonce"'): clock.value+=17
        return loads(value,*args,**kwargs)
    monkeypatch.setattr(a.json,'loads',parse)
    directory=tmp_path/'parse';cli=FakeCLI(directory)
    a.run(permit(),record(),directory,cli_factory=lambda _:cli,watch_factory=lambda d:FakeWatch(d,cli))
    rows=[r for r in timeline(evidence_for(directory)) if r['stage']=='result_parse']
    assert rows[-1]['elapsed_monotonic_ms']-rows[-2]['elapsed_monotonic_ms']==17000


def test_nonfinite_probe_deadline_cannot_bypass_budget(tmp_path,monkeypatch):
    test_bridge_exact_dispatch_auth_guard_and_no_double_allocate(tmp_path,monkeypatch)
    for deadline in (None,float('inf'),float('nan')):
        with pytest.raises(ValueError,match='FINITE_PROBE_HOST_DEADLINE'):
            a.bridge(tmp_path/'bridge','probe',deadline)


def test_preflight_timeout_retains_terminal_blocked_record(tmp_path,monkeypatch):
    permitted=permit()
    def timeout(*args,**kwargs): raise TimeoutError('CANARY_AUTH')
    monkeypatch.setattr(a,'authorize',timeout)
    with pytest.raises(TimeoutError): a.run(permitted,record(),tmp_path/'never')
    evidence=next(a.EVIDENCE_ROOT.iterdir())
    saved=json.loads((evidence/'summary.json').read_text())
    assert saved['outcome']=='BLOCKED' and saved['cleanup']['status']=='NOT_NEEDED'
    assert not (tmp_path/'never').exists()
    assert 'CANARY' not in ''.join(p.read_text() for p in evidence.iterdir())
