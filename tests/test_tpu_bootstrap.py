"""Offline bootstrap tests: fake package transport/install, no Colab or TPU."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import pytest
from tools import tpu_bootstrap as b


def manifest():
    return json.loads(Path(b.__file__).with_name('tpu_wheels.json').read_text())


def test_reviewed_lock_and_pipless_launcher():
    data=manifest();packages=b.validate_manifest(data)
    assert sum(n['size'] for n in packages)==452635819
    text=b.requirements(packages)
    assert len(text.splitlines())==30 and 'torch-xla[tpu]==2.9.0' in text
    assert hashlib.sha256(text.encode()).hexdigest()=='fa623ac4cf1cce5d7710464beb5f68e51e18771c30cda1725794706805439078'
    # The reviewed Windows artifact used CRLF; package pins/hashes are identical.
    assert hashlib.sha256(text.replace('\n','\r\n').encode()).hexdigest()=='b94adc3e6ac48f0b6091496f6bccd1bd4de1740fa19864b8468ab0b2e3969b3f'
    code=b.launcher('print("probe")','mock-nonce',3500)
    compile(code,'remote','exec')
    assert '--without-pip' in code and '--python' in code and '--no-index' in code
    assert 'ensurepip' not in b.PREFLIGHT
    for value in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError): b.launcher('', '', value)


@pytest.mark.parametrize('url',['http://files.pythonhosted.org/a','https://evil.invalid/a',
    'https://files.pythonhosted.org.evil/a','https://u:p@files.pythonhosted.org/a',
    'https://files.pythonhosted.org:444/a'])
def test_untrusted_urls_and_redirects_rejected(url):
    with pytest.raises(ValueError): b.trusted_url(url)
    with pytest.raises(ValueError): b.Redirect().redirect_request(None,None,302,'',{},url)


def test_manifest_rejects_duplicate_and_path_escape():
    data=manifest();data['packages'][1]['filename']='../escape.whl'
    with pytest.raises(ValueError): b.validate_manifest(data)
    data=manifest();data['packages'][1]['name']=data['packages'][0]['name']
    with pytest.raises(ValueError): b.validate_manifest(data)


@pytest.mark.parametrize('failure',['none','length','truncated','redirect','hash','timeout'])
def test_download_hash_and_failure_no_retry(tmp_path,monkeypatch,failure):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:z.writestr('tiny.py','x=1')
    payload=buf.getvalue()
    node=dict(name='tiny',filename='tiny.whl',url='https://files.pythonhosted.org/tiny.whl',
              size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    calls=[]
    class Response(io.BytesIO):
        headers={'Content-Length':str(len(payload)+(1 if failure=='length' else 0))}
        def geturl(self):return 'https://evil.invalid/x' if failure=='redirect' else node['url']
    class Opener:
        def open(self,url,timeout):
            calls.append(url)
            if failure=='timeout':raise TimeoutError('private')
            return Response(payload[:-1] if failure=='truncated' else payload)
    if failure in ('length','truncated','redirect','timeout'):
        with pytest.raises((ValueError,TimeoutError)):b.download([node],tmp_path,lambda:10,Opener())
    else:
        assert b.download([node],tmp_path,lambda:10,Opener())==len(payload)
        monkeypatch.setattr(b.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**10))
        if failure=='hash':
            node['sha256']='0'*64
            with pytest.raises(ValueError,match='HASH'):b.verify_wheels([node],tmp_path,lambda:10)
        else:assert b.verify_wheels([node],tmp_path,lambda:10)==3
    assert len(calls)==1


@pytest.mark.parametrize('failure',['none','old-pip','missing-option','preflight','venv','pip_target','install','pip_check','timeout',
                                  'install-resolver','install-unknown','install-metadata','missing-python'])
def test_isolated_pipeline_with_mock_processes(tmp_path,monkeypatch,capsys,failure):
    data=manifest();calls=[]
    # Resource limits are only applied by real POSIX remote child processes.
    monkeypatch.setitem(sys.modules,'resource',SimpleNamespace(RLIMIT_FSIZE=1,setrlimit=lambda *a:None))
    monkeypatch.setattr(b.tempfile,'mkdtemp',lambda **kw:str(tmp_path))
    def downloaded(packages,directory,remaining):return b.TOTAL_BYTES
    monkeypatch.setattr(b,'download',downloaded)
    monkeypatch.setattr(b,'verify_wheels',lambda *a:1234)
    class Child:
        pid=99999
        def __init__(self,argv,**kwargs):
            calls.append(argv);self.argv=argv;self.returncode=0
            text=''
            if argv[0]=='/usr/bin/python3.12' and '-c' in argv:
                if failure=='missing-python':raise FileNotFoundError('CANARY_TOKEN /private/path')
                text=json.dumps(dict(python='3.12.3'))
                if failure=='preflight':self.returncode=1;text='AssertionError: ABI_MISMATCH'
            elif '--without-pip' in argv:
                env=Path(argv[-1]);env.mkdir();(env/'pyvenv.cfg').write_text('include-system-site-packages = false')
                if failure=='venv':self.returncode=1
            elif '--help' in argv:text='Usage' if failure=='missing-option' else '--python'
            elif '--version' in argv:
                text='pip 22.2' if failure=='old-pip' else 'pip 25.0'
                if '--python' in argv and failure=='pip_target':self.returncode=1
            elif 'install' in argv and failure.startswith('install'):
                self.returncode=23
                text={'install-resolver':'ERROR: ResolutionImpossible: conflicting dependencies',
                      'install-metadata':'ERROR: torch has inconsistent Name',
                      'install-unknown':'unrecognized issue'}.get(failure,'')
                text+='\nBearer CANARY_TOKEN /private/CANARY_PATH https://CANARY.invalid/?token=secret'
            elif 'check' in argv and failure=='pip_check':self.returncode=13
            elif '-c' in argv and 'importlib.metadata' in argv[-1]:text=json.dumps({n['name']:n['version'] for n in data['packages']})
            elif '-c' in argv:text='TORCHYAND_TPU_PROBE:'+json.dumps(dict(nonce='mock',status='PASS'))
            kwargs['stdout'].write(text.encode());kwargs['stdout'].flush()
        def wait(self,timeout=None):
            if timeout is None:return self.returncode
            assert 0<timeout<=3500
            if failure=='timeout' and len(calls)==1:
                # Timeout once; wait after the mocked kill completes.
                if not hasattr(self,'raised'):self.raised=True;self.returncode=-9;raise subprocess.TimeoutExpired('mock',timeout)
            return self.returncode
    monkeypatch.setattr(b.subprocess,'Popen',Child)
    monkeypatch.setattr(b.os,'killpg',lambda *a:None,raising=False)
    monkeypatch.setattr(b.signal,'SIGKILL',9,raising=False)
    b.run(data,'mock probe','mock',3500)
    output=capsys.readouterr().out
    report=json.loads(output.split('TORCHYAND_BOOTSTRAP:')[1])
    assert report['status']==('PASS' if failure=='none' else 'BLOCKED')
    if failure=='preflight':assert report['reason']=='ABI_MISMATCH'
    if failure.startswith('install'):
        diagnostic=report['command_failure']
        assert diagnostic['returncode']==23 and diagnostic['stage']=='install'
        expected={'install-resolver':'DEPENDENCY_CONFLICT','install-metadata':'INVALID_WHEEL'}.get(failure,'UNKNOWN')
        assert diagnostic['category']==expected and diagnostic['output_bytes']>0
    if failure=='pip_check':assert report['command_failure']['returncode']==13 and report['command_failure']['stage']=='pip_check'
    if failure=='timeout':assert report['command_failure']['returncode']==-9
    if failure=='missing-python':assert report['reason']=='PYTHON312_UNAVAILABLE' and 'command_failure' not in report
    assert 'CANARY' not in output and 'Bearer' not in output
    if failure=='none':
        installs=[a for a in calls if 'install' in a]
        assert len(installs)==1 and '--python' in installs[0] and '--no-index' in installs[0]
        assert '--require-hashes' in installs[0] and '--without-pip' in calls[3]
        assert any('check' in a and '--python' in a for a in calls)
    else:assert 'TORCHYAND_TPU_PROBE:' not in output
    assert not any('ensurepip' in x or '--upgrade' in x or '--no-deps' in x for a in calls for x in a)


@pytest.mark.parametrize('text,category',[
    ('ERROR: No matching distribution found for torch','NO_MATCHING_DISTRIBUTION'),
    ('ERROR: Package requires a different Python','REQUIRES_PYTHON'),
    ('ERROR: Wheel torch is invalid','INVALID_WHEEL'),
    ('PermissionError: Permission denied','INSTALL_PERMISSION'),
    ('Could not find the Python interpreter','PIP_TARGET_ERROR'),
    ('unrecognized failure','UNKNOWN')])
def test_bounded_error_classification(text,category):
    payload=b'Bearer CANARY_TOKEN https://CANARY.invalid /private/path\n'+b'x'*100000+text.encode()
    diagnostic=b.command_failure('install',7,io.BytesIO(payload))
    assert diagnostic==dict(stage='install',returncode=7,output_bytes=len(payload),inspected_bytes=b.SCAN_LIMIT,category=category)
    assert 'CANARY' not in json.dumps(diagnostic)


@pytest.mark.parametrize('key,value',[('returncode',True),('returncode',0),('returncode',2**31),
    ('category','CANARY_TOKEN'),('stage','CANARY_PATH'),('output_bytes',b.LOG_LIMIT+1),
    ('output_bytes',-1),('inspected_bytes',b.SCAN_LIMIT+1),('inspected_bytes',True)])
def test_host_failure_projection_rejects_invalid_fields(key,value):
    diagnostic=dict(stage='install',returncode=23,output_bytes=100000,inspected_bytes=65536,category='UNKNOWN')
    diagnostic[key]=value
    with pytest.raises(ValueError):b.validate_failure(diagnostic)


def test_actual_local_child_exit_code_is_not_adapter_exit_one():
    # No pip/install/network: a real local subprocess with synthetic pip output.
    with tempfile.TemporaryFile() as log:
        child=subprocess.run([sys.executable,'-I','-c',
            'import sys; print("ResolutionImpossible Bearer CANARY_TOKEN /private/path"); sys.exit(37)'],
            stdout=log,stderr=subprocess.STDOUT,timeout=10)
        diagnostic=b.command_failure('install',child.returncode,log)
    assert diagnostic['returncode']==37 and diagnostic['category']=='DEPENDENCY_CONFLICT'
    assert 'CANARY' not in json.dumps(diagnostic)


def test_timeout_and_untrusted_error_do_not_print_secret(monkeypatch,capsys):
    def denied(*a,**k):raise OSError('Authorization: CANARY_SECRET')
    monkeypatch.setattr(b.subprocess,'Popen',denied)
    b.run(manifest(),'','mock',60)
    output=capsys.readouterr().out
    assert 'CANARY_SECRET' not in output and 'BLOCKED' in output


def test_existing_local_pip_can_manage_empty_pipless_venv(tmp_path):
    # Real local mechanism check, using the installed host Python (not a claim
    # about CPython 3.12 or Colab). No dependencies installed and no network.
    env=tmp_path/'empty'
    subprocess.run([sys.executable,'-I','-m','venv','--without-pip',str(env)],check=True,capture_output=True,timeout=30)
    python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    query=subprocess.run([str(python),'-I','-c','import importlib.util; assert importlib.util.find_spec("pip") is None'],check=True,capture_output=True,timeout=10)
    result=subprocess.run([sys.executable,'-I','-m','pip','--isolated','--disable-pip-version-check','--python',str(python),'check'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'No broken requirements found' in result.stdout
