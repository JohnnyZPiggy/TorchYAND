"""Stdlib remote bootstrap for a pip-less CPython 3.12 venv; no action on import.

Only launcher() is used on the host. run() runs inside an explicitly authorized
TPU session through the existing CLI; it never allocates devices or sessions.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

TOTAL_BYTES = 452635819
HOSTS = {'files.pythonhosted.org', 'download-r2.pytorch.org'}
STAGES = ('preflight', 'venv', 'pip_target', 'download', 'hash', 'install', 'pip_check', 'probe', 'complete')
PREFLIGHT_ERRORS = ('PYTHON_NOT_312', 'ABI_MISMATCH', 'FREE_THREADED_UNSUPPORTED', 'PLATFORM_MISMATCH',
                    'VENV_MISSING', 'SPACE_INSUFFICIENT')
REASONS = PREFLIGHT_ERRORS+('PYTHON312_UNAVAILABLE', 'PIP_22_3_REQUIRED', 'PIP_PYTHON_OPTION_MISSING',
    'COMMAND_FAILED', 'VENV_NOT_ISOLATED', 'BOOTSTRAP_DEADLINE', 'UNTRUSTED_WHEEL_URL',
    'WHEEL_LENGTH_MISMATCH', 'WHEEL_TRUNCATED', 'WHEEL_BYTE_CAP', 'WHEEL_HASH_MISMATCH',
    'EXPANDED_SPACE_INSUFFICIENT', 'INSTALLED_VERSION_MISMATCH')


def trusted_url(url):
    u = urllib.parse.urlparse(url)
    if u.scheme != 'https' or u.hostname not in HOSTS or u.username or u.password or u.port not in (None, 443):
        raise ValueError('UNTRUSTED_WHEEL_URL')


def validate_manifest(manifest):
    packages = manifest['packages']
    if len(packages) != 30 or sum(n['size'] for n in packages) != TOTAL_BYTES:
        raise ValueError('FROZEN_MANIFEST_MISMATCH')
    if len({n['name'] for n in packages}) != 30 or len({n['filename'] for n in packages}) != 30:
        raise ValueError('DUPLICATE_WHEEL')
    for n in packages:
        trusted_url(n['url'])
        if (type(n['size']) is not int or n['size'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', n['sha256'])
                or not re.fullmatch(r'[A-Za-z0-9_.+%-]+\.whl', n['filename'])
                or urllib.parse.unquote(n['url'].rsplit('/', 1)[-1]) != n['filename']
                or not re.fullmatch(r'[a-z0-9-]+', n['name'])
                or not re.fullmatch(r'[0-9][A-Za-z0-9_.+]*', n['version'])):
            raise ValueError('INVALID_WHEEL_RECORD')
    roots = {n['name']: n['version'] for n in packages}
    if any(roots.get(k) != v for k, v in {'torch':'2.9.0+cpu', 'torch-xla':'2.9.0', 'libtpu':'0.0.21', 'tpu-info':'0.14.2'}.items()):
        raise ValueError('ROOT_VERSION_MISMATCH')
    return packages


def requirements(packages):
    return '\n'.join(n['name']+('[tpu]' if n['name']=='torch-xla' else '')+'=='+n['version']+
                     ' --hash=sha256:'+n['sha256'] for n in sorted(packages, key=lambda n:n['name']))+'\n'


def launcher(probe, nonce, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('FINITE_BOOTSTRAP_BUDGET_REQUIRED')
    manifest = json.loads(Path(__file__).with_name('tpu_wheels.json').read_text())
    validate_manifest(manifest)
    # A private exec namespace prevents the notebook kernel from importing torch.
    return ('scope={"__name__":"torchyand_remote"}\nexec('+repr(Path(__file__).read_text())+',scope)\n'
            'scope["run"]('+repr(manifest)+','+repr(probe)+','+repr(nonce)+','+repr(seconds)+')\n')


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        trusted_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(packages, directory, remaining, opener=None):
    """Fresh wheel files only, exact lengths and trusted redirects, no retries."""
    opener = urllib.request.build_opener(Redirect()) if opener is None else opener
    total = 0
    for n in packages:
        trusted_url(n['url'])
        target = directory/n['filename']
        with opener.open(n['url'], timeout=min(30, remaining())) as response, target.open('xb') as out:
            trusted_url(response.geturl())
            if int(response.headers.get('Content-Length', -1)) != n['size']:
                raise ValueError('WHEEL_LENGTH_MISMATCH')
            count = 0
            while count < n['size']:
                remaining()
                chunk = response.read(min(1024*1024, n['size']-count))
                if not chunk:
                    raise ValueError('WHEEL_TRUNCATED')
                out.write(chunk); count += len(chunk); total += len(chunk)
                if total > TOTAL_BYTES:
                    raise ValueError('WHEEL_BYTE_CAP')
    return total


def verify_wheels(packages, directory, remaining):
    unpacked = 0
    for n in packages:
        h = hashlib.sha256()
        target = directory/n['filename']
        if target.stat().st_size != n['size']:
            raise ValueError('WHEEL_LENGTH_MISMATCH')
        with target.open('rb') as source:
            while chunk := source.read(1024*1024):
                remaining(); h.update(chunk)
        if h.hexdigest() != n['sha256']:
            raise ValueError('WHEEL_HASH_MISMATCH')
        with zipfile.ZipFile(target) as wheel:
            unpacked += sum(n.file_size for n in wheel.infolist())
    if shutil.disk_usage(directory).free < unpacked+2*1024**3:
        raise ValueError('EXPANDED_SPACE_INSUFFICIENT')
    return unpacked


PREFLIGHT = '''import sys,sysconfig,platform,shutil,importlib.util,json
assert sys.version_info[:2]==(3,12), 'PYTHON_NOT_312'
assert sysconfig.get_config_var('SOABI')=='cpython-312-x86_64-linux-gnu', 'ABI_MISMATCH'
assert not sysconfig.get_config_var('Py_GIL_DISABLED'), 'FREE_THREADED_UNSUPPORTED'
libc=platform.libc_ver()
assert platform.machine()=='x86_64' and libc[0]=='glibc' and tuple(map(int,libc[1].split('.'))) >=(2,31), 'PLATFORM_MISMATCH'
assert importlib.util.find_spec('venv'), 'VENV_MISSING'
assert shutil.disk_usage('/tmp').free>=8*1024**3, 'SPACE_INSUFFICIENT'
print(json.dumps(dict(python=sys.version.split()[0],soabi=sysconfig.get_config_var('SOABI'),glibc=libc[1])))
'''


def run(manifest, probe, nonce, seconds):
    started = time.monotonic()
    report = dict(nonce=nonce, status='BLOCKED', stage='preflight', timings={})
    root = None
    def remaining():
        value = started+seconds-time.monotonic()
        if not math.isfinite(value) or value < 1:
            raise TimeoutError('BOOTSTRAP_DEADLINE')
        return value
    def command(argv):
        # Bound private stdout+stderr on disk; no raw package/HTTP output escapes.
        import resource
        def cap(): resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
        with tempfile.TemporaryFile(dir=root) as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=cap)
            try:
                code = child.wait(timeout=remaining())
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
            if code:
                log.seek(0)
                failure = log.read(65536).decode('utf-8', errors='replace')
                for reason in PREFLIGHT_ERRORS:
                    if failure.rstrip().endswith('AssertionError: '+reason):
                        raise ValueError(reason)
                raise RuntimeError('COMMAND_FAILED')
            log.seek(0)
            return log.read(65536).decode('utf-8', errors='replace')
    def stage(name, function, *args):
        report['stage'] = name
        before = time.monotonic()
        try:
            return function(*args)
        finally:
            report['timings'][name] = round(time.monotonic()-before, 3)
    try:
        os.umask(0o077)
        packages = validate_manifest(manifest)
        info = stage('preflight', command, ['/usr/bin/python3.12', '-I', '-c', PREFLIGHT])
        report['python'] = json.loads(info)['python']
        manager = [sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check']
        version = command(manager+['--version'])
        match = re.match(r'pip (\d+)\.(\d+)', version)
        if not match or tuple(map(int, match.groups())) < (22, 3):
            raise ValueError('PIP_22_3_REQUIRED')
        # Check documented support before any package download, without upgrades.
        if '--python' not in command(manager+['--help']):
            raise ValueError('PIP_PYTHON_OPTION_MISSING')
        root = Path(tempfile.mkdtemp(prefix='torchyand-isolated-'))
        env = root/'env'
        stage('venv', command, ['/usr/bin/python3.12', '-I', '-m', 'venv', '--without-pip', str(env)])
        if 'include-system-site-packages = false' not in (env/'pyvenv.cfg').read_text().lower():
            raise ValueError('VENV_NOT_ISOLATED')
        target = manager+['--python', str(env/'bin/python')]
        stage('pip_target', command, target+['--version'])
        wheels = root/'wheels'; wheels.mkdir(mode=0o700)
        report['bytes'] = stage('download', download, packages, wheels, remaining)
        report['unpacked_bytes'] = stage('hash', verify_wheels, packages, wheels, remaining)
        report['verified_wheels'] = len(packages)
        lock = root/'requirements.txt'; lock.write_text(requirements(packages))
        stage('install', command, target+['install', '--no-index', '--find-links', str(wheels), '--require-hashes',
              '--only-binary=:all:', '--no-cache-dir', '--no-compile', '-r', str(lock)])
        stage('pip_check', command, target+['check'])
        versions = json.loads(command([str(env/'bin/python'), '-I', '-c',
            'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in '+
            repr([n['name'] for n in packages])+'}))']))
        if any(versions[n['name']] != n['version'] for n in packages):
            raise ValueError('INSTALLED_VERSION_MISMATCH')
        # A new isolated process alone imports torch/XLA and initializes PJRT.
        output = stage('probe', command, [str(env/'bin/python'), '-I', '-c', probe])
        for line in output.splitlines():
            if line.startswith('TORCHYAND_TPU_PROBE:'):
                print(line)
        report.update(status='PASS', stage='complete')
    except Exception as error:
        report['error_class'] = type(error).__name__ if type(error).__name__ in ('ValueError','RuntimeError','TimeoutExpired','TimeoutError','FileNotFoundError','AssertionError','OSError','HTTPError','URLError') else 'Exception'
        # Only local constant diagnostics; never provider/pip exception text.
        safe = str(error)
        if safe in REASONS:
            report['reason'] = safe
        if isinstance(error, FileNotFoundError) and report['stage']=='preflight':
            report['reason'] = 'PYTHON312_UNAVAILABLE'
    finally:
        report['seconds'] = round(time.monotonic()-started, 3)
        print('TORCHYAND_BOOTSTRAP:'+json.dumps(report))
