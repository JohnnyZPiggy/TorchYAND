"""CPU numerical evidence; all injected backend tests are MOCK_ONLY."""
import importlib.util
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1]/'benchmarks/bench_tpu_xla.py'
spec = importlib.util.spec_from_file_location('tpu_harness', PATH)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def test_shapes_seed_and_identity():
    a = h.make_inputs()
    assert a['A'].shape == (256, 32) and a['x'].shape == a['b'].shape == (4, 32)
    assert a['c'].shape == (4,)
    assert h.input_id(a) == h.input_id(h.make_inputs())
    assert h.input_id(a) != h.input_id(h.make_inputs(seed=124))
    assert (a['x'][~a['mask']] == 0).all()
    torch.testing.assert_close(a['x'].sum(-1), torch.ones(4, dtype=torch.float64))
    assert a['regular'].all()


@pytest.mark.parametrize('scale', [1., .01])
def test_independent_cpu_derivatives_operator_and_pcg(scale):
    d = h.make_inputs(T=64, n=8, scale=scale)
    checks = h.cpu_case(d)
    assert set(checks) == set(h.specs(d))
    assert all(v['status'] == 'PASS' for v in checks.values())
    refs, M, P = h.independent_references(d)
    result = h.fixed_pcg(*h.specs(d)['fixed_pcg'][1])
    assert result[3][0] and result[4][0] == 0 and not result[5].any()
    assert (result[4] <= 8).all()
    # Independent direct least-squares solution on the projected subspace.
    dense = torch.linalg.pinv(M) @ d['b'].unsqueeze(-1)
    torch.testing.assert_close(result[0], dense.squeeze(-1), rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(P @ P, P, rtol=2e-8, atol=2e-9)


def test_budget_exhaustion_is_not_convergence():
    d = h.make_inputs()
    out = h.fixed_pcg(*h.specs(d)['fixed_pcg'][1])
    assert out[7][1:].any()
    assert (out[4][out[7]] == 8).all()
    assert not out[3][out[7]].any()


@pytest.mark.parametrize('control', ['empty', 'singleton', 'irregular', 'singular', 'negative', 'nonfinite'])
def test_guard_controls(control):
    d = h.make_inputs(T=32, n=8)
    if control in ('empty', 'singleton'):
        d['mask'][:] = False
        if control == 'singleton':
            d['mask'][:, 0] = True
        d['nu'].zero_(); d['regular'][:] = False
        d['count'] = d['mask'].sum(-1)
        d['b'].zero_()
    if control == 'irregular':
        d['nu'].zero_(); d['regular'][:] = False
        d['b'] = h.geometry.face_tangent(d['u'], d['mask'])
    if control in ('singular', 'negative'):
        d['cp'].zero_()
        d['damping'][:] = 0 if control == 'singular' else -1
    if control == 'nonfinite':
        d['b'][1, 0] = float('nan')
    out = h.fixed_pcg(*h.specs(d)['fixed_pcg'][1])
    if control in ('empty', 'singleton'):
        assert out[3].all() and (out[4] == 0).all()
    elif control in ('negative', 'singular'):
        assert out[5][1:].all() and out[6][1:].all() and not out[3][1:].any()
    elif control == 'nonfinite':
        assert out[5][1] and not out[3][1]
    else:
        assert not out[5].any()
    if control != 'nonfinite':
        _, args = h.specs(d)['projected_hessian']
        residual = (d['b']-h.projected_hessian(out[0], *args[1:])).norm(dim=-1)
        torch.testing.assert_close(out[1], residual)


@pytest.mark.parametrize('shape', [(1, 8, 4), (32, 3, 4), (32, 8, 1)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        h.make_inputs(T=shape[0], n=shape[1], B=shape[2])


@pytest.mark.parametrize('options', [
    ['--device', 'tpu', '--mode', 'reference', '--precision', 'default'],
    ['--device', 'cpu', '--mode', 'xla-compile', '--precision', 'default'],
    ['--device', 'cpu', '--mode', 'reference', '--precision', 'high'],
    ['--device', 'cpu', '--mode', 'reference', '--precision', 'invalid'],
    ['--device', 'cpu', '--mode', 'reference', '--precision', 'default', '--warmup', '0'],
    ['--device', 'cpu', '--mode', 'reference', '--precision', 'default', '--repeats', '1'],
])
def test_cli_rejects_before_backend_import(options, monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail('backend must not be imported')
    monkeypatch.setattr(h.importlib, 'import_module', forbidden)
    with pytest.raises(SystemExit) as error:
        h.main(options+['--output-dir', str(tmp_path/'run')])
    assert error.value.code == 2
    assert not (tmp_path/'run').exists()


def test_mock_only_missing_and_unsupported_versions(monkeypatch):
    with pytest.raises(RuntimeError, match='PJRT'):
        h.validate_tpu_config(('2.8.0', '2.8.0'), {})
    with pytest.raises(RuntimeError, match='candidate'):
        h.validate_tpu_config(('2.14.1', '2.8.0'), {'PJRT_DEVICE': 'TPU'})
    monkeypatch.setattr(h.importlib.metadata, 'version', lambda name: (_ for _ in ()).throw(
        h.importlib.metadata.PackageNotFoundError(name)))
    with pytest.raises(h.importlib.metadata.PackageNotFoundError):
        h.load_tpu('default')


def test_mock_only_cpu_xla_identity_rejected():
    with pytest.raises(RuntimeError, match='identity'):
        h.validate_tpu_identity('CPU', SimpleNamespace(type='xla'), ['CPU:0'], [{'device_kind': 'CPU'}])
    with pytest.raises(RuntimeError, match='identity'):
        h.validate_tpu_identity('TPU', SimpleNamespace(type='xla'), ['TPU:0'], [])


def test_cpu_report_no_xla_import(monkeypatch, tmp_path):
    builtin_import = builtins.__import__
    def guarded_builtin(name, *args, **kwargs):
        if name.startswith('torch_xla'):
            pytest.fail('CPU path imported XLA')
        return builtin_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_builtin)
    real_import = h.importlib.import_module
    def guarded_import(name, *args, **kwargs):
        if name.startswith('torch_xla'):
            pytest.fail('CPU path imported XLA')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(h.importlib, 'import_module', guarded_import)
    out = tmp_path/'run'
    assert h.main(['--device', 'cpu', '--mode', 'reference', '--precision', 'default',
                   '--output-dir', str(out)]) == 0
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['device'] == 'cpu' and manifest['status'] == 'PASS'
    assert manifest['TPU_BENCHMARK'] == manifest['XLA_COMPILE'] == 'NOT RUN'
    assert not (out/'timings.json').exists()
    assert len(json.loads((out/'correctness.json').read_text())) == 1
    with pytest.raises(FileExistsError):
        h.main(['--device', 'cpu', '--mode', 'reference', '--precision', 'default', '--output-dir', str(out)])


def test_failure_report_nonzero_no_fake_timings(monkeypatch, tmp_path):
    def fail(_):
        raise AssertionError('injected CPU correctness failure')
    monkeypatch.setattr(h, 'cpu_case', fail)
    out = tmp_path/'failure'
    assert h.main(['--device', 'cpu', '--mode', 'reference', '--precision', 'default',
                   '--output-dir', str(out)]) == 1
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['status'] == 'FAIL'
    assert 'injected CPU correctness failure' in manifest['error']
    assert not (out/'timings.json').exists()


def test_mock_only_no_tpu_init_on_version_rejection(monkeypatch):
    monkeypatch.setenv('PJRT_DEVICE', 'TPU')
    monkeypatch.setattr(h.importlib.metadata, 'version', lambda _: '0.0.0')
    def forbidden(*args):
        pytest.fail('must reject versions before importing XLA')
    monkeypatch.setattr(h.importlib, 'import_module', forbidden)
    with pytest.raises(RuntimeError, match='candidate'):
        h.load_tpu('default')
