"""Experiment-only CPU reference / separately authorized TPU XLA harness.

CPU never imports XLA. TPU code is unverified; this program never provisions hardware.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torchyand import MVSKData, MVSKOracle
from torchyand.geometry import kernels as geometry
from torchyand.linalg import kernels as pcg
from torchyand.linalg import batched_pcg
from torchyand.operators.kernels import projected_hessian
from torchyand.oracle import kernels as oracle

BASE_SHA = '087accddf28ee566363ae94c8422b5be60fc23f0'
K = 8
PCG_FIELDS = ('x', 'residual_norm', 'relative_residual', 'converged',
              'iterations', 'breakdown', 'nonpositive_curvature', 'max_iter_reached')


def value_grad(x, A, mu, c):
    return oracle.value_grad(x, x @ A.T, A, mu, c)


def fixed_pcg(b, z, A, c, mask, nu, count, regular, damping):
    """All numerical state is explicit; K and callback structure are static."""
    project = lambda v: geometry.level_tangent(v, mask, nu)
    matvec = lambda v: projected_hessian(v, z, A, c, mask, nu, count, regular, damping)
    bn = pcg.norm(b)
    tolerance = (1e-5 if b.dtype == torch.float32 else 1e-10) * bn
    state = pcg.fixed(matvec, b, torch.zeros_like(b), None, project, tolerance, K)
    return pcg.finish(matvec, b, state, bn, tolerance, K)


def make_inputs(seed=123, T=256, n=32, B=4, changing_mask=False, scale=1.0):
    if T < 2 or n < 4 or B != 4 or not 0 < scale <= 1:
        raise ValueError('T>=2, n>=4, B=q=4, 0<scale<=1 required')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    returns = scale * torch.randn(T, n, generator=rng, dtype=torch.float64)
    mu = returns.mean(0)
    A = returns - mu
    c = torch.tensor([1., 1., .2, .1], dtype=torch.float64)
    mask = torch.ones(B, n, dtype=torch.bool)
    mask[1, ::3] = False
    if changing_mask:
        mask[1] = mask[1].roll(seed % n)
    x = torch.rand(B, n, generator=rng, dtype=torch.float64) * mask
    x = x / x.sum(-1, keepdim=True)
    u = torch.randn(B, n, generator=rng, dtype=torch.float64)
    v = torch.randn(B, n, generator=rng, dtype=torch.float64)
    z = x @ A.T
    # Convex PCG control uses its own coefficient vector; oracle cases retain skew.
    cp = c.clone()
    cp[2] = 0
    grad = oracle.gradient(z, A, mu, cp)
    _, _, nu, count, regular, _ = geometry.geometry(grad, mask)
    b = geometry.level_tangent(u, mask, nu)
    b[0] = 0  # Zero RHS alongside nonzero independent systems.
    damping = torch.full((B,), .1, dtype=torch.float64)
    return dict(A=A, mu=mu, c=c, cp=cp, x=x, u=u, v=v, z=z, mask=mask,
                nu=nu, count=count, regular=regular, b=b, damping=damping)


def specs(d):
    hargs = (d['z'], d['A'], d['cp'], d['mask'], d['nu'], d['count'],
             d['regular'], d['damping'])
    return {
        'value_grad': (value_grad, (d['x'], d['A'], d['mu'], d['c'])),
        'hvp': (oracle.hvp, (d['z'], d['v'], d['A'], d['c'])),
        'third_action': (oracle.third_action, (d['z'], d['u'], d['v'], d['A'], d['c'])),
        'projected_hessian': (projected_hessian, (d['v'], *hargs)),
        'fixed_pcg': (fixed_pcg, (d['b'], *hargs)),
    }


def independent_references(d):
    """Autograd scalar objective and explicit dense projector/Hessian on CPU double."""
    A, mu, c = d['A'], d['mu'], d['c']
    def objective(x):
        z = A @ x
        return -c[0] * (mu @ x) + sum(
            sign * c[i-1] * (z ** i).mean() for i, sign in ((2, 1), (3, -1), (4, 1)))
    grad = torch.func.grad(objective)
    values, grads, hvps, thirds = [], [], [], []
    for x, u, v in zip(d['x'], d['u'], d['v']):
        hv = lambda p: torch.func.jvp(grad, (p,), (v,))[1]
        values.append(objective(x)); grads.append(grad(x)); hvps.append(hv(x))
        thirds.append(torch.func.jvp(hv, (x,), (u,))[1])
    m = d['mask'].double()
    P = torch.diag_embed(m) - m.unsqueeze(-1) * m.unsqueeze(-2) / m.sum(-1)[:, None, None]
    P = P - d['nu'].unsqueeze(-1) * d['nu'].unsqueeze(-2)
    weights = 2*d['cp'][1] + 12*d['cp'][3]*d['z']**2
    H = torch.einsum('ti,bt,tj->bij', A, weights, A) / A.shape[0]
    M = P @ H @ P + d['damping'][:, None, None]*P
    mv = lambda v: (M @ v.unsqueeze(-1)).squeeze(-1)
    project = lambda v: (P @ v.unsqueeze(-1)).squeeze(-1)
    result = batched_pcg(mv, d['b'], projector=project, mode='fixed', max_iter=K, rtol=1e-10)
    return dict(value_grad=(torch.stack(values), torch.stack(grads)), hvp=torch.stack(hvps),
                third_action=torch.stack(thirds), projected_hessian=mv(d['v']),
                fixed_pcg=tuple(getattr(result, name) for name in PCG_FIELDS)), M, P


def compare(actual, expected, rtol=2e-8, atol=2e-9):
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    if len(actual) != len(expected):
        raise AssertionError('output arity mismatch')
    errors = []
    for a, e in zip(actual, expected):
        a, e = a.detach().cpu(), e.detach().cpu()
        if a.shape != e.shape:
            raise AssertionError('output shape mismatch')
        if a.dtype != e.dtype:
            raise AssertionError('output dtype mismatch')
        if e.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(e).all():
                raise AssertionError('nonfinite output')
            torch.testing.assert_close(a.double(), e.double(), rtol=rtol, atol=atol)
            diff = a.double() - e.double()
            errors.append(dict(max_abs=diff.abs().max().item(),
                               relative_norm=(diff.norm()/e.double().norm().clamp_min(1e-30)).item()))
        else:
            torch.testing.assert_close(a, e, rtol=0, atol=0)
            errors.append(dict(exact=True))
    return errors


def input_id(d):
    digest = hashlib.sha256()
    for name, v in sorted(d.items()):
        digest.update(f'{name}:{tuple(v.shape)}:{v.dtype}'.encode())
        digest.update(v.contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_case(d):
    refs, M, P = independent_references(d)
    checks = {}
    outputs = {}
    for name, (fn, args) in specs(d).items():
        outputs[name] = fn(*args)
        checks[name] = dict(status='PASS', errors=compare(outputs[name], refs[name]))
    public = MVSKOracle(MVSKData(d['A'], d['mu'], *d['A'].shape), d['c'])
    compare(outputs['value_grad'], public.value_grad(d['x'])[:2])
    compare(outputs['hvp'], public.hvp(d['x'], d['v']))
    compare(outputs['third_action'], public.third_action(d['x'], d['u'], d['v']))
    solved = outputs['fixed_pcg']
    residual = (d['b']-(M @ solved[0].unsqueeze(-1)).squeeze(-1)).norm(dim=-1)
    compare(solved[1], residual)
    compare((P @ solved[0].unsqueeze(-1)).squeeze(-1), solved[0])
    checks['fixed_pcg']['diagnostics'] = {name: tensor.tolist() for name, tensor in zip(PCG_FIELDS[1:], solved[1:])}
    return checks


def validate_tpu_config(versions, env):
    """Pure pre-initialization checks, also usable in MOCK_ONLY tests."""
    if env.get('PJRT_DEVICE') != 'TPU':
        raise RuntimeError('PJRT_DEVICE=TPU required; no automatic backend selection')
    if versions != ('2.8.0', '2.8.0'):
        raise RuntimeError('Only documentation candidate torch=2.8.0 / torch_xla=2.8.0 allowed')
    if env.get('XLA_USE_SPMD') not in (None, '0'):
        raise RuntimeError('SPMD outside experiment scope')


def validate_tpu_identity(device_type, device, real_devices, attributes):
    if (device_type != 'TPU' or device.type != 'xla' or not real_devices
            or any(not x.startswith('TPU:') for x in real_devices)
            or not attributes or any('TPU' not in str(a.get('device_kind', '')).upper() for a in attributes)):
        raise RuntimeError('Actual TPU identity/attributes missing or inconsistent')


def load_tpu(precision):
    # No device initialization before environment and exact package version checks.
    versions = (importlib.metadata.version('torch'), importlib.metadata.version('torch_xla'))
    validate_tpu_config(versions, os.environ)
    xla = importlib.import_module('torch_xla')
    runtime = importlib.import_module('torch_xla.runtime')
    backends = importlib.import_module('torch_xla.backends')
    metrics = importlib.import_module('torch_xla.debug.metrics')
    if (torch.__version__.split('+')[0], xla.__version__.split('+')[0]) != versions:
        raise RuntimeError('Imported and installed versions differ')
    device = xla.device()
    attrs = runtime.global_runtime_device_attributes()
    validate_tpu_identity(runtime.device_type(), device, xla.real_devices(), attrs)
    if runtime.process_count() != 1 or runtime.is_spmd():
        raise RuntimeError('Single process, non-SPMD execution required')
    backends.set_mat_mul_precision(precision)
    if backends.get_mat_mul_precision() != precision:
        raise RuntimeError('Precision setting not honored')
    xla.sync(wait=True)
    return device, xla, metrics, attrs


def metric_snapshot(metrics):
    compile_data = metrics.metric_data('CompileTime')
    return dict(compile_samples=compile_data[0] if compile_data else 0,
                counters={n: metrics.counter_value(n) for n in metrics.counter_names()},
                report=metrics.metrics_report())


def tpu_case(d, functions, device, xla, metrics, warmup, repeats):
    """Untested TPU-only path. Synchronization includes execution, never fallback."""
    quantized = {k: v.float() if v.is_floating_point() else v for k, v in d.items()}
    # CPU float32 is the algorithm/status control; CPU float64 records backend errors.
    cpu32 = {name: fn(*args) for name, (fn, args) in specs(quantized).items()}
    double = {k: v.double() if v.is_floating_point() else v for k, v in quantized.items()}
    cpu64 = {name: fn(*args) for name, (fn, args) in specs(double).items()}
    transferred = {k: v.to(device) for k, v in quantized.items()}
    xla.sync(wait=True)
    checks, timings = {}, []
    for name, (_, args) in specs(transferred).items():
        call = lambda: functions[name](*args)
        before = metric_snapshot(metrics)
        start = time.perf_counter(); output = call(); xla.sync(wait=True)
        cold = (time.perf_counter()-start)*1000
        first = metric_snapshot(metrics)
        rtol, atol = (5e-4, 5e-5) if name == 'fixed_pcg' else (5e-5, 5e-6)
        errors = compare(output, cpu32[name], rtol, atol)
        # Float64 comparison enforces provisional numerical accuracy too; status uses
        # float32 control because stopping tolerance is deliberately dtype-dependent.
        actual_float = output[:3] if name == 'fixed_pcg' else output
        ref_float = cpu64[name][:3] if name == 'fixed_pcg' else cpu64[name]
        actual_float = tuple(v.double() for v in actual_float) if isinstance(actual_float, tuple) else actual_float.double()
        errors64 = compare(actual_float, ref_float, rtol, atol)
        diagnostics = {}
        if name == 'fixed_pcg':
            solution = output[0].detach().cpu().double()
            _, projected_args = specs(double)['projected_hessian']
            true_residual = (double['b']-projected_hessian(solution, *projected_args[1:])).norm(dim=-1)
            compare(output[1].detach().cpu().double(), true_residual, rtol, atol)
            diagnostics = {key: value.detach().cpu().tolist()
                           for key, value in zip(PCG_FIELDS[1:], output[1:])}
            diagnostics.update(cpu64_true_residual=true_residual.tolist(),
                               float32_stopping_tolerance=(1e-5*double['b'].norm(dim=-1)).tolist())
        for _ in range(warmup):
            output = call(); xla.sync(wait=True)
        warmed = metric_snapshot(metrics)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter(); output = call(); xla.sync(wait=True)
            samples.append((time.perf_counter()-start)*1000)
        after = metric_snapshot(metrics)
        fallback = {n: v-before['counters'].get(n, 0) for n, v in after['counters'].items()
                    if n.startswith('aten::') and v > before['counters'].get(n, 0)}
        if fallback:
            raise RuntimeError(f'Unexpected CPU fallback counters: {fallback}')
        compare(output, cpu32[name], rtol, atol)
        checks[name] = dict(status='PASS', errors=errors, cpu64_errors=errors64, diagnostics=diagnostics,
                            tolerances=dict(rtol=rtol, atol=atol, approval='PROVISIONAL'))
        timings.append(dict(operation=name, cold_first_ms=cold, median_ms=statistics.median(samples),
                            p95_ms=sorted(samples)[min(len(samples)-1, int(.95*len(samples)))],
                            samples_ms=samples, before=before, first=first, warmed=warmed, after=after,
                            warm_compile_samples=after['compile_samples']-warmed['compile_samples'],
                            reuse_status='UNVERIFIED_REQUIRES_METRIC_REVIEW'))
    return checks, timings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'tpu'), required=True)
    parser.add_argument('--mode', choices=('reference', 'lazy', 'xla-compile'), required=True)
    parser.add_argument('--precision', choices=('default', 'high', 'highest'), required=True)
    parser.add_argument('--case', choices=('smoke', 'reuse'), default='smoke')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeats < 2 or not 0 <= args.seed < 2**63-20:
        parser.error('warmup>=1, repeats>=2 and 0<=seed<2**63-20 required')
    if ((args.device == 'cpu' and (args.mode != 'reference' or args.precision != 'default'))
            or (args.device == 'tpu' and args.mode == 'reference')):
        parser.error('cpu requires reference/default; tpu requires lazy or xla-compile')
    return args


def main(argv=None):
    args = parse_args(argv)
    # Never overwrite an earlier run, including a failed one.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=10).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, timeout=10).strip())
    manifest = dict(schema_version=1, source_sha=sha, source_dirty=dirty, baseline_sha=BASE_SHA,
                    time_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    torch=str(torch.__version__), device=args.device, mode=args.mode,
                    precision=args.precision, seed=args.seed, case=args.case, K=K,
                    warmup=args.warmup, repeats=args.repeats, command=sys.argv,
                    shapes=dict(T=256 if args.case == 'smoke' else 1024,
                                n=32 if args.case == 'smoke' else 128, B=4, q=4),
                    dtype='float64' if args.device == 'cpu' else 'float32',
                    TPU_DEVICE='NOT RUN', XLA_COMPILE='NOT RUN', TPU_BENCHMARK='NOT RUN',
                    TPU_ALLOCATED='NO_BY_HARNESS', status='FAIL', input_ids=[])
    checks, timings = [], []
    try:
        if args.device == 'tpu':
            device, xla, metrics, attrs = load_tpu(args.precision)
            manifest.update(torch_xla=importlib.metadata.version('torch_xla'), hardware=attrs,
                            libtpu=importlib.metadata.version('libtpu'), PJRT_DEVICE='TPU',
                            TPU_DEVICE='VERIFIED', XLA_COMPILE='NOT RUN')
        functions = None
        total = 1 if args.case == 'smoke' else 20
        for i in range(total):
            d = make_inputs(args.seed+i, T=256 if total == 1 else 1024,
                            n=32 if total == 1 else 128, changing_mask=i >= 10)
            manifest['input_ids'].append(input_id(d))
            if args.device == 'cpu':
                checked = cpu_case(d)
            else:
                if functions is None:
                    functions = {name: xla.compile(fn, full_graph=True) if args.mode == 'xla-compile' else fn
                                 for name, (fn, _) in specs(d).items()}
                checked, measured = tpu_case(d, functions, device, xla, metrics, args.warmup, args.repeats)
                timings.append(dict(state=i, measurements=measured))
            checks.append(dict(state=i, input_id=input_id(d), checks=checked))
        manifest['status'] = 'PASS'
        if args.device == 'tpu':
            manifest.update(TPU_BENCHMARK='PASS', XLA_COMPILE='PASS' if args.mode == 'xla-compile' else 'NOT RUN')
    except Exception as error:
        manifest['error'] = f'{type(error).__name__}: {error}'
        print(manifest['error'], file=sys.stderr)
    finally:
        for name, data in (('manifest.json', manifest), ('correctness.json', checks)):
            (args.output_dir/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if timings:
            (args.output_dir/'timings.json').write_text(json.dumps(timings, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        (args.output_dir/'summary.md').write_text(
            f"device={args.device}; mode={args.mode}; status={manifest['status']}\n\n"
            f"TPU_DEVICE={manifest['TPU_DEVICE']}; XLA_COMPILE={manifest['XLA_COMPILE']}; "
            f"TPU_BENCHMARK={manifest['TPU_BENCHMARK']}\n\n"
            f"Error: {manifest.get('error', 'none')}\n\n"
            'No speedup ratio computed. CPU outputs are numerical evidence only.\n', encoding='utf-8')
    print(f"{args.device}/{args.mode}: {manifest['status']}; reports: {args.output_dir}")
    return 0 if manifest['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
