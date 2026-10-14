"""Real projected-Hessian PCG core experiment; no production modifications.

Run: python benchmarks/bench_pcg_compile.py
CPU smoke: python benchmarks/bench_pcg_compile.py --smoke --max-iter 8

All timed variants share a core-only boundary: no public input validation,
geometry/cache construction, RHS projection or result diagnostics. They call
the frozen PCG initialize/step/fixed/finish kernels; the public adaptive/fixed
PCG results are independent correctness controls outside timing. Finish is
included, so diagnostics use the true residual. Compilation and warmup are
untimed. Fullgraph/Inductor failures are reported verbatim, never bypassed.
"""
import argparse
import csv
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_yand_solver import generate, synchronize
from torchyand.geometry import build_geometry
from torchyand.geometry.kernels import level_tangent
from torchyand.operators import ProjectedHessianOperator
from torchyand.linalg import batched_pcg, PCGResult
from torchyand.linalg import kernels
from torchyand.solver import YANDOptions


FIELDS = ('device', 'dtype', 'rhs_count', 'mode', 'compiled', 'wall_time_ms',
          'iterations', 'residual_norm', 'relative_residual', 'converged',
          'breakdown', 'nonpositive_curvature', 'correctness', 'fullgraph',
          'speedup_vs_eager_fixed', 'speedup_vs_eager_adaptive',
          'compiled_fixed_over_eager_fixed', 'compiled_fixed_over_eager_adaptive',
          'T', 'n', 'seed', 'damping', 'max_iter', 'rtol', 'atol',
          'warmup', 'repeats', 'torch_version', 'device_name', 'failure_reason')


def make_core(matvec, projector, max_iter, rtol, mode):
    def core(b):
        b_norm = kernels.norm(b)
        tolerance = rtol * b_norm  # Public default atol=0.
        x0 = torch.zeros_like(b)
        if mode == 'fixed':
            state = kernels.fixed(matvec, b, x0, None, projector, tolerance, max_iter)
        else:
            state = kernels.initialize(matvec, b, x0, None, projector, tolerance)
            for _ in range(max_iter):
                if not (~state[4] & ~state[6]).any().item():
                    break
                state = kernels.step(matvec, state, None, projector, tolerance)
        return kernels.finish(matvec, b, state, b_norm, tolerance, max_iter)
    return core


def system(dtype, device, rhs_count, seed, damping):
    oracle, x = generate(1024, 128, dtype, device, seed)
    cached = oracle.cache(x)
    geom = build_geometry(oracle.gradient(x, cache=cached))
    operator = ProjectedHessianOperator(oracle, x, geom, cache=cached, damping=damping)
    projector = lambda v: level_tangent(v, geom.free_mask, geom.nu)
    rng = torch.Generator(device='cpu').manual_seed(seed + 1)
    shape = (128,) if rhs_count == 1 else (rhs_count, 128)
    raw = (torch.randint(0, 2, shape, generator=rng).to(torch.float64) * 2 - 1)
    b = projector(raw.to(device=device, dtype=dtype))  # No probe normalization.
    error = kernels.norm(projector(b) - b)
    floor = 128 * torch.finfo(dtype).eps * 128**.5 * kernels.norm(b).clamp_min(1)
    assert (error <= floor).all(), 'RHS must satisfy the unchanged PCG membership floor'
    assert geom.level_regular.all(), 'Representative geometry must be regular'
    return operator, projector, b


def check(actual_tuple, reference, b):
    actual = PCGResult(*actual_tuple)
    eps = torch.finfo(b.dtype).eps
    for name in ('x', 'residual_norm', 'relative_residual'):
        a, r = getattr(actual, name), getattr(reference, name)
        assert a.shape == r.shape and a.dtype == r.dtype and a.device == r.device
        assert torch.isfinite(a).all() and torch.isfinite(r).all()
        scale = (r.abs().max().clamp_min(1) if name == 'x'
                 else kernels.norm(b).max().clamp_min(1) if name == 'residual_norm'
                 else torch.ones((), dtype=b.dtype, device=b.device))
        torch.testing.assert_close(a, r, rtol=256 * eps, atol=256 * eps * scale.item())
    for name in ('converged', 'breakdown', 'nonpositive_curvature', 'max_iter_reached'):
        assert torch.equal(getattr(actual, name), getattr(reference, name)), name
    return actual


def measure(call, device, warmup, repeats):
    for _ in range(warmup):
        call()
    synchronize(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = call()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
        del result
    return statistics.median(times)


def numbers(tensor):
    # Preserve each RHS's diagnostics, rather than averaging masks/residuals.
    return tensor.detach().cpu().tolist()


def diagnostics(result):
    return {name: numbers(getattr(result, name)) for name in
            ('iterations', 'residual_norm', 'relative_residual', 'converged',
             'breakdown', 'nonpositive_curvature')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('all', 'cpu', 'cuda'), default='all')
    parser.add_argument('--smoke', action='store_true', help='CPU float32 direction RHS only')
    parser.add_argument('--max-iter', type=int, default=32, help='Fixed-loop budget (same for all variants)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--csv', type=Path, default=ROOT / 'benchmarks/results/pcg_compile_benchmark.csv')
    args = parser.parse_args()
    if args.max_iter < 1 or args.warmup < 1 or args.repeats < 2:
        parser.error('max-iter>=1, warmup>=1, repeats>=2 required')
    if args.smoke and args.device == 'cuda':
        parser.error('--smoke is a CPU-only run')
    devices = [torch.device('cpu')] if args.device in ('all', 'cpu') or args.smoke else []
    if args.device in ('all', 'cuda') and not args.smoke:
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', torch.cuda.current_device()))
        else:
            print('CUDA unavailable: CUDA cases skipped.', flush=True)
    dtypes = (torch.float32,) if args.smoke else (torch.float32, torch.float64)
    rhs_counts = (1,) if args.smoke else (1, 4)
    print(f'PyTorch={torch.__version__}; threads={torch.get_num_threads()}; '
          f'T=1024 n=128 max_iter={args.max_iter}; fullgraph=True; backend=inductor', flush=True)
    print('Core-only timing for ALL variants; public validation/control solves are untimed. '
          'Speedup = eager time / compiled time; also record compiled/eager time ratios.', flush=True)
    rows = []
    failed = False
    for device in devices:
        for dtype in dtypes:
            for rhs_count in rhs_counts:
                damping = YANDOptions().lambda0
                operator, projector, b = system(dtype, device, rhs_count, args.seed, damping)
                rtol = 1e-5 if dtype == torch.float32 else 1e-10
                controls = {mode: batched_pcg(operator.matvec_unchecked, b,
                            projector=projector, max_iter=args.max_iter, mode=mode)
                            for mode in ('adaptive', 'fixed')}
                check(tuple(getattr(controls['adaptive'], name) for name in PCGResult.__dataclass_fields__),
                      controls['fixed'], b)
                times = {}
                for mode, compiled in (('adaptive', False), ('fixed', False), ('fixed', True)):
                    row = dict.fromkeys(FIELDS, '')
                    row.update(device=str(device), dtype=str(dtype).removeprefix('torch.'),
                               rhs_count=rhs_count, mode=mode, compiled=compiled,
                               T=1024, n=128, seed=args.seed, damping=damping, max_iter=args.max_iter,
                               rtol=rtol, atol=0., warmup=args.warmup, repeats=args.repeats,
                               torch_version=str(torch.__version__),
                               device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU')
                    core = make_core(operator.matvec_unchecked, projector, args.max_iter, rtol, mode)
                    if compiled:
                        torch._dynamo.reset()
                    stage = 'compile' if compiled else 'eager'
                    try:
                        if compiled:
                            core = torch.compile(core, fullgraph=True, backend='inductor')
                        print(f'{device}/{dtype}/rhs={rhs_count}/{mode}/compiled={compiled}: '
                              'untimed first call', flush=True)
                        output = core(b)  # Completes lazy compilation before any timing.
                        synchronize(device)
                        row['fullgraph'] = compiled
                        stage = 'correctness'
                        checked = check(output, controls[mode], b)
                        check(output, controls['fixed'], b)
                        # Independently verify reported true residual with the real operator.
                        true_norm = kernels.norm(b - operator.matvec_unchecked(checked.x))
                        eps = torch.finfo(dtype).eps
                        torch.testing.assert_close(checked.residual_norm, true_norm,
                            rtol=256 * eps, atol=256 * eps * kernels.norm(b).max().clamp_min(1).item())
                        row.update(diagnostics(checked), correctness=True)
                        stage = 'timing'
                        row['wall_time_ms'] = measure(lambda: core(b), device, args.warmup, args.repeats)
                        times[(mode, compiled)] = row['wall_time_ms']
                        if compiled:
                            fixed, adaptive = times[('fixed', False)], times[('adaptive', False)]
                            row.update(speedup_vs_eager_fixed=fixed / row['wall_time_ms'],
                                       speedup_vs_eager_adaptive=adaptive / row['wall_time_ms'],
                                       compiled_fixed_over_eager_fixed=row['wall_time_ms'] / fixed,
                                       compiled_fixed_over_eager_adaptive=row['wall_time_ms'] / adaptive)
                        print(f"  median={row['wall_time_ms']:.6f} ms; "
                              f"iterations={row['iterations']}; residual={row['residual_norm']}; "
                              f"relative={row['relative_residual']}; converged={row['converged']}; "
                              f"speedup_fixed={row['speedup_vs_eager_fixed']}; "
                              f"speedup_adaptive={row['speedup_vs_eager_adaptive']}", flush=True)
                    except Exception as exc:
                        failed = True
                        row['failure_reason'] = f'{stage} failure:\n{traceback.format_exc()}'
                        if stage == 'compile':
                            # Backend codegen failures occur after Dynamo graph
                            # capture; distinguish them from actual graph breaks.
                            row['fullgraph'] = ('captured_backend_failed'
                                                if type(exc).__name__ == 'InductorError' else False)
                        if stage == 'correctness':
                            row['correctness'] = False
                        print(row['failure_reason'], flush=True)
                    rows.append(row)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV: {args.csv}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
