"""Scoped compiled-PCG experiment on the unchanged public YAND solver.

Run: python benchmarks/bench_yand_compiled_pcg.py --device cuda
The eager solver uses its existing adaptive default. Only this script's
scoped PCG adapter selects fixed execution, using unchanged PCG defaults
(including max_iter=100). It retains finite and projector-membership checks.
One tensor-argument fullgraph callable per case handles both RHS ranks;
compiler callbacks reveal actual specializations, not forced graph counts.
Cold compilation, warm solver runs and diagnostics are outside steady timing.
No backend fallback. Timed recompilation invalidates the experiment.
"""
import argparse
from contextlib import contextmanager
import csv
import importlib
import json
import math
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
from torchyand.linalg import PCGResult
from torchyand.linalg import kernels
from torchyand.operators import ProjectedHessianOperator
from torchyand.operators.kernels import projected_hessian
from torchyand.geometry.kernels import level_tangent
from torchyand.solver import solve_yand, YANDOptions, SolverStatus
from torch._dynamo.backends.registry import lookup_backend

pcg_module = importlib.import_module('torchyand.linalg.pcg')
affine = importlib.import_module('torchyand.geometry.affine_normal')
stochastic = importlib.import_module('torchyand.logdet.stochastic')


def fixed_core(b, z, A, c, mask, nu, count, regular, damping, rtol, atol, max_iter):
    def projector(v):
        return level_tangent(v, mask, nu)

    def matvec(v):
        return projected_hessian(v, z, A, c, mask, nu, count, regular, damping)

    norm = kernels.norm(b)
    tolerance = atol + rtol * norm
    state = kernels.fixed(matvec, b, torch.zeros_like(b), None, projector, tolerance, max_iter)
    return kernels.finish(matvec, b, state, norm, tolerance, max_iter)


@contextmanager
def scoped_pcg(callback):
    originals = [(module, module.batched_pcg) for module in (affine, stochastic)]
    try:
        for module, _ in originals:
            module.batched_pcg = callback
        yield
    finally:
        for module, original in originals:
            module.batched_pcg = original


def inputs(matvec, b, *, projector=None, **options):
    if options.get('x0') is not None or options.get('M_inv') is not None:
        raise ValueError('Benchmark adapter supports only default x0=0 and M_inv=identity')
    if set(options) - {'x0', 'M_inv', 'rtol', 'atol', 'max_iter', 'mode'}:
        raise ValueError('Unsupported benchmark PCG option')
    _, rtol, atol = affine._options({k: v for k, v in options.items() if k != 'x0'}, b.dtype)
    owner = getattr(matvec, '__self__', None)
    if not isinstance(owner, ProjectedHessianOperator) or getattr(matvec, '__name__', '') != 'matvec_unchecked':
        raise TypeError('Expected the real bound ProjectedHessianOperator.matvec_unchecked')
    if projector is None:
        raise ValueError('A PCG projector is required')
    # Same safe public PCG validation and unchanged membership floor.
    pcg_module._input(b, 'b')
    checked_projector = pcg_module._callback(projector, 'projector')
    pb = checked_projector(b)
    if not torch.isfinite(pb).all().item():
        raise ValueError('projector(b) must be finite')
    floor = 128 * torch.finfo(b.dtype).eps * b.shape[-1]**.5 * kernels.norm(b).clamp_min(1)
    if (kernels.norm(pb - b) > floor).any().item():
        raise ValueError('b must already belong to the projector subspace')
    geom = owner.geom
    tensors = (b, owner.cache.z, owner.oracle.A, owner.oracle.coefficients,
               geom.free_mask, geom.nu, geom.free_count, geom.level_regular, owner.damping)
    # Stabilize strides across changing numerical states, without changing shape.
    return (*(v.contiguous() for v in tensors), rtol, atol, options.get('max_iter', 100))


def signature(args):
    return (tuple((tuple(v.shape), v.stride(), v.dtype, v.device, v.requires_grad)
                  for v in args[:9]), args[9:])


def kind(args):
    if args[0].ndim == 1:
        return 'direction_[n]'
    if args[0].ndim == 2 and args[0].shape[0] == 4:
        return 'stochastic_[4,n]'
    raise ValueError(f'Unexpected solver RHS shape: {args[0].shape}')


class Backend:
    def __init__(self):
        self.inductor = lookup_backend('inductor')
        self.events = []
        self.current_kind = None
        self.locked = False

    def __call__(self, graph, args):
        if self.locked:
            raise RuntimeError('Recompilation during timed solve; steady-state result is invalid')
        event = dict(rhs=self.current_kind, nodes=len(list(graph.graph.nodes)), completed=False)
        self.events.append(event)
        start = time.perf_counter()
        try:
            result = self.inductor(graph, args)
            event['completed'] = True
            return result
        finally:
            event['backend_ms'] = (time.perf_counter() - start) * 1000


class Adapter:
    def __init__(self, device):
        self.device = device
        self.backend = Backend()
        self.compiled = torch.compile(fixed_core, backend=self.backend, fullgraph=True, dynamic=False)
        self.seen = set()
        self.cold_calls = []

    def run(self, args):
        key = signature(args)
        self.backend.current_kind = kind(args)
        before = len(self.backend.events)
        if self.backend.locked and key not in self.seen:
            raise RuntimeError('New PCG tensor metadata during timing; prewarm was incomplete')
        if key not in self.seen:
            synchronize(self.device)
            start = time.perf_counter()
            try:
                output = self.compiled(*args)
                synchronize(self.device)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                self.cold_calls.append(dict(rhs=kind(args), first_call_compile_and_execution_ms=elapsed,
                                            graph_events=len(self.backend.events) - before))
            # Estimate compilation overhead against the same already-warm call.
            samples = []
            for _ in range(3):
                synchronize(self.device)
                start = time.perf_counter()
                self.compiled(*args)
                synchronize(self.device)
                samples.append((time.perf_counter() - start) * 1000)
            self.cold_calls[-1]['estimated_compile_ms'] = max(0., elapsed - statistics.median(samples))
            self.seen.add(key)
            return output
        return self.compiled(*args)

    def __call__(self, matvec, b, *, projector=None, **options):
        return PCGResult(*self.run(inputs(matvec, b, projector=projector, **options)))


def timed_solve(oracle, x, options, seed, device):
    synchronize(device)
    start = time.perf_counter()
    result = solve_yand(oracle, x, options=options, seed=seed)
    synchronize(device)
    return (time.perf_counter() - start) * 1000, result


def compare(eager, compiled, dtype):
    assert torch.equal(eager.status, compiled.status), 'Solver status changed'
    assert torch.equal(eager.converged, compiled.converged), 'Convergence flag changed'
    tol = 256 * torch.finfo(dtype).eps
    for name in ('value', 'kkt_residual'):
        a, b = getattr(eager, name), getattr(compiled, name)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        absolute = tol
        if name == 'kkt_residual':
            # Compare residual behavior at the existing solver stopping scale,
            # rather than requiring identical near-zero residual rounding.
            stop_rtol = 1e-5 if dtype == torch.float32 else 1e-10
            scale = torch.maximum(eager.grad.abs().max(), compiled.grad.abs().max()).clamp_min(1)
            absolute = max(tol, stop_rtol * scale.item())
        torch.testing.assert_close(a, b, rtol=tol, atol=absolute)


def solve_row(result, elapsed, mode, base):
    return dict(base, execution_mode=mode, wall_time_ms=elapsed,
                status=SolverStatus(result.status.item()).name,
                iterations=result.iterations.item(), accepted_steps=result.accepted_steps.item(),
                final_objective=result.value.item(), final_kkt_residual=result.kkt_residual.item())


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('all', 'cpu', 'cuda'), default='all')
    parser.add_argument('--T', type=int, help='Single-case sample count; requires --n and --dtype')
    parser.add_argument('--n', type=int, help='Single-case asset count; requires --T and --dtype')
    parser.add_argument('--dtype', choices=('float32', 'float64'),
                        help='Single-case dtype; requires --T and --n')
    parser.add_argument('--max-iter', type=int, default=YANDOptions().max_iter)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmarks/results')
    args = parser.parse_args(argv)
    if args.max_iter < 1 or args.repeats < 2:
        parser.error('max-iter>=1 and repeats>=2 required')
    selection = (args.T, args.n, args.dtype)
    if any(value is not None for value in selection):
        if not all(value is not None for value in selection):
            parser.error('--T, --n and --dtype must be provided together')
        if args.T < 1 or args.n < 1:
            parser.error('--T and --n must be positive')
    return args


@torch.no_grad()
def main():
    args = _parse_args()
    dtypes = (getattr(torch, args.dtype),) if args.dtype is not None else (torch.float32, torch.float64)
    shapes = ((args.T, args.n),) if args.T is not None else ((1024, 128), (4096, 256))
    devices = [torch.device('cpu')] if args.device in ('all', 'cpu') else []
    if args.device in ('all', 'cuda'):
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', torch.cuda.current_device()))
        else:
            print('CUDA unavailable: skipped.', flush=True)
    rows, summaries = [], []
    failed = False
    for device in devices:
        for dtype in dtypes:
            for T, n in shapes:
                base = dict(device=str(device), dtype=str(dtype).removeprefix('torch.'), T=T, n=n,
                            logdet_mode='stochastic', seed=args.seed, max_iter=args.max_iter,
                            pcg_max_iter=100, repeats=args.repeats, torch_version=str(torch.__version__))
                summary = dict(base, correctness=False)
                adapter = None
                try:
                    print(f'Case {base}', flush=True)
                    oracle, x = generate(T, n, dtype, device, args.seed)
                    options = YANDOptions(logdet_mode='stochastic', max_iter=args.max_iter)
                    captured = {}
                    original = pcg_module.batched_pcg

                    def collect(matvec, b, *, projector=None, **kw):
                        packed = inputs(matvec, b, projector=projector, **kw)
                        captured.setdefault(signature(packed), packed)
                        return original(matvec, b, projector=projector, **kw)

                    # Untimed eager warmup discovers real call metadata (lifted
                    # probe geometry differs from direction geometry).
                    with scoped_pcg(collect):
                        eager = solve_yand(oracle, x, options=options, seed=args.seed)
                    synchronize(device)
                    if {kind(p) for p in captured.values()} != {'direction_[n]', 'stochastic_[4,n]'}:
                        raise RuntimeError('This trajectory did not exercise both requested PCG RHS paths')
                    eager_times = []
                    for _ in range(args.repeats):
                        elapsed, result = timed_solve(oracle, x, options, args.seed, device)
                        eager_times.append(elapsed)
                        compare(eager, result, dtype)
                    eager_ms = statistics.median(eager_times)
                    rows.append(solve_row(eager, eager_ms, 'eager_adaptive_pcg', base))
                    torch._dynamo.reset()  # Only between matrix/dtype/device cases.
                    adapter = Adapter(device)
                    for packed in captured.values():
                        print(f'Precompiling {kind(packed)}; shape={tuple(packed[0].shape)}; '
                              'PCG fixed budget=100', flush=True)
                        actual = adapter.run(packed)
                        reference = fixed_core(*packed)
                        from bench_pcg_compile import check
                        check(actual, PCGResult(*reference), packed[0])
                    with scoped_pcg(adapter):
                        # Full warm solve also compiles any metadata only seen
                        # on the compiled numerical trajectory, before timing.
                        warm = solve_yand(oracle, x, options=options, seed=args.seed)
                        synchronize(device)
                        compare(eager, warm, dtype)
                        adapter.backend.locked = True
                        compiled_times = []
                        for _ in range(args.repeats):
                            elapsed, result = timed_solve(oracle, x, options, args.seed, device)
                            compiled_times.append(elapsed)
                            compare(eager, result, dtype)
                    compiled_ms = statistics.median(compiled_times)
                    compile_ms = sum(c['estimated_compile_ms'] for c in adapter.cold_calls)
                    savings = eager_ms - compiled_ms
                    summary.update(correctness=True, eager_ms=eager_ms, compiled_ms=compiled_ms,
                                   steady_state_speedup=eager_ms / compiled_ms,
                                   total_estimated_compile_cost_ms=compile_ms,
                                   break_even_repeated_solves=(math.ceil(compile_ms / savings) if savings > 0 else None))
                    rows.append(solve_row(result, compiled_ms, 'compiled_fixed_pcg', base))
                    print(json.dumps(summary), flush=True)
                except Exception:
                    failed = True
                    summary['failure_reason'] = traceback.format_exc()
                    print(summary['failure_reason'], flush=True)
                finally:
                    if adapter is not None:
                        summary.update(cold_calls=adapter.cold_calls, graph_events=adapter.backend.events,
                                       successful_graph_compilations=sum(e['completed'] for e in adapter.backend.events),
                                       compilation_by_rhs={rhs: sum(e['completed'] for e in adapter.backend.events if e['rhs'] == rhs)
                                                           for rhs in ('direction_[n]', 'stochastic_[4,n]')})
                        print('PCG compilation report: ' + json.dumps({
                            'cold_calls': adapter.cold_calls,
                            'successful_graphs_by_rhs': summary['compilation_by_rhs'],
                            'total_graph_attempts': len(adapter.backend.events)}), flush=True)
                    summaries.append(summary)
                if failed:
                    break  # No fallback or repeated attempts after compiler/correctness failure.
            if failed:
                break
        if failed:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'yand_compiled_pcg.csv'
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            print(json.dumps(row), flush=True)
    summary_path = args.output_dir / 'yand_compiled_pcg_summary.json'
    summary_path.write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(f'Results: {csv_path}\nCompilation/speedup/break-even: {summary_path}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
