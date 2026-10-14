"""Reuse a single fullgraph PCG callable across numerical iteration states.

Run: python benchmarks/bench_pcg_compile_reuse.py
CPU smoke: python benchmarks/bench_pcg_compile_reuse.py --smoke --max-iter 8

Per dtype/device: 10 states with a fixed face, then 10 changing face masks.
Only tensor arguments change. x/gradient/geometry/RHS preparation is untimed;
PCG consumes their z/nu/mask representations. All tensor metadata stays fixed.
A counting backend delegates to real Inductor, with no alternative backend or
fallback. Compiler callbacks count capture attempts and native completions;
they are NOT inferred from latency or from disk-cache misses. Native compiler
failure stops the local experiment and preserves its exact traceback.
"""
import argparse
import csv
import json
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
from bench_pcg_compile import check, measure, diagnostics
from torchyand.geometry import build_geometry
from torchyand.geometry.kernels import level_tangent
from torchyand.operators import ProjectedHessianOperator
from torchyand.operators.kernels import projected_hessian
from torchyand.linalg import batched_pcg, PCGResult
from torchyand.linalg import kernels
from torchyand.solver import YANDOptions
from torch._dynamo.backends.registry import lookup_backend


def reusable_fixed_pcg(b, z, A, c, free_mask, nu, free_count, level_regular,
                       damping, max_iter):
    """Benchmark-only composition; all changing numerical state is explicit."""
    def projector(v):
        return level_tangent(v, free_mask, nu)

    def matvec(v):
        return projected_hessian(v, z, A, c, free_mask, nu, free_count,
                                 level_regular, damping)

    b_norm = kernels.norm(b)
    rtol = 1e-5 if b.dtype == torch.float32 else 1e-10
    tolerance = rtol * b_norm  # Existing default atol=0, M_inv=identity, x0=0.
    state = kernels.fixed(matvec, b, torch.zeros_like(b), None, projector,
                          tolerance, max_iter)
    return kernels.finish(matvec, b, state, b_norm, tolerance, max_iter)


class CountingInductor:
    """One callback per Dynamo graph; count failed attempts separately."""
    def __init__(self):
        self.backend = lookup_backend('inductor')
        self.events = []
        self.current_state = None

    def __call__(self, graph, example_inputs):
        event = dict(state=self.current_state, nodes=len(list(graph.graph.nodes)),
                     native_completed=False)
        self.events.append(event)
        start = time.perf_counter()
        try:
            result = self.backend(graph, example_inputs)
            event['native_completed'] = True
            return result
        except Exception:
            event['failure_reason'] = traceback.format_exc()
            raise
        finally:
            event['backend_time_ms'] = (time.perf_counter() - start) * 1000


def states(dtype, device, seed, count):
    oracle, _ = generate(1024, 128, dtype, device, seed)
    rng = torch.Generator(device='cpu').manual_seed(seed + 1)
    raw_probes = (torch.randint(0, 2, (4, 128), generator=rng).to(torch.float64) * 2 - 1)
    raw_probes = raw_probes.to(device=device, dtype=dtype)
    result = []
    metadata = None
    for group in ('A_numeric_only', 'B_changing_mask'):
        for index in range(count):
            # All faces have 112 free assets, with 16 fixed coordinates at 0.
            # In B the active block rotates; rank, shapes and strides stay fixed.
            mask = torch.ones(128, dtype=torch.bool)
            mask[:16] = False
            if group == 'B_changing_mask':
                mask = torch.roll(mask, shifts=5 * (index + 1))
            weights = torch.rand(128, dtype=torch.float64, generator=rng) + .1
            weights = torch.where(mask, weights, 0)
            x = (weights / weights.sum()).to(device=device, dtype=dtype).contiguous()
            mask = mask.to(device=device).contiguous()
            cached = oracle.cache(x)
            gradient = oracle.gradient(x, cache=cached)
            geom = build_geometry(gradient, mask)
            assert geom.level_regular.all()
            damping = torch.tensor(YANDOptions().lambda0 * (1 + index / count),
                                   dtype=dtype, device=device)
            operator = ProjectedHessianOperator(oracle, x, geom, cache=cached, damping=damping)
            projector = lambda v, g=geom: level_tangent(v, g.free_mask, g.nu)
            b = projector(raw_probes).contiguous()  # Same projector, no normalization.
            floor = 128 * torch.finfo(dtype).eps * 128**.5 * kernels.norm(b).clamp_min(1)
            assert (kernels.norm(projector(b) - b) <= floor).all()
            assert (x >= 0).all() and (x[~mask] == 0).all()
            tensors = tuple(v.contiguous() for v in
                            (b, cached.z, oracle.A, oracle.coefficients, mask, geom.nu,
                             geom.free_count, geom.level_regular, damping))
            current = tuple((tuple(v.shape), v.stride(), v.dtype, v.device,
                             v.requires_grad) for v in tensors)
            if metadata is None:
                metadata = current
            assert current == metadata, 'Only tensor values may change between states'
            result.append((group, index, tensors, operator, projector))
    return result


FIELDS = ('device', 'dtype', 'group', 'state', 'eager_fixed_ms', 'compiled_fixed_ms',
          'first_call_ms', 'first_call_compile_events', 'state_compile_events',
          'total_graph_capture_events', 'total_native_compilations', 'recompilation_count',
          'correctness', 'iterations', 'residual_norm', 'relative_residual', 'converged',
          'breakdown', 'nonpositive_curvature', 'T', 'n', 'q', 'max_iter', 'seed',
          'damping', 'torch_version', 'failure_reason')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('all', 'cpu', 'cuda'), default='all')
    parser.add_argument('--smoke', action='store_true', help='Try CPU float32 only; stop on native failure')
    parser.add_argument('--states-per-group', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmarks/results')
    args = parser.parse_args()
    if args.states_per_group < 10 or args.max_iter < 1 or args.warmup < 1 or args.repeats < 2:
        parser.error('states-per-group>=10, max-iter>=1, warmup>=1, repeats>=2 required')
    if args.smoke and args.device == 'cuda':
        parser.error('--smoke is CPU-only')
    devices = [torch.device('cpu')] if args.device in ('all', 'cpu') or args.smoke else []
    if args.device in ('all', 'cuda') and not args.smoke:
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', torch.cuda.current_device()))
        else:
            print('CUDA unavailable: CUDA cases skipped.', flush=True)
    dtypes = (torch.float32,) if args.smoke else (torch.float32, torch.float64)
    rows, summaries = [], []
    abort = False
    print('T=1024 n=128 q=4; tensor-state arguments; fullgraph=True; backend=Inductor; '
          f'seed={args.seed} max_iter={args.max_iter} states/group={args.states_per_group}', flush=True)
    for device in devices:
        for dtype in dtypes:
            sequence = states(dtype, device, args.seed, args.states_per_group)
            # Reset only between dtype/device experiments, NEVER between states.
            torch._dynamo.reset()
            backend = CountingInductor()
            compiled = torch.compile(reusable_fixed_pcg, backend=backend,
                                     fullgraph=True, dynamic=False)
            summary = dict(device=str(device), dtype=str(dtype), planned_states=len(sequence),
                           completed_states=0, reuse_verified=False, mask_reuse_verified=False)
            local_rows = []
            for group, index, tensors, operator, projector in sequence:
                row = dict.fromkeys(FIELDS, '')
                row.update(device=str(device), dtype=str(dtype).removeprefix('torch.'),
                           group=group, state=index, T=1024, n=128, q=4,
                           max_iter=args.max_iter, seed=args.seed, damping=tensors[-1].item(),
                           torch_version=str(torch.__version__))
                backend.current_state = f'{group}/{index}'
                before = len(backend.events)
                call_args = (*tensors, args.max_iter)
                b = tensors[0]
                public = batched_pcg(operator.matvec_unchecked, b, projector=projector,
                                     max_iter=args.max_iter, mode='fixed')
                eager_output = reusable_fixed_pcg(*call_args)
                eager = check(eager_output, public, b)
                row['eager_fixed_ms'] = measure(lambda: reusable_fixed_pcg(*call_args),
                                               device, args.warmup, args.repeats)
                stage = 'compile/first call'
                synchronize(device)
                start = time.perf_counter()
                first_done = False
                try:
                    output = compiled(*call_args)
                    synchronize(device)
                    row['first_call_ms'] = (time.perf_counter() - start) * 1000
                    first_done = True
                    row['first_call_compile_events'] = len(backend.events) - before
                    stage = 'correctness'
                    checked = check(output, eager, b)
                    true_residual = kernels.norm(b - operator.matvec_unchecked(checked.x))
                    eps = torch.finfo(dtype).eps
                    torch.testing.assert_close(checked.residual_norm, true_residual,
                        rtol=256 * eps, atol=256 * eps * kernels.norm(b).max().clamp_min(1).item())
                    row.update(diagnostics(checked), correctness=True)
                    stage = 'steady-state timing'
                    row['compiled_fixed_ms'] = measure(lambda: compiled(*call_args),
                                                      device, args.warmup, args.repeats)
                    summary['completed_states'] += 1
                except Exception:
                    if not first_done:
                        row['first_call_ms'] = (time.perf_counter() - start) * 1000
                        row['first_call_compile_events'] = len(backend.events) - before
                    row['failure_reason'] = f'{stage}:\n{traceback.format_exc()}'
                    if stage == 'correctness':
                        row['correctness'] = False
                    print(row['failure_reason'], flush=True)
                    abort = True  # No fallback, retry, or extra local compiler attempts.
                row.update(state_compile_events=len(backend.events) - before,
                           total_graph_capture_events=len(backend.events),
                           total_native_compilations=sum(e['native_completed'] for e in backend.events),
                           recompilation_count=max(0, len(backend.events) - 1))
                rows.append(row)
                local_rows.append(row)
                print(f"{device}/{dtype}/{group}/{index}: eager={row['eager_fixed_ms']:.6f} ms; "
                      f"first_call={row['first_call_ms']:.6f} ms; compiled={row['compiled_fixed_ms']} ms; "
                      f"compile_events={row['state_compile_events']}; correctness={row['correctness']}", flush=True)
                if abort:
                    break
            summary.update(graph_capture_events=len(backend.events),
                           native_compilations=sum(e['native_completed'] for e in backend.events),
                           recompilation_count=max(0, len(backend.events) - 1),
                           compile_events=backend.events)
            completed = summary['completed_states'] == len(sequence)
            summary['reuse_verified'] = completed and len(backend.events) == 1
            summary['mask_reuse_verified'] = completed and all(
                r['state_compile_events'] == 0 for r in local_rows if r['group'] == 'B_changing_mask')
            for group in ('A_numeric_only', 'B_changing_mask'):
                selected = [r for r in local_rows if r['group'] == group]
                summary[group] = dict(completed_states=sum(r['correctness'] is True for r in selected),
                                     compile_events=sum(r['state_compile_events'] for r in selected))
                times = [r['compiled_fixed_ms'] for r in selected if r['compiled_fixed_ms'] != '']
                if times:
                    summary[group]['median_steady_state_ms'] = statistics.median(times)
            summaries.append(summary)
            print(f"Capture events={summary['graph_capture_events']}; native compilations="
                  f"{summary['native_compilations']}; recompilations={summary['recompilation_count']}; "
                  f"completed={summary['completed_states']}/{len(sequence)}; "
                  f"reuse_verified={summary['reuse_verified']}; "
                  f"mask_reuse_verified={summary['mask_reuse_verified']}", flush=True)
            if abort:
                break
        if abort:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'pcg_compile_reuse.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / 'pcg_compile_reuse_summary.json'
    summary_path.write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(f'CSV: {csv_path}\nSummary: {summary_path}', flush=True)
    return 1 if abort else 0


if __name__ == '__main__':
    raise SystemExit(main())
