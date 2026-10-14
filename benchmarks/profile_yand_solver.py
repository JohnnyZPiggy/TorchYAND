"""Profile one real eager YAND solve separately on CPU and available CUDA.

Run: python benchmarks/profile_yand_solver.py
CPU smoke + trace: python benchmarks/profile_yand_solver.py --device cpu --export-trace

Uses the end-to-end benchmark's synthetic data, not isolated kernels. Data
preparation/imports/copies and CUDA warmup are outside the profiler. The full
solve_yand call (including its public state initialization) is recorded.
Profiler times include instrumentation overhead and are not benchmark times.
No solver or numerical tolerance changes, hooks, compile or optimization.
"""
import argparse
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_yand_solver import generate, synchronize
from torchyand.solver import solve_yand, YANDOptions, SolverStatus


T, N, SEED = 1024, 128, 123
DTYPE = torch.float64
SMALL_OPS = {
    'aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::pow',
    'aten::square', 'aten::abs', 'aten::neg', 'aten::clamp_min',
    'aten::maximum', 'aten::minimum', 'aten::eq', 'aten::ne', 'aten::lt',
    'aten::le', 'aten::gt', 'aten::ge', 'aten::isfinite',
    'aten::bitwise_and', 'aten::bitwise_or', 'aten::bitwise_not',
}
REDUCTIONS = {'aten::sum', 'aten::mean', 'aten::all', 'aten::any',
              'aten::max', 'aten::min', 'aten::amax', 'aten::amin',
              'aten::prod', 'aten::argmin', 'aten::argmax'}


def operation_counts(averages, events):
    """Report visible host call counts, separately from actual device events."""
    groups = (
        ('matrix/vector products', lambda k: k in {
            'aten::mm', 'aten::matmul', 'aten::bmm', 'aten::mv', 'aten::addmm', 'aten::addmv'}),
        ('reductions', lambda k: k in REDUCTIONS),
        ('norms', lambda k: k.startswith('aten::') and 'norm' in k),
        ('where', lambda k: k == 'aten::where'),
        ('item/scalar extraction', lambda k: k in {'aten::item', 'aten::_local_scalar_dense'}),
        ('synchronization/waits', lambda k: 'synchroniz' in k.lower() or 'streamwait' in k.lower()),
        ('CUDA runtime launches', lambda k: 'launch' in k.lower()
         and ('cuda' in k.lower() or k.startswith('cu'))),
        ('small elementwise ops (launch candidates)', lambda k: k in SMALL_OPS),
    )
    lines = ['High-level counts (nested ATen calls can overlap; host calls are not kernel counts):']
    for label, matches in groups:
        selected = sorted((e for e in averages if matches(e.key)), key=lambda e: -e.count)
        lines.append(f'  {label}: {sum(e.count for e in selected)} calls')
        for event in selected:
            lines.append(f'    {event.key}: {event.count}')
    device_events = sum(str(e.device_type).endswith('CUDA') for e in events)
    lines.append(f'  Visible CUDA device events (kernels/copies/etc.): {device_events}')
    lines.append('  Empty runtime/device groups mean no such events were captured, not zero GPU cost.')
    return '\n'.join(lines)


def device_sort_key(averages):
    # Newer profiler versions use generic device-time names; older versions
    # use CUDA-specific names. Inspect captured events instead of assuming.
    sample = next(iter(averages), None)
    for key in ('self_device_time_total', 'self_cuda_time_total'):
        if sample is not None and hasattr(sample, key):
            return key
    raise RuntimeError('Profiler exposes neither self_device_time_total nor self_cuda_time_total')


def run_profile(device, max_iter, top, output_dir, export_trace):
    oracle, x0 = generate(T, N, DTYPE, device, SEED)
    options = YANDOptions(logdet_mode='stochastic', max_iter=max_iter)
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        if device.type == 'cuda':
            warmup = solve_yand(oracle, x0, options=options, seed=SEED)
            synchronize(device)
            del warmup
        synchronize(device)
        with profile(activities=activities, record_shapes=True,
                     profile_memory=False, with_stack=False) as prof:
            with record_function('solve_yand_end_to_end'):
                result = solve_yand(oracle, x0, options=options, seed=SEED)
            # Complete queued GPU work before ending capture. This explicit
            # boundary synchronization is separate from the solve span.
            synchronize(device)

    averages = prof.key_averages()
    keys = ['self_cpu_time_total', 'cpu_time_total']
    if device.type == 'cuda':
        keys.append(device_sort_key(averages))
    sections = [
        f'device={device}; T={T}; n={N}; dtype=float64; logdet_mode=stochastic; '
        f'seed={SEED}; max_iter={max_iter}',
        f'PyTorch={torch.__version__}; CPU threads={torch.get_num_threads()}; '
        f'activities={[activity.name for activity in activities]}; record_shapes=True; '
        'profile_memory=False; with_stack=False',
        f'status={SolverStatus(result.status.item()).name}; converged={result.converged.item()}; '
        f'iterations={result.iterations.item()}; accepted_steps={result.accepted_steps.item()}',
        'Profiler durations contain instrumentation overhead.',
    ]
    for key in keys:
        sections.extend((f'\nTop {top} operations by {key}',
                         averages.table(sort_by=key, row_limit=top, max_name_column_width=80)))
    sections.append(operation_counts(averages, prof.events()))
    report = '\n'.join(sections)
    print(report, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'yand_solver_profile_{device.type}'
    report_path = output_dir / f'{stem}.txt'
    report_path.write_text(report + '\n', encoding='utf-8')
    print(f'Report: {report_path}', flush=True)
    if export_trace:
        trace_path = output_dir / f'{stem}.json'
        prof.export_chrome_trace(str(trace_path))
        print(f'Chrome trace: {trace_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('all', 'cpu', 'cuda'), default='all')
    parser.add_argument('--max-iter', type=int, default=10)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--export-trace', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmarks/results')
    args = parser.parse_args()
    if args.max_iter < 1 or args.top < 30:
        parser.error('--max-iter must be positive and --top must be at least 30')
    devices = [torch.device('cpu')] if args.device in ('all', 'cpu') else []
    if args.device in ('all', 'cuda'):
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', torch.cuda.current_device()))
        else:
            print('CUDA unavailable: CUDA profile skipped.', flush=True)
    for device in devices:
        print(f'Profiling real solve_yand on {device}...', flush=True)
        run_profile(device, args.max_iter, args.top, args.output_dir, args.export_trace)


if __name__ == '__main__':
    main()
