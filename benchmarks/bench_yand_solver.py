"""End-to-end eager YAND benchmark, with all solver defaults preserved.

Run: python benchmarks/bench_yand_solver.py
Smoke: python benchmarks/bench_yand_solver.py --smoke --device cpu

Data preparation, imports, device copies, CUDA warmup and diagnostics are
outside timing. Timing includes the complete public solve_yand call, including
its state initialization. Each solve starts from the same x0 and RNG seed.
No compile, kernel substitutions, tolerance overrides or solver optimization.
"""
import argparse
import csv
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchyand import MVSKOracle, prepare_mvsk_data
from torchyand.solver import solve_yand, YANDOptions, SolverStatus


# Compact coverage of each requested T and n, rather than a Cartesian grid.
CASES = ((256, 32, 'stochastic'), (1024, 128, 'stochastic'),
         (4096, 256, 'stochastic'), (256, 32, 'exact'))
FIELDS = ('device', 'dtype', 'T', 'n', 'logdet_mode', 'wall_time_ms',
          'final_status', 'converged', 'iterations', 'accepted_steps',
          'face_updates', 'fallback_count', 'initial_objective',
          'final_objective', 'final_kkt_residual', 'objective_decrease',
          'seed', 'max_iter', 'torch_version', 'device_name')


def generate(T, n, dtype, device, seed):
    # Generate in CPU float64 so devices/dtypes receive the same problem.
    rng = torch.Generator(device='cpu').manual_seed(seed)
    returns = .1 * torch.randn(T, n, dtype=torch.float64, generator=rng)
    returns = returns + torch.linspace(-.002, .002, n, dtype=torch.float64)
    x0 = torch.rand(n, dtype=torch.float64, generator=rng) + .1
    x0 = x0 / x0.sum()
    returns = returns.to(device=device, dtype=dtype)
    x0 = x0.to(device=device, dtype=dtype)
    oracle = MVSKOracle(prepare_mvsk_data(returns), [1., 1., .2, .1])
    return oracle, x0


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_case(T, n, mode, dtype, device, seed, max_iter):
    oracle, x0 = generate(T, n, dtype, device, seed)
    options = YANDOptions(logdet_mode=mode, max_iter=max_iter)
    with torch.no_grad():
        initial = oracle.value(x0).item()
        if device.type == 'cuda':
            # Warm the actual full pipeline and CUDA libraries before timing.
            warmup = solve_yand(oracle, x0, options=options, seed=seed)
            synchronize(device)
            del warmup
        synchronize(device)
        start = time.perf_counter()
        result = solve_yand(oracle, x0, options=options, seed=seed)
        synchronize(device)
        wall_time_ms = (time.perf_counter() - start) * 1000

    # All scalar reads / output formatting happen after the timed region.
    final = result.value.item()
    return dict(device=str(device), dtype=str(dtype).removeprefix('torch.'),
                T=T, n=n, logdet_mode=mode, wall_time_ms=wall_time_ms,
                final_status=SolverStatus(result.status.item()).name,
                converged=result.converged.item(), iterations=result.iterations.item(),
                accepted_steps=result.accepted_steps.item(),
                face_updates=result.face_updates.item(), fallback_count=result.fallback_count.item(),
                initial_objective=initial, final_objective=final,
                final_kkt_residual=result.kkt_residual.item(), objective_decrease=initial-final,
                seed=seed, max_iter=max_iter, torch_version=str(torch.__version__),
                device_name=(torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'))


def print_table(rows):
    formatted = []
    for row in rows:
        formatted.append({key: (f'{value:.6g}' if isinstance(value, float) else str(value))
                          for key, value in row.items()})
    widths = {key: max(len(key), *(len(row[key]) for row in formatted)) for key in FIELDS}
    print(' | '.join(key.ljust(widths[key]) for key in FIELDS))
    print('-+-'.join('-' * widths[key] for key in FIELDS))
    for row in formatted:
        print(' | '.join(row[key].ljust(widths[key]) for key in FIELDS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('all', 'cpu', 'cuda'), default='all')
    parser.add_argument('--smoke', action='store_true', help='Only (T,n)=(256,32), stochastic, both dtypes')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-iter', type=int, default=YANDOptions().max_iter,
                        help='Iteration budget; all numerical tolerances retain solver defaults')
    parser.add_argument('--csv', type=Path, default=ROOT / 'benchmarks/results/yand_solver_benchmark.csv')
    args = parser.parse_args()
    if args.max_iter < 1:
        parser.error('--max-iter must be positive')
    devices = [torch.device('cpu')] if args.device in ('all', 'cpu') else []
    if args.device in ('all', 'cuda'):
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', torch.cuda.current_device()))
        else:
            print('CUDA unavailable: CUDA cases skipped.')
    if not devices:
        return
    cases = CASES[:1] if args.smoke else CASES
    print(f'PyTorch {torch.__version__}; CPU threads={torch.get_num_threads()}; '
          f'seed={args.seed}; max_iter={args.max_iter}; eager end-to-end wall timing', flush=True)
    rows = []
    for device in devices:
        for dtype in (torch.float32, torch.float64):
            for T, n, mode in cases:
                print(f'Running {device} {dtype} T={T} n={n} {mode}', flush=True)
                rows.append(benchmark_case(T, n, mode, dtype, device, args.seed, args.max_iter))
    print_table(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV: {args.csv}')


if __name__ == '__main__':
    main()
