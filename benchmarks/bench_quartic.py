"""Eager/compiled-public quartic benchmark; solver implementations unchanged.

Run: python benchmarks/bench_quartic.py --threads 8
Generation, device copies and diagnostics are outside the timed region.
"""

import argparse
import csv
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchyand.polynomial import minimize_quartic_interval
from torchyand.polynomial.quartic_line_search import _minimize_quartic_interval_unchecked
from torchyand.polynomial.common import validate_coefficients


BATCH_SIZES = (4096, 65536, 1000000)
FIELDS = (
    "device", "dtype", "batch_size", "median_time_ms", "problems_per_second",
    "ns_per_problem", "success_count", "max_normalized_stationary_residual",
    "max_normalized_known_candidate_gap", "interval_violation_count",
    "torch_version", "device_name", "skipped_reason",
    "execution_mode", "correctness_passed", "speedup",
)


def generate_problems(batch_size, dtype, seed):
    """Integrate known derivative roots to obtain six balanced shuffled families.

    Family 0: generic quartic, 3 distinct stationary roots.
    Family 1: quartic, 1 real stationary root + conjugate pair.
    Family 2/3: cubic/quadratic degeneration.
    Family 4: increasing quartic on a positive interval.
    Family 5: symmetric double well. Tiny batches cover as many as possible.
    The known roots provide a lightweight untimed diagnostic, not a replacement
    for the independent NumPy correctness tests (coefficients undergo rounding).
    """
    generator = torch.Generator().manual_seed(seed)
    random = torch.rand((batch_size, 6), generator=generator, dtype=dtype)
    kind = torch.arange(batch_size) % 6
    r1, r2, r3 = -1.3+.4*random[:, 0], -.2+.4*random[:, 1], .9+.4*random[:, 2]
    a = .5+random[:, 3]
    # Derivative is a*x³+b*x²+c*x+d, initially with three known real roots.
    b = -a*(r1+r2+r3)
    c = a*(r1*r2+r1*r3+r2*r3)
    d = -a*r1*r2*r3
    known = torch.stack((r1, r2, r3), -1)
    valid = torch.ones_like(known, dtype=torch.bool)
    one = kind == 1
    u, v = r2, .5+random[:, 4]
    b = torch.where(one, -a*(r1+2*u), b)
    c = torch.where(one, a*(2*r1*u+u*u+v*v), c)
    d = torch.where(one, -a*r1*(u*u+v*v), d)
    valid[one, 1:] = False
    cubic = kind == 2
    b = torch.where(cubic, a, b)
    c = torch.where(cubic, -a*(r1+r3), c)
    d = torch.where(cubic, a*r1*r3, d)
    known[cubic, 1] = r3[cubic]
    valid[cubic, 2] = False
    a = torch.where(cubic, torch.zeros_like(a), a)
    quadratic = kind == 3
    c = torch.where(quadratic, .5+random[:, 3], c)
    d = torch.where(quadratic, -c*r2, d)
    a = torch.where(quadratic, torch.zeros_like(a), a)
    b = torch.where(quadratic, torch.zeros_like(b), b)
    known[quadratic, 0] = r2[quadratic]
    valid[quadratic, 1:] = False
    monotone = kind == 4
    a = torch.where(monotone, torch.ones_like(a), a)
    b = torch.where(monotone, torch.zeros_like(b), b)
    c = torch.where(monotone, torch.ones_like(c), c)
    d = torch.where(monotone, torch.ones_like(d), d)
    valid[monotone] = False  # its real stationary root is outside this interval
    wells = kind == 5
    s = .7+.4*random[:, 2]
    a = torch.where(wells, torch.ones_like(a), a)
    b = torch.where(wells, torch.zeros_like(b), b)
    c = torch.where(wells, -s*s, c)
    d = torch.where(wells, torch.zeros_like(d), d)
    known[wells] = torch.stack((-s, torch.zeros_like(s), s), -1)[wells]
    coeff = torch.stack((2*random[:, 5]-1, d, c/2, b/3, a/4), -1)
    lower, upper = -1.5-random[:, 4], 1.5+random[:, 5]
    lower = torch.where(monotone, .1+.4*random[:, 4], lower)
    order = torch.randperm(batch_size, generator=generator)
    return tuple(t[order].contiguous() for t in (coeff, lower, upper, known, valid))


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def shifted(coeff, x):
    a1, a2, a3, a4 = (coeff[:, i].unsqueeze(-1) for i in range(1, 5))
    return x*(a1+x*(a2+x*(a3+x*a4)))


@torch.inference_mode()
def measure(inputs, device, warmup, repeats, solver=minimize_quartic_interval,
            execution_mode="eager"):
    coeff, lower, upper, known, valid = inputs
    if execution_mode != "eager":
        # Lazy compilation, correctness and warmup all finish before timing.
        result = solver(coeff, lower, upper)
        synchronize(device)
        del result
        compare_public(inputs, solver)
    for _ in range(warmup):
        result = solver(coeff, lower, upper)
        synchronize(device)
        del result
    durations = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        result = solver(coeff, lower, upper)
        synchronize(device)
        durations.append(time.perf_counter_ns()-start)
        del result
    median_ns = statistics.median(durations)
    # An additional untimed call for diagnostics; no reductions/scalar reads
    # or result transfers occur inside the solver timing interval.
    result = solver(coeff, lower, upper)
    synchronize(device)
    residual = result.root_result.residual[result.root_result.valid]
    max_residual = residual.max().item() if residual.numel() else float("nan")
    # Compare shifted objective with known construction roots + endpoints in
    # float64 on the same device; no large A0 cancellation in this diagnostic.
    candidates = torch.cat((lower[:, None], upper[:, None], known), -1).double()
    masks = torch.cat((torch.ones_like(valid[:, :2]), valid & (known >= lower[:, None]) & (known <= upper[:, None])), -1)
    values = shifted(coeff.double(), torch.where(masks, candidates, torch.zeros_like(candidates)))
    best = torch.where(masks, values, torch.full_like(values, float("inf"))).amin(-1)
    chosen = shifted(coeff.double(), result.x.double().unsqueeze(-1)).squeeze(-1)
    scale = torch.where(masks, values.abs(), torch.zeros_like(values)).amax(-1).clamp_min(1)
    gap = ((chosen-best).clamp_min(0)/scale)[result.success]
    count = coeff.shape[0]
    return {
        "median_time_ms": median_ns/1e6,
        "problems_per_second": count*1e9/median_ns,
        "ns_per_problem": median_ns/count,
        "success_count": result.success.sum().item(),
        "max_normalized_stationary_residual": max_residual,
        "max_normalized_known_candidate_gap": gap.max().item() if gap.numel() else float("nan"),
        "interval_violation_count": (result.success & ((result.x < lower) | (result.x > upper))).sum().item(),
        "correctness_passed": True if execution_mode != "eager" else "",
    }


@torch.inference_mode()
def compare_public(inputs, solver):
    coeff, lower, upper = inputs[:3]
    eager = minimize_quartic_interval(coeff, lower, upper)
    compiled = solver(coeff, lower, upper)
    if not torch.equal(eager.success, compiled.success):
        raise AssertionError("compiled success mask differs from eager")
    tol = 256 * torch.finfo(coeff.dtype).eps
    torch.testing.assert_close(compiled.x, eager.x, rtol=tol, atol=tol, equal_nan=True)
    torch.testing.assert_close(compiled.value, eager.value, rtol=tol, atol=tol, equal_nan=True)
    if not torch.equal(compiled.candidate_valid, eager.candidate_valid):
        raise AssertionError("compiled candidate validity differs from eager")
    torch.testing.assert_close(compiled.candidate_x, eager.candidate_x,
                               rtol=tol, atol=tol, equal_nan=True)
    def selected_slot(result):
        distance = (result.candidate_x-result.x.unsqueeze(-1)).abs()
        distance = torch.where(result.candidate_valid, distance, torch.full_like(distance, float("inf")))
        return distance.argmin(-1)
    if not torch.equal(selected_slot(compiled)[eager.success], selected_slot(eager)[eager.success]):
        raise AssertionError("compiled candidate selection differs from eager")
    violations = compiled.success & ((compiled.x < lower) | (compiled.x > upper))
    if violations.any().item():
        raise AssertionError("compiled interval violations are nonzero")


def compiled_public():
    torch.compiler.reset()  # Outside timing; no silent specialization-limit fallback.
    return torch.compile(minimize_quartic_interval, backend="inductor",
                         fullgraph=False, dynamic=False)


def compiled_kernel():
    torch.compiler.reset()
    # No fallback or suppress_errors: fullgraph failures are reported verbatim.
    return torch.compile(_minimize_quartic_interval_unchecked, backend="inductor",
                         fullgraph=True, dynamic=False)


def validate_kernel_inputs(inputs):
    coeff, lower, upper, known, valid = inputs
    *parts, lower, upper = validate_coefficients(*coeff.unbind(-1), lower, upper)
    if (lower > upper).any().item():
        raise ValueError("Every interval must satisfy lower <= upper")
    return torch.stack(parts, -1), lower, upper, known, valid


def print_table(rows):
    text = [[f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in FIELDS] for row in rows]
    widths = [max(len(key), *(len(row[i]) for row in text)) for i, key in enumerate(FIELDS)]
    print(" | ".join(key.ljust(w) for key, w in zip(FIELDS, widths)))
    for row in text:
        print(" | ".join(value.ljust(w) for value, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--t4-kernel-only", action="store_true",
                        help="Only T4 CUDA, float32/64 and B=4096/65536/1000000")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or (args.threads is not None and args.threads < 1):
        parser.error("warmup>=1, repeats>=2, threads>=1 required")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cuda_available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device()) if cuda_available else None
    cpu_name = platform.processor() or platform.machine() or "CPU"
    print(f"Python={platform.python_version()}; torch={torch.__version__}; threads={torch.get_num_threads()}; "
          f"CUDA={cuda_available}; CUDA version={torch.version.cuda}; GPU={gpu.name if gpu else 'unavailable'}; "
          f"GPU memory={gpu.total_memory if gpu else 'unavailable'}; seed={args.seed}; "
          f"warmup={args.warmup}; repeats={args.repeats}", flush=True)
    output = Path(__file__).resolve().parent / "results_quartic_compile_kernel.csv"
    rows = []
    unavailable = {}
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for dtype in (torch.float32, torch.float64):
            for batch_size in BATCH_SIZES:
                cpu_inputs = generate_problems(batch_size, dtype, args.seed)
                for device in (("cuda",) if args.t4_kernel_only else ("cpu", "cuda")):
                    inputs = None
                    eager_ms = None
                    for mode in ("eager", "compiled-public", "compiled-kernel"):
                        compile_key = (device, mode, dtype)
                        row = dict.fromkeys(FIELDS, "")
                        row.update(device=device, execution_mode=mode,
                                   dtype=str(dtype).removeprefix("torch."), batch_size=batch_size,
                                   torch_version=str(torch.__version__), device_name=cpu_name if device == "cpu" else (gpu.name if gpu else "unavailable"))
                        if device == "cuda" and not cuda_available:
                            row["skipped_reason"] = "CUDA unavailable"
                        elif args.t4_kernel_only and "T4" not in gpu.name:
                            row["skipped_reason"] = "T4 required; other GPUs excluded"
                        elif mode != "eager" and compile_key in unavailable:
                            row["skipped_reason"] = unavailable[compile_key]
                        else:
                            try:
                                if inputs is None:
                                    inputs = tuple(t.to(device=device) for t in cpu_inputs)
                                    synchronize(device)
                                solver = (compiled_kernel() if mode == "compiled-kernel" else
                                          compiled_public() if mode == "compiled-public" else minimize_quartic_interval)
                                measured_inputs = validate_kernel_inputs(inputs) if mode == "compiled-kernel" else inputs
                                synchronize(device)
                                row.update(measure(measured_inputs, device, args.warmup, args.repeats, solver, mode))
                                if mode == "eager":
                                    eager_ms = row["median_time_ms"]
                                elif eager_ms is not None:
                                    row["speedup"] = eager_ms / row["median_time_ms"]
                            except torch.OutOfMemoryError as exc:
                                row["skipped_reason"] = f"out of memory: {exc}"
                                if device == "cuda":
                                    torch.cuda.empty_cache()
                            except AssertionError as exc:
                                if mode == "eager":
                                    raise
                                row["correctness_passed"] = False
                                row["skipped_reason"] = f"correctness failed: {exc}"
                            except Exception as exc:
                                if mode == "eager":
                                    raise
                                row["skipped_reason"] = f"compile/runtime unavailable: {traceback.format_exc()}"
                                unavailable[compile_key] = row["skipped_reason"]
                                print(row["skipped_reason"], flush=True)
                        rows.append(row)
                        writer.writerow(row)
                        handle.flush()
                        status = row["skipped_reason"] or f"{row['median_time_ms']:.3f} ms, success={row['success_count']}/{batch_size}"
                        print(f"{mode}/{device}/{row['dtype']}/B={batch_size}: {status}", flush=True)
                    del inputs
                del cpu_inputs
    print_table(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
