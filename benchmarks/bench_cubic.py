"""Eager/compiled cubic benchmark; generation, copies and compilation untimed.

Run from any directory: python benchmarks/bench_cubic.py
Use --help to configure warmup, repeats, threads and the companion batch cap.
In a Colab GPU runtime, upload/clone the project with its torchyand package,
then run !python /path/to/YAND-TORCH/benchmarks/bench_cubic.py --threads 8
to match the existing CPU baseline's intra-op thread count. Other defaults
are unchanged. A CUDA-enabled PyTorch installation is required for GPU rows.
"""

import argparse
import csv
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

# Support running the script directly without an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchyand.polynomial import cubic_real_roots
from torchyand.polynomial.cubic import _cubic_real_roots_unchecked
from torchyand.polynomial.common import validate_coefficients


BATCH_SIZES = (1, 16, 256, 4096, 65536, 1000000)
FIELDS = (
    "method", "device", "dtype", "batch_size", "median_time_ms",
    "polynomials_per_second", "ns_per_polynomial", "max_normalized_residual",
    "valid_root_count", "skipped_reason",
    "torch_version", "device_name",
    "execution_mode", "correctness_passed",
)


def generate_coefficients(batch_size, dtype, seed):
    """Balanced shuffled family labels; repeated rows alternate double/triple.

    Roots stay at O(1), with separated real roots and complex imaginary parts.
    Small batches contain as many families as their size permits.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.rand((batch_size, 3), dtype=dtype, generator=generator)
    r1 = -1.2 + .4 * random[:, 0]
    r2 = .2 + .4 * random[:, 1]
    r3 = 1.2 + .4 * random[:, 2]
    index = torch.arange(batch_size)
    kind = index % 3
    double = (kind == 2) & ((index // 3) % 2 == 0)
    triple = (kind == 2) & ~double
    r3 = torch.where(double, r2, r3)
    r2 = torch.where(triple, r1, r2)
    r3 = torch.where(triple, r1, r3)
    a = torch.ones_like(r1)
    b = -(r1 + r2 + r3)
    c = r1*r2 + r1*r3 + r2*r3
    d = -r1*r2*r3
    # (x-r1) ((x-r2)^2+v^2): one real plus conjugate pair.
    v = .5 + random[:, 2]
    complex_pair = kind == 1
    b = torch.where(complex_pair, -(r1 + 2*r2), b)
    c = torch.where(complex_pair, 2*r1*r2 + r2*r2 + v*v, c)
    d = torch.where(complex_pair, -r1*(r2*r2 + v*v), d)
    order = torch.randperm(batch_size, generator=generator)
    # Separate contiguous coefficient vectors: no per-repeat unbind/copy cost.
    return tuple(t[order].contiguous() for t in (a, b, c, d))


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def environment_info():
    """Read environment metadata outside every timed region."""
    cuda_available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device()) if cuda_available else None
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cpu_thread_count": torch.get_num_threads(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda or "unavailable",
        "cpu_name": platform.processor() or platform.machine() or "CPU",
        "gpu_name": gpu.name if gpu is not None else "unavailable",
        "gpu_total_memory": f"{gpu.total_memory} bytes ({gpu.total_memory / 2**30:.2f} GiB)" if gpu is not None else "unavailable",
    }


@torch.inference_mode()
def measure(coefficients, method, device, warmup, repeats, solver=None,
            execution_mode="eager"):
    if solver is None:
        solver = lambda *args: cubic_real_roots(*args, method=method)
    # torch.compile is lazy: explicitly invoke it before warmup and timing.
    if execution_mode in ("compiled", "compiled-kernel"):
        result = solver(*coefficients)
        synchronize(device)
        del result
        compare_compiled(coefficients, solver)
    for _ in range(warmup):
        result = solver(*coefficients)
        synchronize(device)
        del result
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        result = solver(*coefficients)
        synchronize(device)
        times.append(time.perf_counter_ns() - start)
        # Exclude result destruction, diagnostic reductions and scalar reads.
        del result
    elapsed_ns = statistics.median(times)
    result = solver(*coefficients)
    synchronize(device)
    residuals = result.residual[result.valid]
    maximum = residuals.max().item() if residuals.numel() else float("nan")
    count = result.valid.sum().item()
    batch_size = coefficients[0].numel()
    return {
        "median_time_ms": elapsed_ns / 1e6,
        "polynomials_per_second": batch_size * 1e9 / elapsed_ns,
        "ns_per_polynomial": elapsed_ns / batch_size,
        "max_normalized_residual": maximum,
        "valid_root_count": count,
        "correctness_passed": True if execution_mode in ("compiled", "compiled-kernel") else "",
    }


def analytic_callable(a, b, c, d):
    """Benchmark-only wrapper; the production solver is unchanged."""
    return cubic_real_roots(a, b, c, d, method="analytic")


def make_compiled_callable(execution_mode="compiled"):
    # Reset specialization caches outside timing so the six batch sizes and
    # two dtypes do not hit a recompile limit and silently fall back to eager.
    torch.compiler.reset()
    target = _cubic_real_roots_unchecked if execution_mode == "compiled-kernel" else analytic_callable
    return torch.compile(target, backend="inductor", fullgraph=execution_mode == "compiled-kernel",
                         dynamic=False)


@torch.inference_mode()
def compare_compiled(coefficients, solver):
    eager = analytic_callable(*coefficients)
    compiled = solver(*coefficients)
    if not torch.equal(eager.degree, compiled.degree):
        raise AssertionError("compiled degree differs from eager")
    if not torch.equal(eager.valid, compiled.valid):
        raise AssertionError("compiled valid mask differs from eager")
    eps = torch.finfo(coefficients[0].dtype).eps
    # Repeated roots are ill-conditioned; bounds remain tight for float64.
    root_tol = 128 * eps
    torch.testing.assert_close(compiled.roots, eager.roots, equal_nan=True,
                               rtol=root_tol, atol=root_tol)
    if not torch.isfinite(compiled.roots[compiled.valid]).all():
        raise AssertionError("compiled valid roots are nonfinite")
    if not torch.isnan(compiled.roots[~compiled.valid]).all():
        raise AssertionError("compiled invalid slots are not NaN")
    # Compare each normalized residual, with an epsilon floor for exact roots.
    allowed = 4 * eager.residual[eager.valid] + 128 * eps
    if not (compiled.residual[compiled.valid] <= allowed).all():
        raise AssertionError("compiled normalized residual materially worsened")


def print_table(rows):
    formatted = []
    for row in rows:
        formatted.append([
            f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
            for key in FIELDS
        ])
    widths = [max(len(key), *(len(row[i]) for row in formatted)) for i, key in enumerate(FIELDS)]
    print(" | ".join(key.ljust(width) for key, width in zip(FIELDS, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU intra-op threads; default keeps the PyTorch setting")
    parser.add_argument("--companion-max-batch", type=int, default=65536,
                        help="Reference eigensolve cap to bound large-batch runtime/memory")
    parser.add_argument("--t4-kernel-only", action="store_true",
                        help="Only T4 CUDA, B=4096/65536/1000000; eager, compiled-public, fullgraph kernel")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.companion_max_batch < 1:
        parser.error("warmup >= 1, repeats >= 2 and companion-max-batch >= 1 required")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
    environment = environment_info()
    cuda_available = environment["cuda_available"]
    for key in ("python_version", "torch_version", "cpu_thread_count",
                "cuda_available", "cuda_version", "gpu_name", "gpu_total_memory"):
        print(f"{key}: {environment[key]}", flush=True)
    print(f"seed={args.seed}; warmup={args.warmup}; repeats={args.repeats}", flush=True)
    print("Timing covers the complete public solver call, including validation "
          "and output construction; generation/copies/compilation/correctness checks "
          "are excluded. Compiled-public permits graph breaks; compiled-kernel "
          "requires fullgraph=True. Kernel input validation is outside timing.", flush=True)
    rows = []
    # Keep the established GPU/CPU baseline intact.
    output = Path(__file__).resolve().parent / "results_cubic_compile_kernel.csv"
    compilation_unavailable = {}
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for dtype in (torch.float32, torch.float64):
            for batch_size in ((4096, 65536, 1000000) if args.t4_kernel_only else BATCH_SIZES):
                cpu_coefficients = generate_coefficients(batch_size, dtype, args.seed)
                for device in (("cuda",) if args.t4_kernel_only else ("cpu", "cuda")):
                    coefficients = None
                    modes = (("analytic", "eager"),
                                                   ("analytic", "compiled"),
                                                   ("analytic", "compiled-kernel"))
                    if not args.t4_kernel_only:
                        modes += (("companion", "eager"),)
                    for method, execution_mode in modes:
                        compilation_key = (device, execution_mode, dtype)
                        row = dict.fromkeys(FIELDS, "")
                        label = f"analytic-{execution_mode}" if method == "analytic" else method
                        row.update(method=label, device=device, execution_mode=execution_mode,
                                   dtype=str(dtype).removeprefix("torch."), batch_size=batch_size,
                                   torch_version=environment["torch_version"],
                                   device_name=environment["cpu_name"] if device == "cpu" else environment["gpu_name"])
                        if device == "cuda" and not cuda_available:
                            row["skipped_reason"] = "CUDA unavailable"
                        elif args.t4_kernel_only and "T4" not in environment["gpu_name"]:
                            row["skipped_reason"] = "T4 required; other GPU results are excluded"
                        elif method == "companion" and batch_size > args.companion_max_batch:
                            row["skipped_reason"] = f"reference batch cap {args.companion_max_batch}"
                        elif execution_mode != "eager" and compilation_key in compilation_unavailable:
                            row["skipped_reason"] = compilation_unavailable[compilation_key]
                        else:
                            try:
                                if coefficients is None:
                                    coefficients = tuple(t.to(device=device) for t in cpu_coefficients)
                                    coefficients = validate_coefficients(*coefficients)
                                    synchronize(device)
                                solver = make_compiled_callable(execution_mode) if execution_mode != "eager" else None
                                row.update(measure(coefficients, method, device, args.warmup,
                                                   args.repeats, solver, execution_mode))
                            except torch.OutOfMemoryError as exc:
                                row["skipped_reason"] = f"out of memory: {str(exc).splitlines()[0]}"
                                if device == "cuda":
                                    torch.cuda.empty_cache()
                            except AssertionError as exc:
                                if execution_mode == "eager":
                                    raise
                                row["correctness_passed"] = False
                                row["skipped_reason"] = f"correctness failed: {str(exc).splitlines()[0]}"
                            except Exception as exc:
                                if execution_mode == "eager":
                                    raise
                                reason = f"compile/runtime unavailable: {type(exc).__name__}: {exc}"
                                print(reason, flush=True)  # Preserve the actual graph-break diagnostics.
                                row["skipped_reason"] = reason
                                compilation_unavailable[compilation_key] = reason
                        rows.append(row)
                        writer.writerow(row)
                        handle.flush()
                        status = row["skipped_reason"] or f"{row['median_time_ms']:.3f} ms"
                        print(f"{label}/{device}/{row['dtype']}/B={batch_size}: {status}", flush=True)
                    del coefficients
                del cpu_coefficients
    print_table(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
