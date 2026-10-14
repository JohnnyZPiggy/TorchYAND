"""Eager public MVSK benchmark; projection and cached operations timed separately.

Run: python benchmarks/bench_mvsk.py --threads 8
No generation, preprocessing, device copies or diagnostics in timed regions.
"""
import argparse
import csv
import ctypes
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchyand import MVSKData, MVSKOracle, prepare_mvsk_data


SHAPES = [(252, 64, 1), (1000, 256, 1), (4096, 256, 1), (4096, 1024, 1),
          (256, 128, 1), (256, 128, 16), (256, 128, 256), (256, 128, 4096),
          (2048, 256, 1), (2048, 256, 16), (2048, 256, 256)]
OPERATIONS = ("cache", "value_grad", "value", "gradient", "hvp", "third_action")
FIELDS = ("operation", "device", "dtype", "T", "n", "batch_size", "median_time_ms",
          "calls_per_second", "ns_per_call", "effective_portfolios_per_second",
          "uses_cache", "diagnostics_passed", "torch_version", "device_name", "skipped_reason")


def cpu_free_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ("total_physical", "available_physical", "total_page", "available_page",
                           "total_virtual", "available_virtual", "extended")]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available_physical
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])*1024
        except (OSError, ValueError):
            pass
    return None


def memory_reason(T, n, B, dtype, device, budget_mb):
    # Conservative allowance for data/centering/copies, saved z, directions,
    # simultaneous reference outputs and transient elementwise/GEMM buffers.
    estimate = (8*T*n + 24*B*T + 24*B*n)*torch.tensor([], dtype=dtype).element_size()
    free_cpu = cpu_free_memory()
    free = torch.cuda.mem_get_info()[0] if device == "cuda" else free_cpu
    cap = budget_mb*2**20
    if free is not None:
        cap = min(cap, int(free*.25))
    if estimate > cap:
        return f"memory preflight: estimated {estimate/2**20:.1f} MiB exceeds budget {cap/2**20:.1f} MiB"
    if device == "cuda" and free_cpu is not None and estimate > free_cpu*.25:
        return "memory preflight: insufficient CPU staging headroom"
    return ""


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def generate(T, n, B, dtype, seed):
    rng = torch.Generator().manual_seed(seed)
    R = .01*torch.randn(T, n, dtype=dtype, generator=rng)
    shape = (n,) if B == 1 else (B, n)
    x = torch.rand(shape, dtype=dtype, generator=rng)
    x = x/x.sum(-1, keepdim=True)
    u = torch.randn(shape, dtype=dtype, generator=rng)
    v = torch.randn(shape, dtype=dtype, generator=rng)
    u = u/u.norm(dim=-1, keepdim=True)
    v = v/v.norm(dim=-1, keepdim=True)
    return MVSKOracle(prepare_mvsk_data(R), (1., 1., .2, .1)), x, u, v


def callables(oracle, x, u, v):
    cache = oracle.cache(x)  # Constructed before all cached timings.
    return {
        "cache": lambda: oracle.cache(x),
        "value_grad": lambda: oracle.value_grad(x),  # Explicitly no cache.
        "value": lambda: oracle.value(x, cache=cache),
        "gradient": lambda: oracle.gradient(x, cache=cache),
        "hvp": lambda: oracle.hvp(x, v, cache=cache),
        "third_action": lambda: oracle.third_action(x, u, v, cache=cache),
    }


def tensors(operation, result):
    if operation == "cache":
        return (result.z,)
    if operation == "value_grad":
        return (result[0], result[1], result[2].z)
    return (result,)


def diagnostics(oracle, x, u, v, calls, cpu_calls=None):
    expected_shapes = {
        "cache": (x.shape[:-1]+(oracle.T,),),
        "value_grad": (x.shape[:-1], x.shape, x.shape[:-1]+(oracle.T,)),
        "value": (x.shape[:-1],), "gradient": (x.shape,),
        "hvp": (x.shape,), "third_action": (x.shape,),
    }
    uncached = {
        "cache": lambda: oracle.cache(x), "value_grad": lambda: oracle.value_grad(x),
        "value": lambda: oracle.value(x), "gradient": lambda: oracle.gradient(x),
        "hvp": lambda: oracle.hvp(x, v), "third_action": lambda: oracle.third_action(x, u, v),
    }
    tol = 256*torch.finfo(x.dtype).eps
    for name in OPERATIONS:
        actual = tensors(name, calls[name]())
        reference = tensors(name, uncached[name]())
        for a, r, shape in zip(actual, reference, expected_shapes[name]):
            if a.shape != shape or not torch.isfinite(a).all().item():
                raise AssertionError(f"{name}: nonfinite output or incorrect shape")
            torch.testing.assert_close(a, r, rtol=tol, atol=tol)
        if cpu_calls is not None:
            for a, r in zip(actual, tensors(name, cpu_calls[name]())):
                # Sample CUDA parity outside timing; keep transfers small.
                torch.testing.assert_close(a.reshape(-1)[:128].cpu(), r.reshape(-1)[:128],
                                           rtol=tol, atol=tol)


def measure(call, device, B, warmup, repeats):
    for _ in range(warmup):
        result = call()
        synchronize(device)
        del result
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        result = call()
        synchronize(device)
        times.append(time.perf_counter_ns()-start)
        del result
    ns = statistics.median(times)
    return dict(median_time_ms=ns/1e6, calls_per_second=1e9/ns, ns_per_call=ns,
                effective_portfolios_per_second=B*1e9/ns, diagnostics_passed=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-budget-mb", type=int, default=1024)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.memory_budget_mb < 1 or (args.threads is not None and args.threads < 1):
        parser.error("warmup>=1, repeats>=2, memory-budget-mb>=1 and threads>=1 required")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cuda = torch.cuda.is_available()
    cpu_name = platform.processor() or platform.machine() or "CPU"
    gpu_name = torch.cuda.get_device_name() if cuda else "unavailable"
    print(f"Python={platform.python_version()}; torch={torch.__version__}; threads={torch.get_num_threads()}; "
          f"CUDA={cuda}; CUDA version={torch.version.cuda}; GPU={gpu_name}; "
          f"seed={args.seed}; warmup={args.warmup}; repeats={args.repeats}", flush=True)
    print("Full eager public calls, including validation, are timed. All cached operations use a prebuilt cache.", flush=True)
    output = Path(__file__).resolve().parent/"results_mvsk.csv"
    rows = []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for dtype in (torch.float32, torch.float64):
            for T, n, B in SHAPES:
                for device in ("cpu", "cuda"):
                    reason = "CUDA unavailable" if device == "cuda" and not cuda else memory_reason(T, n, B, dtype, device, args.memory_budget_mb)
                    if not reason:
                        cpu_oracle, x_cpu, u_cpu, v_cpu = generate(T, n, B, dtype, args.seed)
                        cpu_calls = callables(cpu_oracle, x_cpu, u_cpu, v_cpu)
                        if device == "cuda":
                            data = MVSKData(cpu_oracle.A.to(device), cpu_oracle.mu.to(device), T, n)
                            oracle = MVSKOracle(data, cpu_oracle.coefficients.to(device))
                            x, u, v = (t.to(device) for t in (x_cpu, u_cpu, v_cpu))
                            calls = callables(oracle, x, u, v)
                        else:
                            oracle, x, u, v, calls = cpu_oracle, x_cpu, u_cpu, v_cpu, cpu_calls
                        synchronize(device)
                        diagnostics(oracle, x, u, v, calls, cpu_calls if device == "cuda" else None)
                    for name in OPERATIONS:
                        row = dict.fromkeys(FIELDS, "")
                        row.update(operation=name, device=device, dtype=str(dtype).removeprefix("torch."),
                                   T=T, n=n, batch_size=B, uses_cache=name not in ("cache", "value_grad"),
                                   torch_version=str(torch.__version__), device_name=cpu_name if device == "cpu" else gpu_name,
                                   skipped_reason=reason)
                        if not reason:
                            row.update(measure(calls[name], device, B, args.warmup, args.repeats))
                        rows.append(row)
                        writer.writerow(row)
                        handle.flush()
                        status = reason or f"{row['median_time_ms']:.4f} ms"
                        print(f"{device}/{row['dtype']}/T={T}/n={n}/B={B}/{name}: {status}", flush=True)
                    if not reason:
                        del calls, cpu_calls, oracle, cpu_oracle, x, u, v, x_cpu, u_cpu, v_cpu
                        if device == "cuda":
                            del data
    widths = [max(len(k), *(len(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])) for row in rows)) for k in FIELDS]
    print(" | ".join(k.ljust(w) for k, w in zip(FIELDS, widths)))
    for row in rows:
        print(" | ".join((f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])).ljust(w) for k, w in zip(FIELDS, widths)))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
