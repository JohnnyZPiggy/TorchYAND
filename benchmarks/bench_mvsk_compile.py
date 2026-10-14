"""Fullgraph MVSK kernel experiment; run with --t4-kernel-only on Colab GPU.

value_grad includes projection in both modes. HVP/third_action use prebuilt z.
Eager timings include public validation; compiled timings contain tensor math.
Compilation failures are recorded verbatim and never fall back to eager.
"""
import argparse
import csv
from pathlib import Path
import platform
import sys
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchyand import MVSKData, MVSKOracle
from torchyand.oracle import kernels
from bench_mvsk import generate, measure, memory_reason, synchronize

SHAPES = ((4096, 256, 1), (4096, 1024, 1), (256, 128, 4096))
OPERATIONS = ("value_grad", "hvp", "third_action")
FIELDS = ("operation", "execution_mode", "device", "dtype", "T", "n", "batch_size",
          "median_time_ms", "calls_per_second", "ns_per_call",
          "effective_portfolios_per_second", "speedup", "diagnostics_passed",
          "fullgraph", "torch_version", "device_name", "skipped_reason")


def value_grad_with_projection(x, A, mu, c):
    # Benchmark-only composition of the existing frozen tensor kernels.
    return kernels.value_grad(x, kernels.projection(x, A), A, mu, c)


def check(compiled, eager, dtype):
    actual = compiled if isinstance(compiled, tuple) else (compiled,)
    expected = eager[:2] if isinstance(eager, tuple) else (eager,)
    assert len(actual) == len(expected)
    eps = torch.finfo(dtype).eps
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == e.dtype and a.device == e.device
        assert torch.isfinite(a).all().item() and torch.isfinite(e).all().item()
        torch.testing.assert_close(a, e, rtol=256*eps, atol=256*eps)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t4-kernel-only", action="store_true",
                        help="CUDA only, both dtypes and the three specified shapes")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-budget-mb", type=int, default=1024)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.memory_budget_mb < 1 or (args.threads is not None and args.threads < 1):
        parser.error("warmup>=1, repeats>=2, memory-budget-mb>=1 and threads>=1 required")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name() if cuda else "unavailable"
    print(f"Python={platform.python_version()}; torch={torch.__version__}; "
          f"threads={torch.get_num_threads()}; CUDA={cuda}; CUDA version={torch.version.cuda}; "
          f"GPU={gpu_name}; seed={args.seed}; warmup={args.warmup}; repeats={args.repeats}", flush=True)
    print("value_grad includes z projection; HVP/third_action use prebuilt z. "
          "Eager includes validation; compiled-kernel excludes validation. fullgraph=True, no fallback.", flush=True)
    devices = ("cuda",) if args.t4_kernel_only else ("cpu", "cuda")
    output = Path(__file__).resolve().parent / "results_mvsk_compile.csv"
    failed = False
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                for T, n, B in SHAPES:
                    reason = "CUDA unavailable" if device == "cuda" and not cuda else memory_reason(T, n, B, dtype, device, args.memory_budget_mb)
                    if not reason:
                        oracle, x, u, v = generate(T, n, B, dtype, args.seed)
                        if device == "cuda":
                            oracle = MVSKOracle(MVSKData(oracle.A.cuda(), oracle.mu.cuda(), T, n), oracle.coefficients.cuda())
                            x, u, v = x.cuda(), u.cuda(), v.cuda()
                        cached = oracle.cache(x)
                        eager = {
                            "value_grad": lambda: oracle.value_grad(x),
                            "hvp": lambda: oracle.hvp(x, v, cache=cached),
                            "third_action": lambda: oracle.third_action(x, u, v, cache=cached),
                        }
                        specs = {
                            "value_grad": (value_grad_with_projection, (x, oracle.A, oracle.mu, oracle.coefficients)),
                            "hvp": (kernels.hvp, (cached.z, v, oracle.A, oracle.coefficients)),
                            "third_action": (kernels.third_action, (cached.z, u, v, oracle.A, oracle.coefficients)),
                        }
                    for operation in OPERATIONS:
                        base = dict(operation=operation, device=device, dtype=str(dtype).removeprefix("torch."),
                                    T=T, n=n, batch_size=B, torch_version=str(torch.__version__),
                                    device_name=gpu_name if device == "cuda" else platform.processor())
                        eager_time = None
                        for mode in ("eager", "compiled-kernel"):
                            row = dict.fromkeys(FIELDS, "")
                            row.update(base, execution_mode=mode, skipped_reason=reason)
                            if not reason:
                                if mode == "eager":
                                    result = eager[operation]()
                                    check(result[:2] if operation == "value_grad" else result, result, dtype)
                                    del result
                                    row.update(measure(eager[operation], device, B, args.warmup, args.repeats))
                                    eager_time = row["median_time_ms"]
                                    row["speedup"] = 1.0
                                else:
                                    fn, inputs = specs[operation]
                                    try:
                                        # Fresh callable per shape/dtype avoids a
                                        # cross-experiment specialization limit.
                                        torch._dynamo.reset()
                                        compiled = torch.compile(fn, fullgraph=True)
                                        call = lambda: compiled(*inputs)
                                        result = call()  # Lazy compilation, untimed.
                                        synchronize(device)
                                        check(result, eager[operation](), dtype)
                                        del result
                                        # measure also completes warmup before timing.
                                        row.update(measure(call, device, B, args.warmup, args.repeats))
                                        row.update(fullgraph=True, speedup=eager_time/row["median_time_ms"])
                                    except Exception:
                                        failed = True
                                        row.update(fullgraph=False, skipped_reason=traceback.format_exc())
                                        print(f"FULLGRAPH/CORRECTNESS FAILURE {base}:\n{row['skipped_reason']}", flush=True)
                                    finally:
                                        if "compiled" in locals():
                                            del compiled, call
                            writer.writerow(row)
                            handle.flush()
                            latency = row["median_time_ms"]
                            print(f"{device}/{base['dtype']}/T={T}/n={n}/B={B}/{operation}/{mode}: "
                                  f"{latency if latency != '' else 'skipped'} ms; speedup={row['speedup']}", flush=True)
                    if not reason:
                        del eager, specs, cached, oracle, x, u, v
        print(f"Saved {output}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
