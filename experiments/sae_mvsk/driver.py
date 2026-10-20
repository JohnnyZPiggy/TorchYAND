# SPDX-License-Identifier: MIT
"""Issue 21: pinned-source MVSK oracle/KKT checks and eager timing only."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tarfile
import tempfile
import time
import urllib.request

ALGORITHM_SHA = '8ba99c4448e5494290fbfa0f8e12120115dcb58b'
FRAMEWORK_SHA = '8081b6e02eee0754903898d5afe5b112f1bb23d0'
SEED = 20261021
ALGORITHM_FILES = {
    "LICENSE": "d01f66e39458e466413cc1fec9b0c9fe95be64c25b0a4583e7d3029957f7fd30",
    "torchyand/__init__.py": "da85b9127efd59c2c87883d01419704a82df9ab53c3ccfbc7bd699111bb15dbb",
    "torchyand/constraints/__init__.py": "ad3bdbb17082b32197b2a91cfe64928b95716350e6a2debaf335a94f258dbe9e",
    "torchyand/constraints/active_set.py": "c64feb06acde1c39b7b6253990bc3d64c7d6fa38ed7df775c1184ab624ebf9f6",
    "torchyand/constraints/active_set_kernels.py": "97544a976bf8ec7c7355a1279d9b266d07dfdd0c74e39e58c0d4f6e9fc63991f",
    "torchyand/constraints/feasibility.py": "16e02b9093fe03de0e70ad3da1f812d3ba74c95fdccdcb6fe8cc642b874e51fe",
    "torchyand/constraints/kernels.py": "66806545fab86e90405bccd82f7ddd21512fc109e2a575b0517ab3d18c5d830b",
    "torchyand/data/__init__.py": "e348f451a4dc83574f6415ef3d30cc6debaa75ae8f1a93a492fd9b48ec8014f8",
    "torchyand/data/mvsk_data.py": "c3986b01ef8543a53e569f0e194cc26435b75132d7ed19126d534b7fd6da88e0",
    "torchyand/diagnostics/__init__.py": "0a86897693c266b88775d321e16592f893480ce0b5a970a0ec6166e29cc3b52f",
    "torchyand/diagnostics/kernels.py": "7422bc6d87d6683b13059f8189790fd9cee3eb78694dfafe6dda919eec248401",
    "torchyand/diagnostics/kkt.py": "d4133f55ba237b3bf8c717a178a8687d86acc8c61926571bb2f02a00af49c274",
    "torchyand/geometry/__init__.py": "3aaba7721b2c1ccc36b38230c937591bf4c28e48336cf5965317abe8bf371373",
    "torchyand/geometry/affine_normal.py": "3745f8c1ec403030524057e0e51e2d6a6c79d5b6db7c1a311f1ea71f6d40c676",
    "torchyand/geometry/affine_normal_kernels.py": "4b8dfcdfd95b6c52a83ab5545d23eba4c9dd412cfafaf54c0bd96a3db1b0221f",
    "torchyand/geometry/kernels.py": "4ac8a3b0d65834d95bd6587f2e3ef2d4f7832d08a8422d3dbb7f41d6293631f0",
    "torchyand/geometry/projectors.py": "e43b082d574bdb23e8cbc09bf9b2f3fbe263ddb24b8d2cd77635f17bba2fe6bf",
    "torchyand/linalg/__init__.py": "5fd11087da4cefeb7d341b72a71644345157578483dd0873c616925df3383e27",
    "torchyand/linalg/kernels.py": "f98a48767b0bd9e5d913e8a28dc30abde5afa70e6e99192e2b03dad52aa6e254",
    "torchyand/linalg/pcg.py": "339fa6af3525e96e389d347ae986cca0c9794d24a5f5bcad4b241ce2121bf5e4",
    "torchyand/line_models/__init__.py": "eb54f662395e5a02f50e22c11ce9a0820deef516ea9752b7320219e0e916c0d5",
    "torchyand/line_models/kernels.py": "4b91fe369cc722861fb497b04550ec7c9b3e32ff0768b347e7be0e2091d93343",
    "torchyand/line_models/mvsk.py": "52db60a1051cb4b56799709761b9d4c10febb934ccae2d136cda1347b89b3137",
    "torchyand/line_search/__init__.py": "7d857177e004197077cd4597639b755ddae24beca1a1edf7b97decdbe3c007ac",
    "torchyand/line_search/kernels.py": "aeb8560125dcc20475305e2d6bac6bb731a5be908ea710e2e9d0e050b3aef66c",
    "torchyand/line_search/mvsk.py": "cd032b0e1b57ff1780f089535879096c40ac38b1e14f7a4b092bf12efcdcf9f5",
    "torchyand/logdet/__init__.py": "37bd571706eb988b06b95d2f625fecd196e173475bd83c645e0481770e121232",
    "torchyand/logdet/exact.py": "fa1856fe4067efd5dcc56dd33d7d8d68d025edabb6339069d0217f78aed97f5a",
    "torchyand/logdet/kernels.py": "a2538d31ed9fd0560b6e7dd0b2e8e275c0b154abd03bcffd9a209f7c06f51cfc",
    "torchyand/logdet/stochastic.py": "7e6230df984ea6c7ed8b11cdedaaf233098384324985c0b30f9e28caf0e64e8b",
    "torchyand/operators/__init__.py": "de607d026052cdf129b1f8305d2cefcfbf4c4ca1ff11ba2369b2bac91f935da3",
    "torchyand/operators/kernels.py": "eab130e673ddb4c7f9000a718fd43aa3939cb9285bcc3f16abe697aeb0ac38b8",
    "torchyand/operators/projected_hessian.py": "dd75f1eb36a65164670a0b70cc77c0d9d46f6903d019165e4132c15ccda00594",
    "torchyand/oracle/__init__.py": "f7950fb3f4585a0368d27e059870b462d9d9937931688b4b1df83abafb7bc4a3",
    "torchyand/oracle/kernels.py": "5ac8736e35b0488200a81fad52ac16f36295b94456e0cf5e8577dab098cb455a",
    "torchyand/oracle/mvsk.py": "3d1a2c25e8d68ade905e54370d4107dc860811d2df409724a4b4da0029cdb3d5",
    "torchyand/polynomial/__init__.py": "884d7fa03f5750b67370c84647c277c2c00ddcdc33b48cb5a78f21882178d88f",
    "torchyand/polynomial/common.py": "8c0a4247d74811bf3595ba5fe0a4ad213563bb6e3c51522bdb56418e614958e5",
    "torchyand/polynomial/cubic.py": "0dd2f1601832f8d7993ee17c38c8f2292c8f1715389a458d62ed97abc4278523",
    "torchyand/polynomial/evaluate.py": "bdd7e6d3936c6e14ec71b937f33d5241afcd19d771f981ea96b6ce20c9cec4d4",
    "torchyand/polynomial/linear.py": "ed1168daccb160443ac10089bbcba9c241bd731befac80f4dd31c69b46047bed",
    "torchyand/polynomial/quadratic.py": "022cc90d95d37968e77d73a1c4694a310b90747437f2d4ce26c2785465f0a55d",
    "torchyand/polynomial/quartic_line_search.py": "a848200d0af70be1b0bee7f0ba0ab594740832ee8a2eab269ae506de588e6e8c",
    "torchyand/solver/__init__.py": "2d07a63428fb104c943c7fa09d695dc0332f5626884e6e1904668d41836f9dd0",
    "torchyand/solver/iteration.py": "ac4e0140b68a4477a88d62411ef00a4c8301d571387008400c6b6c9fffefca28",
    "torchyand/solver/kernels.py": "761e7d02226a8dd917d1b4dcc8725d9528ca0fe3be679cdf6f4a1498132528fb",
    "torchyand/solver/options.py": "e2fcc45456315294bc5f34264ec8d47dc4935aff264589285f29e639f7e6678b",
    "torchyand/solver/result.py": "4519b80671065892d0f07705a06f9f5eae00bee26f2321dd5ecdccebbd08d8a2",
    "torchyand/solver/solve.py": "9f881da5e6f12829c532c81b362fa36792cadc8bb662b35e1526d950cad7bb8d",
    "torchyand/solver/state.py": "6333e2ace033ce9bcabb12d186d31109a9fecd356ac47a0925344e2f41bb13ae"
}


def verify_package(root):
    if not ALGORITHM_FILES:
        raise RuntimeError('SOURCE_MANIFEST_EMPTY')
    for name, expected in ALGORITHM_FILES.items():
        path = root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('ALGORITHM_SOURCE_MISMATCH')


def load_source(local_root=None):
    if local_root is not None:
        root, temporary = Path(local_root), None
    else:
        temporary = tempfile.TemporaryDirectory(prefix='torchyand-pinned-')
        root = Path(temporary.name)
        request = urllib.request.Request(
            'https://codeload.github.com/JohnnyZPiggy/TorchYAND/tar.gz/' + ALGORITHM_SHA,
            headers={'User-Agent': 'TorchYAND-Issue21'})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(10*1024*1024+1)
        if len(raw) > 10*1024*1024:
            raise RuntimeError('SOURCE_ARCHIVE_TOO_LARGE')
        found = set()
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            for member in archive:
                name = member.name.partition('/')[2]
                if name not in ALGORITHM_FILES:
                    continue
                if not member.isfile() or name in found or member.size > 1024*1024:
                    raise RuntimeError('UNSAFE_SOURCE_ARCHIVE')
                content = archive.extractfile(member).read()
                if hashlib.sha256(content).hexdigest() != ALGORITHM_FILES[name]:
                    raise RuntimeError('ALGORITHM_SOURCE_MISMATCH')
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                found.add(name)
        if found != set(ALGORITHM_FILES):
            raise RuntimeError('SOURCE_ARCHIVE_INCOMPLETE')
    verify_package(root)
    sys.path.insert(0, str(root))
    return temporary


def reference_value(R, x, c):
    """Independent raw-return objective; no analytic oracle/data/kernel calls."""
    raw64 = R.to(dtype=torch.float64)
    mean64 = raw64.mean(0)
    centered = (raw64-mean64).to(dtype=x.dtype)
    returns = centered @ x
    return (-c[0]*(mean64.to(x.dtype)*x).sum()
            + c[1]*returns.pow(2).mean() - c[2]*returns.pow(3).mean()
            + c[3]*returns.pow(4).mean())


def discrepancy(actual, reference, atol, rtol):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError('NONFINITE_NUMERICAL_RESULT')
    absolute = (actual-reference).abs()
    ratio = absolute/(atol+rtol*reference.abs())
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    return {'max_abs': absolute.max().item(), 'max_tolerance_ratio': ratio.max().item()}


def numerical_checks(device, dtype):
    from torchyand import MVSKData, MVSKOracle, prepare_mvsk_data
    from torchyand.diagnostics import simplex_kkt_residual
    from torchyand.solver import solve_yand, YANDOptions, SolverStatus

    rng = torch.Generator().manual_seed(SEED)
    R = (.2*torch.randn(31, 5, generator=rng, dtype=torch.float64)).to(dtype).to(device)
    x = torch.softmax(torch.randn(5, generator=rng, dtype=torch.float64), 0).to(dtype).to(device)
    c = torch.tensor([1., .8, .3, .2], dtype=dtype, device=device)
    point = x.clone().requires_grad_(True)
    expected = reference_value(R, point, c)
    expected_grad, = torch.autograd.grad(expected, point)
    oracle = MVSKOracle(prepare_mvsk_data(R), c)
    value, gradient, _ = oracle.value_grad(x)
    atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (2e-12, 2e-10)
    result = {'device': device, 'dtype': str(dtype), 'shape': [31, 5], 'seed': SEED,
              'atol': atol, 'rtol': rtol, 'value': value.item(), 'reference_value': expected.item(),
              'objective_error': discrepancy(value, expected.detach(), atol, rtol),
              'gradient_error': discrepancy(gradient, expected_grad, atol, rtol)}

    # On sum(x)=1, f(t,1-t)=t^2-.5t, unique optimum (.25,.75).
    A = torch.tensor([[-1., 0.], [1., 0.]], dtype=dtype, device=device)
    mu = torch.tensor([.5, 0.], dtype=dtype, device=device)
    quadratic = MVSKOracle(MVSKData(A, mu, 2, 2), [1., 1., 0., 0.])
    initial = torch.tensor([.1, .9], dtype=dtype, device=device)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    solved = solve_yand(quadratic, initial,
        options=YANDOptions(logdet_mode='exact', max_iter=10, kkt_rtol=0., kkt_atol=tolerance), seed=123)
    assert solved.status.item() == int(SolverStatus.CONVERGED_KKT) and solved.all_converged
    assert solved.accepted_steps.item() >= 1
    qpoint = solved.x.detach().clone().requires_grad_(True)
    independent_q = qpoint[0].square()-.5*qpoint[0]
    qgrad, = torch.autograd.grad(independent_q, qpoint)
    gamma = qgrad.mean()  # Both coordinates are strictly interior for this problem.
    residual = (qgrad-gamma).abs().max().item()
    feasibility = abs(solved.x.sum().item()-1.)
    optimum_error = (solved.x-torch.tensor([.25, .75], dtype=dtype, device=device)).abs().max().item()
    diagnostic = simplex_kkt_residual(solved.x, qgrad, free_mask=solved.free_mask,
                                    rtol=0., atol=tolerance)
    assert solved.x.min().item() > 0 and feasibility <= tolerance
    assert diagnostic.valid.item() and diagnostic.kkt_stationary.item()
    assert residual <= tolerance and optimum_error <= tolerance
    assert solved.value.item() <= quadratic.value(initial).item()+tolerance
    result['quadratic'] = {'status': SolverStatus(solved.status.item()).name,
        'x': solved.x.tolist(), 'value': solved.value.item(), 'iterations': solved.iterations.item(),
        'accepted_steps': solved.accepted_steps.item(), 'simplex_sum_error': feasibility,
        'minimum_weight': solved.x.min().item(), 'independent_kkt_residual': residual,
        'reported_kkt_residual': diagnostic.kkt_residual.item(), 'known_solution_max_abs': optimum_error,
        'tolerance': tolerance, 'global_optimum_claim': 'ONLY_THIS_CONVEX_QUADRATIC'}
    result['status'] = 'PASS'
    return result


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def timings(call, device):
    def once():
        synchronize(device)
        start = time.perf_counter()
        output = call()
        synchronize(device)
        elapsed = (time.perf_counter()-start)*1000
        del output
        return elapsed
    first = once()
    for _ in range(2):
        once()
    repetitions = [once() for _ in range(7)]
    quartiles = statistics.quantiles(repetitions, n=4, method='inclusive')
    return {'first_call_for_shape_ms': first, 'cold_scope': 'FIRST_CALL_FOR_SHAPE_NOT_PROCESS_COLD',
            'warmups': 2, 'repetitions_ms': repetitions, 'median_ms': statistics.median(repetitions),
            'iqr_ms': quartiles[2]-quartiles[0], 'min_ms': min(repetitions), 'max_ms': max(repetitions)}


def benchmark(include_cuda):
    from torchyand import MVSKData, MVSKOracle, prepare_mvsk_data
    rng = torch.Generator().manual_seed(SEED)
    R = .01*torch.randn(2048, 256, generator=rng, dtype=torch.float32)
    x = torch.softmax(torch.randn(256, 256, generator=rng, dtype=torch.float32), -1)
    data = prepare_mvsk_data(R)
    cpu = MVSKOracle(data, [1., .8, .3, .2])
    # Eager public value_grad includes its validation and fresh projection on both devices.
    with torch.no_grad():
        result = {'T': 2048, 'n': 256, 'batch': 256, 'dtype': 'torch.float32',
            'seed': SEED, 'operation': 'PUBLIC_EAGER_VALUE_GRAD_VALIDATION_AND_PROJECTION',
            'cpu_threads': torch.get_num_threads(), 'generation_and_centering_timed': False,
            'cpu_resident': timings(lambda: cpu.value_grad(x), 'cpu')}
        if not include_cuda:
            result['gpu'] = 'NOT_RUN'
            return result
        synchronize('cuda')
        start = time.perf_counter()
        gpu = MVSKOracle(MVSKData(data.A.cuda(), data.mu.cuda(), data.T, data.n), cpu.coefficients.cuda())
        gx = x.cuda()
        synchronize('cuda')
        result['resident_gpu_setup_transfer_ms'] = (time.perf_counter()-start)*1000
        result['gpu_compute_only'] = timings(lambda: gpu.value_grad(gx), 'cuda')
        def transfer_included():
            moved = MVSKOracle(MVSKData(data.A.cuda(), data.mu.cuda(), data.T, data.n), cpu.coefficients.cuda())
            value, gradient, _ = moved.value_grad(x.cuda())
            return value.cpu(), gradient.cpu()
        result['gpu_transfer_included'] = timings(transfer_included, 'cuda')
        result['gpu_transfer_scope'] = 'H2D_CENTERED_DATA_COEFFICIENTS_X_ORACLE_CONSTRUCTION_VALUE_GRAD_D2H_RESULTS'
        a = cpu.value_grad(x)
        b = gpu.value_grad(gx)
        result['full_batch_cpu_gpu_parity'] = [discrepancy(v.cpu(), r, 3e-6, 3e-5) for v, r in zip(b[:2], a[:2])]
        result['observed_cpu_over_gpu_compute_median'] = result['cpu_resident']['median_ms']/result['gpu_compute_only']['median_ms']
        result['observed_cpu_over_gpu_transfer_median'] = result['cpu_resident']['median_ms']/result['gpu_transfer_included']['median_ms']
    return result


def main():
    global torch
    started, tick = datetime.now(timezone.utc).isoformat(), time.monotonic()
    sha = os.environ['SAE_CANDIDATE_SHA']
    if len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('INVALID_CANDIDATE_SHA')
    cpu_only = os.environ.get('SAE_CPU_ONLY') == '1'
    report = {'source_sha': sha, 'algorithm_source_sha': ALGORITHM_SHA, 'framework_source_sha': FRAMEWORK_SHA,
              'classification': 'SCIENTIFIC_MVSK_PILOT', 'status': 'FAIL', 'utc_start': started,
              'cpu': 'NOT_RUN', 'gpu': 'NOT_RUN', 'tpu': 'NOT_RUN', 'compilation': 'NOT_RUN',
              'algorithm_files_verified': len(ALGORITHM_FILES), 'seed': SEED}
    temporary = None
    try:
        temporary = load_source(os.environ.get('SAE_LOCAL_SOURCE') if cpu_only else None)
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        report['environment'] = {'torch': str(torch.__version__), 'python': platform.python_version(),
            'platform': platform.system(), 'cpu_threads': torch.get_num_threads(), 'cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'tf32': False}
        report['cpu'] = [numerical_checks('cpu', dtype) for dtype in (torch.float32, torch.float64)]
        if not cpu_only:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA_UNAVAILABLE')
            start = time.perf_counter()
            torch.cuda.init()
            synchronize('cuda')
            report['cuda_initialization_ms'] = (time.perf_counter()-start)*1000
            properties = torch.cuda.get_device_properties(0)
            report['device'] = {'requested': 'T4', 'observed': properties.name,
                'compute_capability': [properties.major, properties.minor], 'total_memory_bytes': properties.total_memory,
                'tensor_device': str(torch.empty(1, device='cuda').device)}
            report['gpu'] = [numerical_checks('cuda', dtype) for dtype in (torch.float32, torch.float64)]
        report['benchmark'] = benchmark(not cpu_only)
        report['status'] = 'PASS'
    except Exception as exc:
        # No raw traceback, URL, path, environment or vendor exception message in public artifacts.
        report['error_class'] = type(exc).__name__
        report['error_code'] = str(exc) if str(exc) in ('CUDA_UNAVAILABLE', 'ALGORITHM_SOURCE_MISMATCH',
            'SOURCE_ARCHIVE_TOO_LARGE', 'UNSAFE_SOURCE_ARCHIVE', 'SOURCE_ARCHIVE_INCOMPLETE') else 'SCIENTIFIC_OR_ENVIRONMENT_CHECK_FAILED'
    finally:
        if temporary is not None:
            temporary.cleanup()
        report.update(utc_end=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-tick)
        output = Path(os.environ.get('SAE_ARTIFACT_DIR', '/content'))
        output.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(report, allow_nan=False, indent=2)+'\n').encode()
        (output/'mvsk-science.json').write_bytes(raw)
        print('ARTIFACT mvsk-science.json', len(raw), hashlib.sha256(raw).hexdigest(), flush=True)
    # A scientific FAIL is downloaded as a failed required artifact, never relabeled PASS.
    return report


if __name__ == '__main__':
    main()
