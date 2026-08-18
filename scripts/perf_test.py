"""Performance test for SinkhornNormalize kernel.

Measures:
1. PyTorch reference baseline time (us)
2. AscendC kernel time (us) via aclrtSynchronizeStream-based timing
3. Computes speedup
"""

import sys
import os
import time
import statistics

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import SinkhornNormalize

SO_NAME = "libsinkhorn_normalize_ops.so"
OP_NAME = "sinkhorn_normalize"
DTYPE = torch.float32
REPEAT = 10
EPS = 1e-6
WARMUP_ITERS = 10
TIMING_ITERS = 100


def time_torch(fn, *args, warmup=WARMUP_ITERS, iters=TIMING_ITERS, **kwargs):
    """Time a callable that runs on NPU. Returns median time (us)."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.npu.synchronize()

    times = []
    for _ in range(iters):
        if hasattr(torch.npu, 'Event'):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.npu.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)  # ms->us
        else:
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

    return statistics.median(times), min(times), max(times)


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    op_fn = getattr(torch.ops.npu, OP_NAME)

    # Reference PyTorch baseline
    ref_model = SinkhornNormalize(repeat=REPEAT, eps=EPS).npu()

    # Test input: [1, 1024, 4, 4]
    x = torch.randn(1, 1024, 4, 4, dtype=DTYPE).npu()

    print(f"=== SinkhornNormalize perf test ===")
    print(f"Input shape: {tuple(x.shape)}, dtype={DTYPE}, repeat={REPEAT}, eps={EPS}")
    print(f"Warmup: {WARMUP_ITERS} iters, Timing: {TIMING_ITERS} iters\n")

    # 1. PyTorch baseline
    def run_baseline():
        with torch.no_grad():
            ref_model(x)
    ref_med, ref_min, ref_max = time_torch(run_baseline)
    print(f"PyTorch baseline (SinkhornNormalize module on NPU):")
    print(f"  median={ref_med:.2f}us, min={ref_min:.2f}us, max={ref_max:.2f}us\n")

    # 2. AscendC kernel
    def run_kernel():
        op_fn(x, REPEAT, EPS)
    ker_med, ker_min, ker_max = time_torch(run_kernel)
    print(f"AscendC kernel (sinkhorn_normalize):")
    print(f"  median={ker_med:.2f}us, min={ker_min:.2f}us, max={ker_max:.2f}us\n")

    speedup = ref_med / ker_med if ker_med > 0 else float('inf')
    print(f"=== Speedup: {speedup:.3f}x (baseline_us={ref_med:.2f}, kernel_us={ker_med:.2f}) ===")
    return {
        "baseline_us": ref_med,
        "kernel_us": ker_med,
        "speedup": speedup,
    }


if __name__ == "__main__":
    main()
