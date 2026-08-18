"""PyTorch通路测试 - SinkhornNormalize."""

import sys
import os

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden

SO_NAME = "libsinkhorn_normalize_ops.so"
OP_NAME = "sinkhorn_normalize"
DTYPE = torch.float32
ATOL = 1e-4
RTOL = 1e-4


def run_test(name, x, repeat=10, eps=1e-6):
    """Run one test case, return (name, passed, max_diff)."""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x.npu(), repeat, eps)
    golden = compute_golden(x, repeat=repeat, eps=eps).npu()
    max_diff = torch.max(torch.abs(y - golden)).item()
    passed = torch.allclose(y.cpu(), golden.cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # P1: Reference shape [1, 1024, 4, 4]
    x = torch.randn(1, 1024, 4, 4, dtype=DTYPE)
    results.append(run_test("P1 reference [1,1024,4,4]", x))

    # P2: Smaller batch [1, 16, 4, 4]
    x = torch.randn(1, 16, 4, 4, dtype=DTYPE)
    results.append(run_test("P2 small batch [1,16,4,4]", x))

    # P3: Larger batch [4, 256, 4, 4]
    x = torch.randn(4, 256, 4, 4, dtype=DTYPE)
    results.append(run_test("P3 large batch [4,256,4,4]", x))

    # P4: 2D input [4, 4]
    x = torch.randn(4, 4, dtype=DTYPE)
    results.append(run_test("P4 2D [4,4]", x))

    # P5: All-zero input
    x = torch.zeros(1, 64, 4, 4, dtype=DTYPE)
    results.append(run_test("P5 zeros", x))

    # P6: All-positive input
    x = torch.rand(1, 64, 4, 4, dtype=DTYPE)
    results.append(run_test("P6 uniform [0,1]", x))

    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6e})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
