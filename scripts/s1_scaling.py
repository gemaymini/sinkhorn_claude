"""把 kernel 时间拆成「每 tile 固定开销」和「每次迭代开销」。

算子签名里 repeat 是入参，所以直接用不同的 repeat 调用同一个 .so 就能做线性拟合：
    t(repeat) = a + b * repeat
  a = 每 tile 固定成本（CopyIn/CopyOut/Exp/减max/launch/sync/指令发射基座）
  b = 每次迭代成本（RowNormalize + ColNormalize）

这直接回答「继续压向量指令还有没有意义」：
  b 占大头 -> 继续优化迭代体
  a 占大头 -> 迭代体再快也没用，要去攻搬运和 launch

用法: python scripts/s1_scaling.py --so build_s1b_div/libsinkhorn_normalize_ops.so
"""

import argparse
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

REPEATS = [1, 2, 3, 5, 8, 10, 15, 20]


def median_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        s.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--mats", type=int, default=1024)
    args = ap.parse_args()

    torch.ops.load_library(args.so)
    op = torch.ops.npu.sinkhorn_normalize
    torch.manual_seed(42)
    x = torch.randn(1, args.mats, 4, 4, dtype=torch.float32).npu()

    print("=" * 70)
    print("迭代次数扫描  mats={}  warmup={}  iters={}".format(
        args.mats, args.warmup, args.iters))
    print("=" * 70)
    print("{:>8} {:>14} {:>16}".format("repeat", "耗时 (us)", "相对 repeat=1"))
    print("-" * 70)

    pts = []
    for r in REPEATS:
        t = median_us(lambda: op(x, r, 1e-6), args.warmup, args.iters)
        pts.append((r, t))
        print("{:>8} {:>14.2f} {:>16.2f}".format(r, t, t - pts[0][1]))

    # 最小二乘拟合 t = a + b*r
    n = len(pts)
    sx = sum(r for r, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(r * r for r, _ in pts)
    sxy = sum(r * t for r, t in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n

    t10 = a + b * 10
    print("-" * 70)
    print("线性拟合 t(repeat) = {:.2f} + {:.2f} × repeat   (单位 us)".format(a, b))
    print()
    print("  repeat=10 时的构成:")
    print("    每 tile 固定开销 a      = {:7.2f} us   ({:.0%})".format(a, a / t10))
    print("    10 次迭代 10×b          = {:7.2f} us   ({:.0%})".format(b * 10, b * 10 / t10))
    print("    合计                    = {:7.2f} us".format(t10))
    print()
    if a > b * 10:
        print("  => 固定开销占大头。继续优化迭代体（RowNorm/ColNorm）收益有限，")
        print("     要去攻 CopyIn/CopyOut 的 MTE 搬运、launch 开销、以及指令发射基座。")
    else:
        print("  => 迭代体占大头。继续压 RowNormalize / ColNormalize 仍然值得。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
