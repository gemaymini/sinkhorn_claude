"""P0-③：把一次 forward 的时间拆成 host CPU / device / sync 三段。

回答的问题：官方口径下 237us 里，有多少是 kernel 真正在算，有多少是
host 侧（dispatch + ACL 调用 + 同步 memcpy）和 launch/sync 的固定开销。

关键产出是 **地板值 M5**：一个最简单的 NPU 算子走完 torch 全链路要多久。
speedup 的理论上限 ≈ 参考实现耗时 / M5，任何 kernel 优化都不可能突破它。

用法（服务器）：
    python scripts/p0_host_breakdown.py --so build_p0_opt/libsinkhorn_normalize_ops.so --tag opt
本地冒烟：
    python scripts/p0_host_breakdown.py --device cpu
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402


def make_sync(device):
    if device == "npu":
        return torch.npu.synchronize
    if device == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def median_us(fn, sync, warmup, repeat, sync_each=True):
    """sync_each=True 复刻官方口径；False 则只测 host 端入队耗时。"""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if sync_each:
            sync()
        samples.append((time.perf_counter() - t0) * 1e6)
    if not sync_each:
        sync()
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=None, help="libsinkhorn_normalize_ops.so 路径")
    ap.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    ap.add_argument("--tag", default="", help="本次运行的标签，便于 A/B 对比")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=500)
    ap.add_argument("--repeat-torch", type=int, default=10,
                    help="参考实现算子多、较慢，用较少迭代")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    sync = make_sync(args.device)

    have_custom = False
    if args.so:
        torch.ops.load_library(args.so)
        have_custom = hasattr(torch.ops.npu, "sinkhorn_normalize")
        if not have_custom:
            print("警告：{} 已加载但未注册 npu::sinkhorn_normalize".format(args.so))

    torch.manual_seed(42)
    x = torch.randn(1, 1024, 4, 4, dtype=torch.float32).to(args.device)
    repeat_n, eps = 10, 1e-6

    def reference():
        with torch.no_grad():
            t = x.softmax(-1) + eps
            t = t / (t.sum(-2, keepdim=True) + eps)
            for _ in range(repeat_n - 1):
                t = t / (t.sum(-1, keepdim=True) + eps)
                t = t / (t.sum(-2, keepdim=True) + eps)
            return t

    print("=" * 78)
    print("P0 时间拆解  device={}  tag={}  warmup={}  repeat={}".format(
        args.device, args.tag or "-", args.warmup, args.repeat))
    if args.so:
        print("  .so = {}".format(args.so))
    print("=" * 78)

    rows = []

    # ---- 地板类测量 ----
    rows.append(("M3  sync 空转（stream 已排空）",
                 median_us(lambda: None, sync, 50, args.repeat)))
    rows.append(("M4  torch.empty_like + sync（分配器）",
                 median_us(lambda: torch.empty_like(x), sync, args.warmup, args.repeat)))
    m5 = median_us(lambda: torch.abs(x), sync, args.warmup, args.repeat)
    rows.append(("M5  单个最简 NPU 算子 + sync  ★地板", m5))

    # ---- 参考实现 ----
    m0 = median_us(reference, sync, 20, max(args.repeat_torch, 20))
    rows.append(("M0  参考实现 forward + sync", m0))

    # ---- 自定义算子 ----
    m1 = m2 = None
    if have_custom:
        op = torch.ops.npu.sinkhorn_normalize

        def custom():
            with torch.no_grad():
                return op(x, repeat_n, eps)

        m1 = median_us(custom, sync, args.warmup, args.repeat, sync_each=True)
        m2 = median_us(custom, sync, args.warmup, args.repeat, sync_each=False)
        rows.append(("M1  自定义算子 forward + sync（官方口径）", m1))
        rows.append(("M2  自定义算子 forward（不 sync）= host 端耗时", m2))

    print("\n{:<44} {:>12}".format("测量项", "median (us)"))
    print("-" * 78)
    for name, v in rows:
        print("{:<44} {:>12.2f}".format(name, v))

    print("\n" + "-" * 78)
    print("推导")
    print("-" * 78)
    print("  任何自定义算子的理论最快耗时 ≈ M5 = {:.2f} us".format(m5))
    print("  => speedup 理论上限 ≈ M0 / M5 = {:.2f} / {:.2f} = {:.1f}x".format(m0, m5, m0 / m5))
    if m1 is not None:
        print("\n  当前 speedup            = M0 / M1 = {:.2f}x".format(m0 / m1))
        print("  host 端占比             = M2 / M1 = {:.1%}".format(m2 / m1))
        print("  device+sync 残差        = M1 - M2 = {:.2f} us".format(m1 - m2))
        print("  距离地板还差            = M1 - M5 = {:.2f} us".format(m1 - m5))
        if m2 > m5:
            print("\n  ⚠ host 端耗时 ({:.2f}us) 已超过单算子地板 ({:.2f}us)，".format(m2, m5))
            print("    说明 host 绑定层存在阻塞调用（同步 memcpy / 设备查询），优先修这里。")
        else:
            print("\n  host 端耗时低于地板，绑定层没有明显阻塞，瓶颈在 kernel 或 launch。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
