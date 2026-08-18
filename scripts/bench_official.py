"""复刻官方 auto_bench.py 口径的评测脚本。

所有后续调优（含 GA 的 fitness）都必须用这套口径，否则结论不可迁移：
    warmup=200, repeat=500, 每次迭代 perf_counter 包住 forward 再 sync，取 median
    正确性 torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)
    speedup = v0_ms / v1_ms

提交文件同样经过 AST 过滤后再 exec，是一次完整的评测预演。

用法：
    python scripts/bench_official.py --module submission/model_new.py
    python scripts/bench_official.py --device cpu --warmup 5 --repeat 20   # 本地冒烟
"""

import argparse
import ast
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p0_ast_check import filter_module_ast, install_npu_stub  # noqa: E402

import torch  # noqa: E402

ATOL = 1e-2
RTOL = 1e-2


def sync_devices(device):
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def set_seed(seed):
    torch.manual_seed(seed)


def time_forward(model, inputs, seed, warmup, repeat, device):
    """与 auto_bench.py::time_forward 完全一致。返回 median 毫秒。"""
    def one_call():
        with torch.no_grad():
            model.forward(*inputs)

    for _ in range(warmup):
        one_call()
    sync_devices(device)

    samples = []
    for _ in range(repeat):
        set_seed(seed)
        start = time.perf_counter()
        one_call()
        sync_devices(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), samples


def load_module(path, apply_filter=True):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    if apply_filter:
        tree, _, _ = filter_module_ast(tree)
        ast.fix_missing_locations(tree)
    ns = {"__name__": "submitted_module", "__file__": os.path.abspath(path)}
    exec(compile(tree, filename=path, mode="exec"), ns)   # noqa: S102
    return ns


def compare(v0, v1):
    ok = torch.allclose(v0.detach().float(),
                        v1.detach().to(v0.device).float(),
                        atol=ATOL, rtol=RTOL, equal_nan=True)
    diff = (v0.detach().float() - v1.detach().to(v0.device).float()).abs()
    tol = ATOL + RTOL * v0.detach().float().abs()
    return ok, diff.max().item(), (diff / tol).max().item()


def call_count():
    try:
        return int(torch.ops.npu.sinkhorn_normalize_call_count())
    except Exception:                                    # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="submission/model_new.py")
    ap.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-filter", action="store_true",
                    help="不做 AST 过滤（调试用）")
    ap.add_argument("--self-test", action="store_true",
                    help="v1 也用参考 Model，仅验证 harness 本身")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    else:
        install_npu_stub()

    ns = load_module(args.module, apply_filter=not args.no_filter)
    init_args = ns["get_init_inputs"]()

    set_seed(args.seed)
    inputs = [t.to(args.device) if isinstance(t, torch.Tensor) else t
              for t in ns["get_inputs"]()]

    v0_model = ns["Model"](*init_args)
    v1_model = (ns["Model"](*init_args) if args.self_test
                else ns["ModelNew"](*init_args))
    if hasattr(v0_model, "to"):
        v0_model = v0_model.to(args.device)
        v1_model = v1_model.to(args.device)

    print("=" * 74)
    print("官方口径评测  device={}  warmup={}  repeat={}  seed={}".format(
        args.device, args.warmup, args.repeat, args.seed))
    print("  输入: {}".format([tuple(t.shape) for t in inputs
                               if isinstance(t, torch.Tensor)]))
    print("=" * 74)

    # ---- 正确性 ----
    before = call_count()
    with torch.no_grad():
        y0 = v0_model.forward(*inputs)
        y1 = v1_model.forward(*inputs)
    sync_devices(args.device)
    ok, max_diff, headroom = compare(y0, y1)
    print("\n[正确性] atol={} rtol={}".format(ATOL, RTOL))
    print("  结果: {}".format("PASSED" if ok else "FAILED"))
    print("  最大绝对误差: {:.3e}".format(max_diff))
    print("  裕度占用    : {:.4f}   (<1 通过，越小越安全)".format(headroom))

    after = call_count()
    if after is not None and not args.self_test:
        delta = after - (before or 0)
        print("\n[反 fallback 自查] 自定义 kernel 调用次数增量 = {}".format(delta))
        if delta <= 0:
            print("  警告：kernel 没有被真实调用，可能走了 PyTorch 内置算子路径！")
    elif not args.self_test:
        print("\n[反 fallback 自查] 未注册 call_count 算子，跳过（需重新编译 .so）")

    if not ok:
        print("\n正确性未通过，不再计时。")
        return 1

    # ---- 计时 ----
    v0_ms, s0 = time_forward(v0_model, inputs, args.seed, args.warmup, args.repeat, args.device)
    v1_ms, s1 = time_forward(v1_model, inputs, args.seed, args.warmup, args.repeat, args.device)

    def stat(s):
        s = sorted(s)
        n = len(s)
        return s[n // 20], statistics.median(s), s[min(n - 1, n * 19 // 20)]

    p5_0, med0, p95_0 = stat(s0)
    p5_1, med1, p95_1 = stat(s1)

    print("\n[计时] 单位 us")
    print("  {:<22} {:>10} {:>10} {:>10}".format("", "p5", "median", "p95"))
    print("  {:<22} {:>10.2f} {:>10.2f} {:>10.2f}".format(
        "v0 参考实现", p5_0 * 1000, med0 * 1000, p95_0 * 1000))
    print("  {:<22} {:>10.2f} {:>10.2f} {:>10.2f}".format(
        "v1 自定义算子", p5_1 * 1000, med1 * 1000, p95_1 * 1000))

    speedup = v0_ms / v1_ms if v1_ms > 0 else float("inf")
    print("\n" + "=" * 74)
    print("Speedup = {:.3f}x    (v0={:.2f}us, v1={:.2f}us)".format(
        speedup, v0_ms * 1000, v1_ms * 1000))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
