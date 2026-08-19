"""官方 auto_bench.py 口径的评测脚本（含抗漂移的交替测量模式）。

官方口径：
    warmup=200, repeat=500, 每次迭代 perf_counter 包住 forward 再 sync，取 median
    正确性 torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)
    speedup = v0_ms / v1_ms
提交文件同样经过 AST 过滤后 exec，是一次完整的评测预演。

【为什么需要 interleaved 模式】
P0 实测发现 baseline (v0) 在同一台机器上跨进程波动达 ±6%（1173~1314us），
而官方口径是"先把 v0 全测完、再把 v1 全测完"，慢漂移会整体偏到某一侧。
交替测量把两者切成 rounds 段轮流测，能抵消慢漂移，A/B 时的判别力显著更高。
最终对外报数用 --mode official，调优内部对比用 --mode interleaved（默认）。

与官方一致，v0 / v1 是**两个独立文件**：v0 只需 Model，v1 只需 ModelNew。

用法：
    python scripts/bench_official.py                          # 交替模式，默认
    python scripts/bench_official.py --mode official          # 严格官方口径
    python scripts/bench_official.py --mode both
    python scripts/bench_official.py --baseline-ms 1.25       # 固定 baseline，只测 v1（GA 用）
    python scripts/bench_official.py --device cpu --self-test # 本地冒烟
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


def _sample(model, inputs, seed, n, device):
    """采 n 个样本，每个样本 = perf_counter(forward) + sync，单位 ms。"""
    out = []
    for _ in range(n):
        set_seed(seed)
        start = time.perf_counter()
        with torch.no_grad():
            model.forward(*inputs)
        sync_devices(device)
        out.append((time.perf_counter() - start) * 1000.0)
    return out


def _warmup(models, inputs, n, device):
    for m in models:
        for _ in range(n):
            with torch.no_grad():
                m.forward(*inputs)
    sync_devices(device)


def time_official(m0, m1, inputs, seed, warmup, repeat, device):
    """严格复刻 auto_bench.py::time_forward：各自 warmup、各自一次测完。"""
    s = {}
    for key, m in (("v0", m0), ("v1", m1)):
        if m is None:
            continue
        _warmup([m], inputs, warmup, device)
        s[key] = _sample(m, inputs, seed, repeat, device)
    return s


def time_interleaved(m0, m1, inputs, seed, warmup, repeat, device, rounds):
    """把 repeat 切成 rounds 段，v0/v1 轮流测，抵消慢漂移。"""
    models = [m for m in (m0, m1) if m is not None]
    _warmup(models, inputs, warmup, device)
    per = max(1, repeat // rounds)
    s = {"v0": [], "v1": []}
    for _ in range(rounds):
        if m0 is not None:
            s["v0"] += _sample(m0, inputs, seed, per, device)
        if m1 is not None:
            s["v1"] += _sample(m1, inputs, seed, per, device)
    return {k: v for k, v in s.items() if v}


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
    lhs = v0.detach().float()
    rhs = v1.detach().to(v0.device).float()
    ok = torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL, equal_nan=True)
    diff = (lhs - rhs).abs()
    tol = ATOL + RTOL * lhs.abs()
    return ok, diff.max().item(), (diff / tol).max().item()


def call_count():
    try:
        return int(torch.ops.npu.sinkhorn_normalize_call_count())
    except Exception:                                    # noqa: BLE001
        return None


def stats(samples):
    s = sorted(samples)
    n = len(s)
    med = statistics.median(s)
    return {
        "p5": s[n // 20], "median": med, "p95": s[min(n - 1, n * 19 // 20)],
        "iqr": s[min(n - 1, n * 3 // 4)] - s[n // 4],
        "cv": (statistics.pstdev(s) / med) if med else 0.0,
    }


LAST = {}


def report(title, s, baseline_ms):
    print("\n[{}] 单位 us".format(title))
    print("  {:<18} {:>9} {:>9} {:>9} {:>8} {:>7}".format(
        "", "p5", "median", "p95", "IQR", "CV"))
    out = {}
    for key, label in (("v0", "v0 参考实现"), ("v1", "v1 自定义算子")):
        if key not in s:
            continue
        st = stats(s[key])
        out[key] = st
        print("  {:<18} {:>9.2f} {:>9.2f} {:>9.2f} {:>8.2f} {:>6.1%}".format(
            label, st["p5"] * 1e3, st["median"] * 1e3, st["p95"] * 1e3,
            st["iqr"] * 1e3, st["cv"]))
    v0_ms = out["v0"]["median"] if "v0" in out else baseline_ms
    v1_ms = out["v1"]["median"]
    src = "实测" if "v0" in out else "固定常量"
    print("  Speedup = {:.3f}x   (v0={:.2f}us [{}], v1={:.2f}us)".format(
        v0_ms / v1_ms, v0_ms * 1e3, src, v1_ms * 1e3))
    LAST["v1_us"] = v1_ms * 1e3
    LAST["v0_us"] = v0_ms * 1e3
    LAST["cv"] = out["v1"]["cv"]
    LAST["iqr_us"] = out["v1"]["iqr"] * 1e3
    return v0_ms / v1_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0-file", dest="v0_file", default="submission/model_ref.py",
                    help="参考实现文件，须定义 Model / get_inputs / get_init_inputs")
    ap.add_argument("--v1-file", dest="v1_file", default="submission/model_new.py",
                    help="提交文件，须定义 ModelNew / get_inputs / get_init_inputs")
    ap.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="interleaved",
                    choices=["interleaved", "official", "both"])
    ap.add_argument("--rounds", type=int, default=10, help="交替模式的分段数")
    ap.add_argument("--baseline-ms", type=float, default=None,
                    help="固定 baseline（ms），设了就不测 v0，评估提速一倍")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--json", action="store_true", help="末行输出机器可读的 JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="v1 也用参考 Model，仅验证 harness 本身")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    else:
        install_npu_stub()

    # 与官方 build_case() 一致：Model 只从 v0 取、ModelNew 只从 v1 取
    ns0 = load_module(args.v0_file, apply_filter=not args.no_filter)
    ns1 = load_module(args.v1_file, apply_filter=not args.no_filter)

    set_seed(args.seed)
    init0 = ns0["get_init_inputs"]() or []
    set_seed(args.seed)
    init1 = ns1["get_init_inputs"]() or []

    set_seed(args.seed)
    # 官方会把 v1 的输入整个替换成 v0 输入的克隆，这里同样只用 v0 的
    inputs = [t.to(args.device) if isinstance(t, torch.Tensor) else t
              for t in ns0["get_inputs"]()]

    v0_model = ns0["Model"](*init0).to(args.device)
    v1_model = (ns0["Model"](*init0) if args.self_test
                else ns1["ModelNew"](*init1)).to(args.device)

    print("=" * 78)
    print("官方口径评测  device={}  mode={}  warmup={}  repeat={}  seed={}".format(
        args.device, args.mode, args.warmup, args.repeat, args.seed))
    print("  v0 参考: {}".format(args.v0_file))
    print("  v1 提交: {}".format(args.v1_file))
    print("  输入: {}".format([tuple(t.shape) for t in inputs
                               if isinstance(t, torch.Tensor)]))
    print("=" * 78)

    # ---- 正确性（始终实测，不受 --baseline-ms 影响）----
    before = call_count()
    with torch.no_grad():
        y0 = v0_model.forward(*inputs)
        y1 = v1_model.forward(*inputs)
    sync_devices(args.device)
    ok, max_diff, headroom = compare(y0, y1)
    print("\n[正确性] atol={} rtol={}".format(ATOL, RTOL))
    print("  结果        : {}".format("PASSED" if ok else "FAILED"))
    print("  最大绝对误差 : {:.3e}".format(max_diff))
    print("  裕度占用    : {:.4f}   (<1 通过，越小越安全)".format(headroom))

    after = call_count()
    if not args.self_test:
        if after is not None:
            delta = after - (before or 0)
            print("\n[反 fallback 自查] kernel 调用次数增量 = {}{}".format(
                delta, "" if delta > 0 else "   ← 警告：kernel 未被真实调用！"))
        else:
            print("\n[反 fallback 自查] 未注册 call_count 算子，跳过")

    if not ok:
        print("\n正确性未通过，不再计时。")
        if args.json:
            import json
            print("JSON " + json.dumps({"ok": False, "reason": "PRECISION",
                                        "headroom": round(headroom, 6)}))
        return 1

    # ---- 计时 ----
    m0_for_timing = None if args.baseline_ms is not None else v0_model
    results = {}
    if args.mode in ("interleaved", "both"):
        s = time_interleaved(m0_for_timing, v1_model, inputs, args.seed,
                             args.warmup, args.repeat, args.device, args.rounds)
        results["interleaved"] = report(
            "交替测量 rounds={}（抗漂移，调优用这个）".format(args.rounds),
            s, args.baseline_ms)
    if args.mode in ("official", "both"):
        s = time_official(m0_for_timing, v1_model, inputs, args.seed,
                          args.warmup, args.repeat, args.device)
        results["official"] = report("严格官方口径（对外报这个）", s, args.baseline_ms)

    print("\n" + "=" * 78)
    for k, v in results.items():
        print("{:<12} Speedup = {:.3f}x".format(k, v))
    print("=" * 78)
    if args.json:
        import json
        print("JSON " + json.dumps({
            "ok": True, "speedup": round(list(results.values())[-1], 5),
            "v1_us": round(LAST.get("v1_us", 0), 3),
            "v0_us": round(LAST.get("v0_us", 0), 3),
            "cv": round(LAST.get("cv", 0), 5),
            "iqr_us": round(LAST.get("iqr_us", 0), 3),
            "headroom": round(headroom, 6)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
