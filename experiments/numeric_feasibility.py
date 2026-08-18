"""数值近似可行性验证 —— 决定 GA 基因空间里哪些 "risky 基因" 实际可用。

评测口径来自 DLBlas/benchmarks/ks/auto_bench.py:
    torch.allclose(lhs, rhs, atol=1e-2, rtol=1e-2, equal_nan=True)

参考实现 (Task03 sinkhorn) 为 ground truth，各变体与之比较。
只用 CPU，不需要 NPU。
"""
import itertools
import torch

ATOL = 1e-2
RTOL = 1e-2
N0, N1, MHC = 1, 1024, 4
REPEAT = 10
EPS = 1e-6


def reference(x, repeat=REPEAT, eps=EPS):
    """赛题给定的 Model.forward，fp32。"""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def variant(x, *, repeat=REPEAT, eps=EPS, use_max=True, dtype=torch.float32):
    """可配置变体：迭代次数 / eps / softmax 是否减 max / 计算精度。"""
    x = x.to(dtype)
    if use_max:
        x = x.softmax(-1)
    else:  # 免 max：直接 exp / sum(exp)
        e = torch.exp(x)
        x = e / e.sum(-1, keepdim=True)
    x = x + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x.float()


def check(ref, out):
    d = (ref - out).abs()
    ok = torch.allclose(ref, out, atol=ATOL, rtol=RTOL, equal_nan=True)
    tol = ATOL + RTOL * ref.abs()
    headroom = (d / tol).max().item()   # <1 表示通过；越小裕度越大
    return ok, d.max().item(), headroom


def doubly_stochastic_residual(y):
    """离双随机矩阵还有多远 —— 用来解释截断迭代为什么可行。"""
    r = (y.sum(-1) - 1.0).abs().max().item()
    c = (y.sum(-2) - y.sum(-2).mean()).abs().max().item()
    return r, c


CASES = [
    # (名称, kwargs, 风险标签)
    ("G0  基准（完全一致）",                dict(),                                          "safe"),
    ("G1  省掉全部 eps",                    dict(eps=0.0),                                   "low"),
    ("G2  softmax 免减 max",                dict(use_max=False),                             "low"),
    ("G3  G1+G2",                           dict(eps=0.0, use_max=False),                    "low"),
    ("G4  bf16 计算",                       dict(dtype=torch.bfloat16),                      "med"),
    ("G5  fp16 计算",                       dict(dtype=torch.float16),                       "med"),
    ("G6  fp16 + 免 eps + 免 max",          dict(dtype=torch.float16, eps=0.0, use_max=False), "med"),
] + [
    (f"G7  截断到 {k} 次迭代", dict(repeat=k), "aggressive") for k in (9, 8, 7, 6, 5, 4, 3, 2)
] + [
    ("G8  fp16 + 截断到 6 次", dict(dtype=torch.float16, repeat=6, eps=0.0, use_max=False), "aggressive"),
]


def main():
    seeds = list(range(20))
    print(f"评测口径: atol={ATOL}, rtol={RTOL}  |  shape=({N0},{N1},{MHC},{MHC})  |  {len(seeds)} 个随机种子\n")
    print(f"{'变体':<28} {'风险':<11} {'通过':<6} {'最大绝对误差':<14} {'裕度占用':<10}")
    print("-" * 78)

    results = {}
    for name, kw, risk in CASES:
        pass_cnt, max_d, max_hr = 0, 0.0, 0.0
        for s in seeds:
            torch.manual_seed(s)
            x = torch.randn(N0, N1, MHC, MHC, dtype=torch.float32)
            ref = reference(x)
            try:
                out = variant(x, **kw)
            except Exception as e:                       # noqa: BLE001
                print(f"{name:<28} {risk:<11} 异常: {type(e).__name__}: {e}")
                break
            ok, d, hr = check(ref, out)
            pass_cnt += int(ok)
            max_d = max(max_d, d)
            max_hr = max(max_hr, hr)
        else:
            verdict = "全过" if pass_cnt == len(seeds) else f"{pass_cnt}/{len(seeds)}"
            print(f"{name:<28} {risk:<11} {verdict:<6} {max_d:<14.3e} {max_hr:<10.3f}")
            results[name] = (pass_cnt == len(seeds), max_hr)

    print("\n注：'裕度占用' = max|diff| / (atol + rtol*|ref|)，<1 通过，越小越安全\n")

    # Sinkhorn 收敛性：解释截断迭代
    print("Sinkhorn 收敛轨迹（第 k 次迭代结果 vs 第 10 次，seed=42）")
    print(f"{'k':<5} {'max|y_k - y_10|':<20} {'行和偏离1':<14} {'裕度占用':<10}")
    print("-" * 55)
    torch.manual_seed(42)
    x = torch.randn(N0, N1, MHC, MHC, dtype=torch.float32)
    y10 = reference(x)
    for k in range(1, 11):
        yk = variant(x, repeat=k)
        _, d, hr = check(y10, yk)
        r, _ = doubly_stochastic_residual(yk)
        print(f"{k:<5} {d:<20.3e} {r:<14.3e} {hr:<10.4f}")


if __name__ == "__main__":
    main()
