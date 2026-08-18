"""在 CPU 上逐条模拟 S1 kernel 的数据流，验证索引运算与数值逻辑。

模拟的是 kernel 里每一步的**确切语义**：
  padded 布局 z[32m + 8r + c]、paddingValue=-1000、Exp 后填充位为 0、
  BlockReduceSum 按 8 个一组求和且结果连续、Brcb 每个和广播到一个 block、
  ColNormalize 的 4 个行组按 32B block 对齐相加、尾块的 nE 向上取偶。

验证不了 AscendC API 的真实行为，但能把索引错误、尾块错误、
填充位污染、inf/NaN 这些逻辑问题全部在本地打出来。
"""

import numpy as np

ATOL, RTOL = 1e-2, 1e-2
PAD_FILL = -1000.0
TINY = 1e-30


def reference(x, repeat=10, eps=1e-6):
    """赛题参考实现（fp32，含 eps）。x: (M,4,4)"""
    x = x.astype(np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    x = e / e.sum(-1, keepdims=True) + eps
    x = x / (x.sum(-2, keepdims=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdims=True) + eps)
        x = x / (x.sum(-2, keepdims=True) + eps)
    return x.astype(np.float32)


def s1_kernel(x_flat, n, repeat=10, eps=1e-6):
    """逐条模拟 kernel。x_flat: (n*16,) float32"""
    nE = (n + 1) & ~1                       # BlockReduceSum 每 repeat 覆盖 2 个矩阵
    col_eps = max(eps, TINY)                # 填充列的和恒为 0，除数必须非零

    # ---- CopyIn: n*4 行，每行 4 个有效 + 右补 4 个 PAD_FILL ----
    z = np.empty(nE * 32, dtype=np.float32)
    for i in range(n * 4):
        z[i * 8: i * 8 + 4] = x_flat[i * 4: i * 4 + 4]
        z[i * 8 + 4: i * 8 + 8] = PAD_FILL
    if nE > n:                              # 尾部补齐的矩阵填 0
        z[n * 32: nE * 32] = 0.0

    # ---- BuildEpsVector: [eps,eps,eps,eps,0,0,0,0] 重复 ----
    eps_pad = np.tile(
        np.array([eps] * 4 + [0.0] * 4, dtype=np.float32), nE * 4)

    # ---- Exp: 填充位 exp(-1000) 下溢为精确的 0 ----
    with np.errstate(under="ignore"):
        z = np.exp(z).astype(np.float32)
    pad_lanes = (np.arange(nE * 32) % 8) >= 4
    assert np.all(z[pad_lanes][: n * 16] == 0.0), "填充位没有变成精确的 0"

    def row_norm(z, denom_eps):
        row_sum = z.reshape(-1, 8).sum(axis=1)              # BlockReduceSum，结果连续
        row_sum = (row_sum + denom_eps).astype(np.float32)  # Adds
        recip = (1.0 / row_sum).astype(np.float32)          # Reciprocal
        return (z * np.repeat(recip, 8)).astype(np.float32)  # Brcb + Mul

    def col_norm(z):
        zz = z.reshape(nE, 4, 8)                            # [矩阵, 行组, block内]
        col_sum = zz.sum(axis=1)                            # 3 次带 stride 的 Add
        col_sum = (col_sum + col_eps).astype(np.float32)    # Adds
        cr = (1.0 / col_sum).astype(np.float32)             # Reciprocal
        return (zz * cr[:, None, :]).astype(np.float32).reshape(-1)  # 4 次带 stride 的 Mul

    z = row_norm(z, 0.0)                    # 第一次 = softmax，分母不带 eps
    z = (z + eps_pad).astype(np.float32)    # x = softmax + eps，只加在有效位
    assert np.all(z[pad_lanes][: n * 16] == 0.0), "加 eps 污染了填充位"
    z = col_norm(z)
    for _ in range(repeat - 1):
        z = row_norm(z, eps)
        z = col_norm(z)

    # ---- CopyOut: 每个 block 取前 4 个 ----
    out = np.empty(n * 16, dtype=np.float32)
    for i in range(n * 4):
        out[i * 4: i * 4 + 4] = z[i * 8: i * 8 + 4]
    return out


def check(n, seed, kind="randn"):
    rng = np.random.default_rng(seed)
    if kind == "zeros":
        x = np.zeros((n, 4, 4), dtype=np.float32)
    elif kind == "big":
        x = (rng.standard_normal((n, 4, 4)) * 8).astype(np.float32)
    elif kind == "huge":
        x = (rng.standard_normal((n, 4, 4)) * 16).astype(np.float32)
    else:
        x = rng.standard_normal((n, 4, 4)).astype(np.float32)

    ref = reference(x)
    got = s1_kernel(x.reshape(-1).copy(), n).reshape(n, 4, 4)

    bad = bool(np.isnan(got).any() or np.isinf(got).any())
    d = np.abs(ref - got)
    tol = ATOL + RTOL * np.abs(ref)
    return d.max(), (d / tol).max(), bad


def main():
    print("=" * 78)
    print("S1 算法仿真  atol={} rtol={}".format(ATOL, RTOL))
    print("=" * 78)
    print("{:<30} {:>14} {:>12} {:>9} {:>7}".format(
        "用例", "最大绝对误差", "裕度占用", "NaN/Inf", "结果"))
    print("-" * 78)

    cases = ([("n={} ({})".format(n, k), n, k) for n, k in
              [(1, "randn"), (2, "randn"), (3, "randn"), (7, "randn"),
               (16, "randn"), (63, "randn"), (64, "randn"), (65, "randn"),
               (127, "randn"), (128, "randn"), (129, "randn"), (333, "randn"),
               (1024, "randn"), (64, "zeros"), (64, "big"), (64, "huge")]])

    fails = 0
    for name, n, kind in cases:
        wd = wh = 0.0
        bad = False
        for s in range(5):
            d, h, b = check(n, s, kind)
            wd, wh, bad = max(wd, d), max(wh, h), bad or b
        ok = (wh < 0.5) and not bad
        fails += 0 if ok else 1
        print("{:<30} {:>14.3e} {:>12.4f} {:>9} {:>7}".format(
            name, wd, wh, "有!" if bad else "无", "PASS" if ok else "FAIL"))

    print("-" * 78)
    print("结论: {}".format("全部通过（裕度占用均 < 0.5）" if fails == 0
                            else "{} 个用例失败".format(fails)))
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
