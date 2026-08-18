"""S1 精度门禁：多形状 × 多 seed，用**官方容差**判定，并报告裕度占用。

比 test_torch.py 更严格的地方：
  - 覆盖奇数矩阵数、单矩阵、超过一个 tile 的规模，专门打 S1 的尾块路径
  - 报告裕度占用 = max|diff| / (atol + rtol*|ref|)，要求 < 0.5 留一半安全边际
  - 检查 NaN / Inf（S1 里 ColNormalize 对填充列做了 1/(0+1e-30) 的处理，必须验证没有溢出）

用法: python scripts/check_shapes.py --so build_s1/libsinkhorn_normalize_ops.so
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401

ATOL = 1e-2
RTOL = 1e-2
HEADROOM_LIMIT = 0.5


def reference(x, repeat=10, eps=1e-6):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


CASES = [
    ("参考形状 [1,1024,4,4]", (1, 1024, 4, 4), "randn"),
    ("单矩阵 [4,4]", (4, 4), "randn"),
    ("奇数个 [1,7,4,4]  ← 尾块", (1, 7, 4, 4), "randn"),
    ("刚过一个 tile [1,65,4,4] ← 尾块", (1, 65, 4, 4), "randn"),
    ("刚过两个 tile [1,129,4,4] ← 尾块", (1, 129, 4, 4), "randn"),
    ("小批 [1,16,4,4]", (1, 16, 4, 4), "randn"),
    ("多维 [4,256,4,4]", (4, 256, 4, 4), "randn"),
    ("非 2 幂 [3,333,4,4] ← 尾块", (3, 333, 4, 4), "randn"),
    ("全零输入 [1,64,4,4]", (1, 64, 4, 4), "zeros"),
    ("均匀 [0,1) [1,64,4,4]", (1, 64, 4, 4), "rand"),
    ("大幅值 [1,64,4,4] ×8", (1, 64, 4, 4), "big"),
]


def make(kind, shape, seed):
    torch.manual_seed(seed)
    if kind == "zeros":
        return torch.zeros(shape, dtype=torch.float32)
    if kind == "rand":
        return torch.rand(shape, dtype=torch.float32)
    if kind == "big":
        return torch.randn(shape, dtype=torch.float32) * 8.0
    return torch.randn(shape, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    torch.ops.load_library(args.so)
    op = torch.ops.npu.sinkhorn_normalize

    print("=" * 84)
    print("S1 精度门禁  atol={} rtol={}  裕度上限={}  seeds={}".format(
        ATOL, RTOL, HEADROOM_LIMIT, args.seeds))
    print("=" * 84)
    print("{:<36} {:>8} {:>13} {:>10} {:>8}".format(
        "用例", "结果", "最大绝对误差", "裕度占用", "NaN/Inf"))
    print("-" * 84)

    failures = 0
    for name, shape, kind in CASES:
        worst_d, worst_h, bad, ok_all = 0.0, 0.0, False, True
        for s in range(args.seeds):
            x = make(kind, shape, s)
            ref = reference(x)
            try:
                got = op(x.npu(), 10, 1e-6).cpu()
            except Exception as e:                       # noqa: BLE001
                print("{:<36} {:>8}  {}".format(name, "异常", e))
                failures += 1
                ok_all = False
                break
            if torch.isnan(got).any() or torch.isinf(got).any():
                bad = True
            d = (ref - got).abs()
            tol = ATOL + RTOL * ref.abs()
            worst_d = max(worst_d, d.max().item())
            worst_h = max(worst_h, (d / tol).max().item())
            if not torch.allclose(ref, got, atol=ATOL, rtol=RTOL, equal_nan=False):
                ok_all = False
        else:
            passed = ok_all and not bad and worst_h < HEADROOM_LIMIT
            if not passed:
                failures += 1
            print("{:<36} {:>8} {:>13.3e} {:>10.4f} {:>8}".format(
                name, "PASS" if passed else "FAIL", worst_d, worst_h,
                "有!" if bad else "无"))

    print("-" * 84)
    if failures:
        print("结论: {} 个用例未通过".format(failures))
    else:
        print("结论: 全部通过，且裕度占用均 < {}".format(HEADROOM_LIMIT))
    print("=" * 84)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
