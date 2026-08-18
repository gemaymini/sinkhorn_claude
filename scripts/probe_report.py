"""解读 probe_layout.bin，判定几个 AscendC API 的确切行为。"""

import os
import sys
import numpy as np

SLOT = 64
NAMES = [
    "S0  DataCopyPad blockLen=4B  srcStride=60B  (抽平面)",
    "S1  DataCopyPad blockLen=16B srcStride=48B  (抽行)",
    "S2  DataCopyPad blockCount=4 blockLen=16B rightPad=4 (展开)",
    "S3  BlockReduceSum(0..63)",
    "S4  Brcb(0..7)",
    "S5  PairReduceSum(0..63)",
    "S6  Reciprocal(1..64)",
    "S7  (未使用)",
]
UNSET = -999.0


def fmt(a, n=64, per=8):
    out = []
    for i in range(0, min(n, len(a)), per):
        chunk = " ".join(
            "{:>9}".format("·" if abs(v - UNSET) < 1e-3 else "{:g}".format(round(float(v), 4)))
            for v in a[i:i + per])
        out.append("    [{:>2}] {}".format(i, chunk))
    return "\n".join(out)


def near(a, b, tol=1e-3):
    return abs(float(a) - float(b)) < tol


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "probe_layout.bin"
    if not os.path.isfile(path):
        print("找不到 {}，请先运行 ./probe_sinkhorn".format(path))
        return 1
    d = np.fromfile(path, dtype=np.float32)
    if d.size < SLOT * 7:
        print("文件长度不对: {} floats".format(d.size))
        return 1

    verdicts = {}
    for i, name in enumerate(NAMES):
        s = d[i * SLOT:(i + 1) * SLOT]
        print("\n" + "=" * 74)
        print(name)
        print("=" * 74)
        print(fmt(s))

        if i == 0:
            want = [0, 16, 32, 48, 64, 80, 96, 112]
            packed = all(near(s[j], want[j]) for j in range(8))
            padded = all(near(s[8 * j], want[j]) for j in range(8))
            if packed:
                verdicts["S0"] = "PACKED"
                print("\n  >>> 紧凑打包：8 个平面元素连续落在 [0..8)")
                print("      => S1 可以直接用 count-based 的 Add/Mul，向量利用率 100%")
            elif padded:
                verdicts["S0"] = "PADDED32"
                print("\n  >>> 每块补齐到 32B：元素落在 0,8,16,... 步长 8")
                print("      => S1 必须用 level-0 API 带 blkStride，或改走别的路线")
            else:
                verdicts["S0"] = "UNKNOWN"
                print("\n  >>> 两种假设都不符，需人工看上面的实际布局")

        elif i == 1:
            packed = all(near(s[j], v) for j, v in enumerate([0, 1, 2, 3, 16, 17, 18, 19]))
            padded = (all(near(s[j], j) for j in range(4))
                      and all(near(s[8 + j], 16 + j) for j in range(4)))
            verdicts["S1"] = "PACKED" if packed else ("PADDED32" if padded else "UNKNOWN")
            print("\n  >>> {}".format(verdicts["S1"]))

        elif i == 2:
            want = []
            for r in range(4):
                want += [4 * r, 4 * r + 1, 4 * r + 2, 4 * r + 3, 0, 0, 0, 0]
            ok = all(near(s[j], want[j]) for j in range(32))
            verdicts["S2"] = "OK" if ok else "UNKNOWN"
            print("\n  >>> 一次调用展开 4 行: {}".format(
                "成立（CopyIn 可以从 4 次/矩阵降到 1 次/tile）" if ok else "不符合预期"))

        elif i == 3:
            want = [64 * b + 28 for b in range(8)]
            contig = all(near(s[j], want[j]) for j in range(8))
            stride8 = all(near(s[8 * j], want[j]) for j in range(8))
            verdicts["S3"] = ("CONTIGUOUS" if contig else
                              ("STRIDE8" if stride8 else "UNKNOWN"))
            print("\n  >>> 期望 8 个和 {} -> {}".format(want, verdicts["S3"]))

        elif i == 4:
            ok = all(near(s[8 * b + l], b) for b in range(8) for l in range(8))
            verdicts["S4"] = "OK" if ok else "UNKNOWN"
            print("\n  >>> Brcb 每 block 广播: {}".format(
                "符合预期（block j 全为 j）" if ok else "不符合预期"))

        elif i == 5:
            want = [4 * j + 1 for j in range(32)]
            contig = all(near(s[j], want[j]) for j in range(32))
            verdicts["S5"] = "CONTIGUOUS" if contig else "UNKNOWN"
            print("\n  >>> 期望 {}... -> {}".format(want[:6], verdicts["S5"]))

        elif i == 6:
            # 量化 Reciprocal 的实际有效位数（硬件通常是快速近似，不是 fp32 精度）
            import math
            errs = []
            for k in range(1, 17):
                exact = 1.0 / k
                errs.append(abs(float(s[k - 1]) - exact) / exact)
            rel = max(errs)
            bits = -math.log2(rel) if rel > 0 else 24.0
            verdicts["S6"] = "{:.0f}bit".format(bits)
            print("\n  >>> Reciprocal 最大相对误差 {:.2e}  =>  约 {:.0f} 位有效精度".format(rel, bits))
            if bits < 20:
                print("      这是快速近似指令，不是 fp32 精度。直接用会让 20 次连续归一化后")
                print("      的误差到 1e-3 量级。对策：Newton-Raphson 修正，或直接用 Div。")

    print("\n" + "=" * 74)
    print("判定汇总")
    print("=" * 74)
    for k in ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]:
        print("  {:<4} {}".format(k, verdicts.get(k, "-")))

    print("\n" + "-" * 74)
    print("对 S1 kernel 设计的影响")
    print("-" * 74)
    v0 = verdicts.get("S0")
    if v0 == "PACKED":
        print("  ✅ 走 SoA 平面路线：16 次 strided DataCopyPad 抽出 16 个连续平面，")
        print("     之后行和/列和全部退化成 count-based 的 Add，10 次迭代 640 条满载指令。")
    elif v0 == "PADDED32":
        print("  ⚠ 平面元素步长为 8，count-based API 不可用。两条备选：")
        print("     (a) 用 level-0 API 带 blkStride=1/repStride=8，每 repeat 只用 8 lane（8x 浪费）")
        print("     (b) 若 S3=CONTIGUOUS 且 S4=OK，改走 padded-AoS + BlockReduceSum + Brcb 批量路线")
    else:
        print("  需要人工看 S0 的实际布局再定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
