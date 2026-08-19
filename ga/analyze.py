"""GA 搜索结果分析。

产出：
  1. 总览（评估数、成功率、缓存命中、最优）
  2. 各 seed 的结果与稳定性
  3. 收敛曲线（best-so-far vs 真实评估次数）
  4. 最优个体的完整基因
  5. 逐基因分析：精确单基因消融 + 边际最优（兜底，总能算出来）
  6. 结局分布（编译/精度/运行失败各多少）
"""

import collections
import glob
import json
import os
import sqlite3
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genome as G  # noqa: E402


def load(db):
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT ts, ok, fitness, v1_us, genome, tag, headroom "
        "FROM evals ORDER BY ts").fetchall()
    con.close()
    curve, best = [], 0.0
    for r in rows:
        best = max(best, r[2] if r[1] else 0.0)
        curve.append(best)
    return rows, curve


def bar(frac, width=28):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ga/runs"
    dbs = sorted(glob.glob(os.path.join(root, "*.sqlite")))
    if not dbs:
        print("没找到结果: {}/*.sqlite".format(root))
        print("（未完成的运行会留下 .sqlite.partial，跑完才改名）")
        return 1

    runs = []          # (名字, rows, curve)
    all_rows = []
    for db in dbs:
        rows, curve = load(db)
        runs.append((os.path.basename(db).replace(".sqlite", ""), rows, curve))
        all_rows += rows

    ok_rows = [r for r in all_rows if r[1]]

    # ---------------- 1. 总览 ----------------
    print("=" * 84)
    print("总览")
    print("=" * 84)
    tags = collections.Counter(r[5] for r in all_rows)
    n = len(all_rows)
    print("  结果库          : {} 个 ({})".format(
        len(dbs), ", ".join(x[0] for x in runs)))
    print("  真实评估总数    : {}".format(n))
    print("  成功 / 成功率   : {} / {:.1%}".format(len(ok_rows), len(ok_rows) / max(1, n)))
    if ok_rows:
        best = max(ok_rows, key=lambda r: r[2])
        print("  全局最优 fitness: {:.4f}   v1={:.2f}us   裕度占用={:.5f}".format(
            best[2], best[3], best[6]))
        print("  全局最优基因组  : {}".format(G.label(json.loads(best[4]))))

    print("\n  各结局分布:")
    for t, c in tags.most_common():
        print("    {:<18} {:>4}  {:>6.1%}  {}".format(t, c, c / n, bar(c / n)))

    if not ok_rows:
        print("\n没有成功个体，后续分析跳过。")
        return 1

    # ---------------- 2. 各 seed ----------------
    print()
    print("=" * 84)
    print("各 seed 的结果（看 GA 稳不稳）")
    print("=" * 84)
    print("  {:<12} {:>8} {:>12} {:>12} {:>10}  {}".format(
        "运行", "评估数", "最优fitness", "最优v1(us)", "成功率", "最优基因组"))
    print("  " + "-" * 80)
    finals = []
    for name, rows, curve in runs:
        okr = [r for r in rows if r[1]]
        if not okr:
            print("  {:<12} {:>8} {:>12}".format(name, len(rows), "全部失败"))
            continue
        b = max(okr, key=lambda r: r[2])
        finals.append(b[2])
        print("  {:<12} {:>8} {:>12.4f} {:>12.2f} {:>9.0%}  {}".format(
            name, len(rows), b[2], b[3], len(okr) / len(rows),
            G.label(json.loads(b[4]))))
    if len(finals) > 1:
        print("  " + "-" * 80)
        print("  最优 fitness 跨 seed: mean={:.4f}  std={:.4f}  极差={:.4f} ({:.2%})".format(
            st.mean(finals), st.pstdev(finals), max(finals) - min(finals),
            (max(finals) - min(finals)) / st.mean(finals)))

    # ---------------- 3. 收敛曲线 ----------------
    print()
    print("=" * 84)
    print("收敛曲线（best-so-far vs 真实评估次数）")
    print("=" * 84)
    maxlen = max(len(c) for _, _, c in runs)
    marks = [m for m in (1, 4, 8, 12, 16, 20, 30, 40, 50, 60, 80, 100) if m <= maxlen]
    if maxlen not in marks:
        marks.append(maxlen)
    print("  {:<12} {}".format("评估次数", "".join("{:>8}".format(m) for m in marks)))
    print("  " + "-" * 80)
    for name, _, curve in runs:
        row = "".join("{:>8.3f}".format(curve[min(m, len(curve)) - 1]) for m in marks)
        print("  {:<12} {}".format(name, row))
    if len(runs) > 1:
        avg = []
        for m in marks:
            vals = [c[min(m, len(c)) - 1] for _, _, c in runs if c]
            avg.append("{:>8.3f}".format(st.mean(vals)))
        print("  " + "-" * 80)
        print("  {:<12} {}".format("平均", "".join(avg)))

    # ---------------- 4. 最优个体 ----------------
    bg = json.loads(best[4])
    print()
    print("=" * 84)
    print("最优个体")
    print("=" * 84)
    print("  {:<16} {:<8} {}".format("基因", "取值", "CMake 选项"))
    print("  " + "-" * 60)
    for k in G.NAMES:
        print("  {:<16} {:<8} -D{}={}".format(k, bg[k], G.CMAKE_OPT[k], bg[k]))
    print("\n  fitness={:.4f}  v1={:.2f}us  裕度占用={:.5f}".format(
        best[2], best[3], best[6]))

    # ---------------- 5. 逐基因分析 ----------------
    print()
    print("=" * 84)
    print("逐基因消融（固定最优个体，只改一位）")
    print("=" * 84)
    exact = []
    for k in G.NAMES:
        alt = [r[2] for r in ok_rows
               if (g := json.loads(r[4]))[k] != bg[k]
               and all(g[o] == bg[o] for o in G.NAMES if o != k)]
        if alt:
            exact.append((best[2] - max(alt), k, bg[k], max(alt), len(alt)))
    if exact:
        print("  {:<16} {:<8} {:>12} {:>12} {:>8}".format(
            "基因", "最优取值", "改掉后最好", "跌幅", "样本"))
        print("  " + "-" * 60)
        for drop, k, v, mx, cnt in sorted(exact, reverse=True):
            print("  {:<16} {:<8} {:>12.4f} {:>12.4f} {:>8}".format(k, v, mx, drop, cnt))
    else:
        print("  缓存里没有「只差一位」的邻居个体，无法做精确消融。")
        print("  改看下面的边际最优。")

    print()
    print("=" * 84)
    print("边际最优（每个取值下的最好 fitness，总能算出来）")
    print("=" * 84)
    for k in G.NAMES:
        vals = G.GENES[k][0]
        cells = []
        for v in vals:
            fs = [r[2] for r in ok_rows if json.loads(r[4])[k] == v]
            cells.append((v, max(fs) if fs else None, len(fs)))
        top = max((c[1] for c in cells if c[1] is not None), default=0)
        line = []
        for v, f, cnt in cells:
            if f is None:
                line.append("{}:  -  ".format(v))
            else:
                mark = "*" if abs(f - top) < 1e-9 else " "
                line.append("{}:{:.3f}{}({})".format(v, f, mark, cnt))
        print("  {:<16} {}".format(k, "  ".join(line)))
    print("\n  格式  取值:最好fitness(该取值的成功样本数)   * = 该基因下的最优")
    print("  注意：边际最优受其它基因干扰，不能替代精确消融，只作参考。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
