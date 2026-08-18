"""对照实验分析：收敛曲线、算法比较、逐基因消融。"""

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
        "SELECT ts, ok, fitness, v1_us, genome, tag FROM evals ORDER BY ts").fetchall()
    con.close()
    curve, best = [], 0.0
    for i, (_, ok, f, _, _, _) in enumerate(rows, 1):
        best = max(best, f if ok else 0.0)
        curve.append(best)
    return rows, curve


def mannwhitney(a, b):
    """小样本 Mann-Whitney U 的 U 统计量与效应量（不查表，只报数值）。"""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.5
    u = sum(1 for x in a for y in b if x > y) + 0.5 * sum(
        1 for x in a for y in b if x == y)
    return u, u / (n1 * n2)          # 后者是"a 优于 b"的概率，0.5 表示无差异


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ga/runs"
    dbs = sorted(glob.glob(os.path.join(root, "*.sqlite")))
    if not dbs:
        print("没找到结果: {}/*.sqlite".format(root))
        return 1

    by_algo = collections.defaultdict(list)
    all_rows = []
    for db in dbs:
        algo = os.path.basename(db).split("_s")[0]
        rows, curve = load(db)
        by_algo[algo].append((curve, rows))
        all_rows += rows

    # ---------------- 最终结果 ----------------
    print("=" * 82)
    print("算法比较（同预算，按真实评估次数计）")
    print("=" * 82)
    print("{:<10} {:>8} {:>22} {:>14} {:>12}".format(
        "算法", "seed数", "最优 fitness (mean±std)", "最优 v1(us)", "AUC"))
    print("-" * 82)
    finals = {}
    for algo in ("ga", "tpe", "local", "random"):
        if algo not in by_algo:
            continue
        bests = [c[-1] for c, _ in by_algo[algo] if c]
        aucs = [sum(c) / len(c) for c, _ in by_algo[algo] if c]
        if not bests:
            print("{:<10} {:>8} {:>22}".format(algo, 0, "无数据（该算法未跑或全部失败）"))
            continue
        v1s = []
        for _, rows in by_algo[algo]:
            ok = [r for r in rows if r[1]]
            if ok:
                v1s.append(min(r[3] for r in ok))
        finals[algo] = bests
        print("{:<10} {:>8} {:>13.4f} ±{:<7.4f} {:>14.2f} {:>12.4f}".format(
            algo, len(bests), st.mean(bests),
            st.pstdev(bests) if len(bests) > 1 else 0.0,
            min(v1s) if v1s else 0.0, st.mean(aucs)))

    # ---------------- 显著性 ----------------
    if "ga" in finals:
        print("\n{:<28} {:>8} {:>28}".format("GA vs", "U", "P(GA 更优)"))
        print("-" * 66)
        for other in ("tpe", "local", "random"):
            if other in finals:
                u, p = mannwhitney(finals["ga"], finals[other])
                print("{:<28} {:>8.1f} {:>28.2f}".format(other, u, p))
        print("(n=3 太小，无法给出可靠 p 值；这里报效应量本身)")

    # ---------------- 收敛曲线 ----------------
    print("\n" + "=" * 82)
    print("收敛曲线（best-so-far，各 seed 平均）")
    print("=" * 82)
    maxlen = max(len(c) for cs in by_algo.values() for c, _ in cs)
    marks = [m for m in (1, 5, 10, 20, 30, 40, 50, 60, 80, 100) if m <= maxlen]
    print("{:<10} {}".format("评估次数", "".join("{:>9}".format(m) for m in marks)))
    print("-" * 82)
    for algo in ("ga", "tpe", "local", "random"):
        if algo not in by_algo:
            continue
        row = []
        for m in marks:
            vals = [c[min(m, len(c)) - 1] for c, _ in by_algo[algo] if c]
            row.append("{:>9.3f}".format(st.mean(vals)) if vals else "{:>9}".format("-"))
        print("{:<10} {}".format(algo, "".join(row)))

    # ---------------- 健康度 ----------------
    tags = collections.Counter(r[5] for r in all_rows)
    print("\n各结局: {}".format(dict(tags)))

    # ---------------- 逐基因消融 ----------------
    ok_rows = [r for r in all_rows if r[1]]
    if not ok_rows:
        print("\n没有成功个体，跳过消融")
        return 0
    best = max(ok_rows, key=lambda r: r[2])
    bg = json.loads(best[4])
    print("\n" + "=" * 82)
    print("逐基因消融：固定最优个体，逐位换成其它取值后的最好 fitness")
    print("最优个体 {}   fitness={:.4f}  v1={:.2f}us".format(
        G.label(bg), best[2], best[3]))
    print("=" * 82)
    print("{:<14} {:<8} {:>12} {:>12} {:>10}".format(
        "基因", "最优取值", "换掉后最好", "跌幅", "样本数"))
    print("-" * 82)
    idx = collections.defaultdict(list)
    for r in ok_rows:
        idx[json.dumps(json.loads(r[4]), sort_keys=True)].append(r[2])
    rank = []
    for k in G.NAMES:
        alt = []
        for r in ok_rows:
            g = json.loads(r[4])
            if g[k] != bg[k] and all(g[o] == bg[o] for o in G.NAMES if o != k):
                alt.append(r[2])
        if alt:
            drop = best[2] - max(alt)
            rank.append((drop, k, bg[k], max(alt), len(alt)))
    for drop, k, v, mx, n in sorted(rank, reverse=True):
        print("{:<14} {:<8} {:>12.4f} {:>12.4f} {:>10}".format(k, v, mx, drop, n))
    if not rank:
        print("（缓存里还没有只差一位的邻居个体，跑完完整预算后会有）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
