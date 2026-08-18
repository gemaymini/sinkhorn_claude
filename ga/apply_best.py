"""把 GA 的搜索结果落到 CMakeLists.txt 的默认值上。

搜索阶段用的是**快档**保真度（精度 2 seed、计时 warmup 60/repeat 180），排名可信但
数值不可直接采信。本脚本：

  1. 扫描所有结果库，按快档 fitness 取 top-K 候选
  2. 把候选**和当前 CMakeLists 默认值（卫冕者）**一起用**全档**重新评测
  3. 只有当挑战者以超过噪声阈值的幅度胜出时才建议替换

最后一条很重要：v1 的测量 CV 约 2%，若不设阈值，采纳一个噪声赢家反而会让
提交件比已验证过的配置更差。

用法:
    python ga/apply_best.py                      # 只看不改
    python ga/apply_best.py --apply              # 确认后写入 CMakeLists.txt
    python ga/apply_best.py --topk 5 --repeats 3
"""

import argparse
import glob
import json
import os
import random
import re
import sqlite3
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genome as G      # noqa: E402
import cache as C       # noqa: E402
import evaluate as E    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAKE = os.path.join(ROOT, "CMakeLists.txt")
MARGIN = 0.015          # 挑战者需超出卫冕者 1.5%（≈ 测量 CV）才算真赢


def read_cmake_defaults():
    txt = open(CMAKE, encoding="utf-8").read()
    g = {}
    for k, opt in G.CMAKE_OPT.items():
        m = re.search(r"^set\(" + opt + r"\s+(\S+)", txt, re.M)
        if not m:
            return None
        g[k] = int(m.group(1))
    return G.repair(g)


def write_cmake_defaults(g):
    txt = open(CMAKE, encoding="utf-8").read()
    for k, opt in G.CMAKE_OPT.items():
        txt = re.sub(r"^(set\(" + opt + r"\s+)\S+", r"\g<1>" + str(g[k]), txt,
                     count=1, flags=re.M)
    open(CMAKE, "w", encoding="utf-8").write(txt)


def collect(dbs):
    """从所有结果库里取成功个体，按快档 fitness 去重排序。"""
    best = {}
    for db in dbs:
        con = sqlite3.connect(db)
        for f, gj in con.execute(
                "SELECT fitness, genome FROM evals WHERE ok=1"):
            g = G.repair(json.loads(gj))
            k = G.key(g)
            if k not in best or f > best[k][0]:
                best[k] = (f, g)
        con.close()
    return sorted(best.values(), key=lambda x: -x[0])


def full_eval(g, py, repeats):
    """全档评测，跑 repeats 个独立进程取中位数。"""
    vs = []
    for i in range(repeats):
        r = E.real_eval(g, py, "build_verify_{}_{}".format(G.key(g), i), fast=False)
        os.system("rm -rf {}".format(
            os.path.join(ROOT, "build_verify_{}_{}".format(G.key(g), i))))
        if not r["ok"]:
            return None, r["tag"]
        vs.append(r["v1_us"])
    return st.median(vs), "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="ga/runs", help="结果库目录")
    ap.add_argument("--topk", type=int, default=3, help="全档复验前几名")
    ap.add_argument("--repeats", type=int, default=3, help="每个候选跑几个独立进程")
    ap.add_argument("--apply", action="store_true", help="确认后写入 CMakeLists.txt")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry", action="store_true", help="不评测，只列候选")
    args = ap.parse_args()

    incumbent = read_cmake_defaults()
    if incumbent is None:
        print("无法从 CMakeLists.txt 读出当前默认值")
        return 1

    dbs = sorted(glob.glob(os.path.join(ROOT, args.runs, "*.sqlite")))
    print("=" * 80)
    print("当前 CMakeLists 默认值（卫冕者）")
    print("=" * 80)
    print("  {}".format(G.label(incumbent)))
    if not dbs:
        print("\n没有找到 GA 结果库（{}/*.sqlite）——".format(args.runs))
        print("说明 GA 还没跑，直接用当前默认值打包即可。")
        return 0

    cands = collect(dbs)
    print("\n{} 个结果库，去重后 {} 个成功个体".format(len(dbs), len(cands)))
    print("\n快档 fitness 排名前 {}：".format(args.topk))
    print("  {:<6} {:>10}  {}".format("排名", "快档fit", "基因组"))
    print("  " + "-" * 74)
    shortlist = []
    for i, (f, g) in enumerate(cands[:args.topk], 1):
        same = " (= 当前默认值)" if G.key(g) == G.key(incumbent) else ""
        print("  {:<6} {:>10.4f}  {}{}".format(i, f, G.label(g), same))
        shortlist.append(g)
    if all(G.key(g) != G.key(incumbent) for g in shortlist):
        shortlist.append(incumbent)

    if args.dry:
        return 0

    print("\n" + "=" * 80)
    print("全档复验（精度 5 seed + warmup200/repeat500，各 {} 个独立进程取中位数）".format(
        args.repeats))
    print("=" * 80)
    print("  {:<52} {:>12} {:>10}".format("基因组", "v1 (us)", "结果"))
    print("  " + "-" * 76)
    scores = []
    for g in shortlist:
        v1, tag = full_eval(g, args.python, args.repeats)
        mark = " ← 卫冕者" if G.key(g) == G.key(incumbent) else ""
        print("  {:<52} {:>12} {:>10}{}".format(
            G.label(g), "{:.2f}".format(v1) if v1 else "-", tag, mark))
        if v1:
            scores.append((v1, g))

    if not scores:
        print("\n全部候选复验失败，保持当前默认值。")
        return 1

    scores.sort()
    best_v1, best_g = scores[0]
    inc = next((v for v, g in scores if G.key(g) == G.key(incumbent)), None)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    if inc is None:
        print("  卫冕者复验失败，改用 {}".format(G.label(best_g)))
        adopt = True
    elif G.key(best_g) == G.key(incumbent):
        print("  当前默认值仍是最优（v1={:.2f}us），无需改动。".format(inc))
        adopt = False
    else:
        gain = (inc - best_v1) / inc
        print("  挑战者 {}".format(G.label(best_g)))
        print("    v1 {:.2f}us  vs  卫冕者 {:.2f}us   领先 {:.2%}".format(
            best_v1, inc, gain))
        if gain > MARGIN:
            print("  超过噪声阈值 {:.1%}，建议替换。".format(MARGIN))
            adopt = True
        else:
            print("  未超过噪声阈值 {:.1%}（v1 的测量 CV 约 2%）——".format(MARGIN))
            print("  采纳噪声赢家会让提交件比已验证的配置更差，保持当前默认值。")
            adopt = False

    if adopt and args.apply:
        write_cmake_defaults(best_g)
        print("\n  已写入 CMakeLists.txt。请重新打包：")
        print("    bash submission/package.sh \"队伍名\" \"UID\"")
    elif adopt:
        print("\n  加 --apply 才会真正写入 CMakeLists.txt。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
