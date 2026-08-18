"""基因定义、合法化与邻域。

基因空间不是拍脑袋定的，是 P0/P1 实测出来的 —— 原方案里 20 位基因中有 5 位
已被实测证伪（见 EXPERIMENT_PLAN.md §2.3 与 §3 的更正），留下来的每一位都有
真机数据支撑其取值范围。
"""

import hashlib
import json
import random

# name -> (取值列表, 是否有序, 一句话说明)
GENES = {
    # ---- 已实测影响大 ----
    "tile_target":   ([8, 16, 32, 48, 64, 96, 128], True,
                      "每核矩阵数。实测 <64 明显更慢（block 启动成本主导）"),
    "eps_mode":      ([0, 1], False,
                      "0=eps 向量从 GM 读(+11.7us)  1=纯算术构造"),
    "tiling_mode":   ([0, 1], False,
                      "0=每次调用同步 memcpy 传 tiling(+75us)  1=静态缓存"),
    # ---- 影响精度 ----
    "div_mode":      ([0, 1], False,
                      "0=Reciprocal+NR+Mul  1=Div。硬件 Reciprocal 只有 9 位精度"),
    "nr_steps":      ([0, 1, 2, 3], True,
                      "Newton-Raphson 步数，仅 div_mode=0 时有效"),
    "use_max":       ([0, 1], False,
                      "softmax 是否减行最大值（大幅值输入的鲁棒性）"),
    # ---- 实测影响小，但要靠消融证明 ----
    "colnorm_wide":  ([0, 1], False, "ColNormalize 每 repeat 处理 1 还是 8 个矩阵"),
    "barrier_mode":  ([0, 1], False, "0=每条向量指令后加 PipeBarrier  1=全去掉"),
    "copyout_mode":  ([0, 1], False, "0=单次批量 DataCopyPad  1=每行一次"),
    "core_query":    ([0, 1], False, "0=每次查设备核数  1=首次后缓存"),
}

# 基因 -> CMake 选项名
CMAKE_OPT = {
    "tile_target": "SN_TILE_TARGET",
    "eps_mode": "SN_S1_EPS_MODE",
    "tiling_mode": "SN_TILING_MODE",
    "div_mode": "SN_S1_DIV_MODE",
    "nr_steps": "SN_S1_NR_STEPS",
    "use_max": "SN_S1_USE_MAX",
    "colnorm_wide": "SN_S1_COLNORM_WIDE",
    "barrier_mode": "SN_S1_BARRIER_MODE",
    "copyout_mode": "SN_S1_COPYOUT_MODE",
    "core_query": "SN_CORE_QUERY",
}

NAMES = list(GENES.keys())

# 已知的强个体，用作种群播种
SEEDS = {
    "s1d_arith": dict(tile_target=64, eps_mode=1, tiling_mode=1, div_mode=1,
                      nr_steps=0, use_max=1, colnorm_wide=1, barrier_mode=1,
                      copyout_mode=0, core_query=1),
    "s1c_gmeps": dict(tile_target=64, eps_mode=0, tiling_mode=1, div_mode=1,
                      nr_steps=0, use_max=1, colnorm_wide=1, barrier_mode=1,
                      copyout_mode=0, core_query=1),
    "s1_div":    dict(tile_target=64, eps_mode=0, tiling_mode=1, div_mode=1,
                      nr_steps=0, use_max=1, colnorm_wide=0, barrier_mode=0,
                      copyout_mode=0, core_query=1),
    "s1b_nr":    dict(tile_target=64, eps_mode=0, tiling_mode=1, div_mode=0,
                      nr_steps=2, use_max=1, colnorm_wide=1, barrier_mode=0,
                      copyout_mode=0, core_query=1),
}


def random_genome(rng=random):
    return {k: rng.choice(v[0]) for k, v in GENES.items()}


def repair(g):
    """把非法/冗余的组合投影到规范形式。

    关键：div_mode=1 时 nr_steps 完全不起作用，必须归一化成同一个值，
    否则 4 个基因型渲染出同一份代码、白白浪费 4 倍评估预算。
    """
    g = dict(g)
    for k, (vals, _, _) in GENES.items():
        if g.get(k) not in vals:
            g[k] = vals[0]
    if g["div_mode"] == 1:
        g["nr_steps"] = 0          # 无效基因，钉死
    return g


def key(g):
    """规范化后的哈希，用于去重缓存。"""
    g = repair(g)
    s = json.dumps({k: g[k] for k in NAMES}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def cmake_args(g):
    g = repair(g)
    args = ["-DSN_KERNEL_VARIANT=1"]
    args += ["-D{}={}".format(CMAKE_OPT[k], g[k]) for k in NAMES]
    return args


def label(g):
    g = repair(g)
    return "t{tile_target}_e{eps_mode}_d{div_mode}n{nr_steps}_m{use_max}_" \
           "w{colnorm_wide}_b{barrier_mode}_o{copyout_mode}_" \
           "T{tiling_mode}_c{core_query}".format(**g)


def mutate(g, rng, p=0.15):
    """有序基因走 ±1 邻域（80%），类别基因均匀重采样。"""
    g = dict(g)
    for k, (vals, ordered, _) in GENES.items():
        if rng.random() >= p:
            continue
        if ordered and rng.random() < 0.8:
            i = vals.index(g[k])
            i = max(0, min(len(vals) - 1, i + rng.choice([-1, 1])))
            g[k] = vals[i]
        else:
            g[k] = rng.choice(vals)
    return repair(g)


def crossover(a, b, rng, p=0.7):
    if rng.random() >= p:
        return dict(a), dict(b)
    c1, c2 = dict(a), dict(b)
    for k in NAMES:
        if rng.random() < 0.5:
            c1[k], c2[k] = c2[k], c1[k]
    return repair(c1), repair(c2)


def space_size():
    """去掉冗余组合后的有效空间大小。"""
    n = 1
    for k, (vals, _, _) in GENES.items():
        if k in ("div_mode", "nr_steps"):
            continue
        n *= len(vals)
    # div_mode=1 时 nr_steps 只有 1 种有效取值
    n *= 1 + len(GENES["nr_steps"][0])
    return n


if __name__ == "__main__":
    print("基因空间")
    print("=" * 78)
    for k, (vals, ordered, doc) in GENES.items():
        print("  {:<14} {:<28} {:<6} {}".format(
            k, str(vals), "有序" if ordered else "类别", doc))
    print("-" * 78)
    raw = 1
    for _, (v, _, _) in GENES.items():
        raw *= len(v)
    print("  笛卡尔积 {} 个，去掉 div_mode=1 时的冗余后有效空间 {} 个".format(
        raw, space_size()))
    print()
    rng = random.Random(0)
    print("随机个体示例与去重验证:")
    a = dict(random_genome(rng)); a["div_mode"] = 1; a["nr_steps"] = 0
    b = dict(a); b["nr_steps"] = 3
    print("  {}  ->  {}".format(label(a), key(a)))
    print("  {}  ->  {}".format(label(b), key(b)))
    print("  两者 nr_steps 不同但 div_mode=1 使其无效，哈希{}".format(
        "相同（去重生效）" if key(a) == key(b) else "不同（去重失效！）"))
