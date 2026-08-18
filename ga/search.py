"""搜索算法：岛屿模型 GA、随机搜索、局部搜索、TPE(Optuna)。

岛屿模型按 tile_target 分岛 —— 实测 tile<64 有明显台阶，是空间里最强的结构断层，
按它分岛能避免小 tile 个体在早期被大 tile 个体直接淘汰、维持多样性。
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genome as G      # noqa: E402
import cache as C       # noqa: E402
import evaluate as E    # noqa: E402


class Budget:
    """按「真实评估次数」计预算，缓存命中不计费——这才是公平的算法对比口径。"""

    def __init__(self, n, cache):
        self.n = n
        self.cache = cache
        self.used = 0
        self.curve = []          # (真实评估次数, 至今最优 fitness)
        self.best = 0.0
        self.best_g = None
        self.wall = []

    def spend(self, g, res):
        if not res.get("cached"):
            self.used += 1
        if res["fitness"] > self.best:
            self.best = res["fitness"]
            self.best_g = G.repair(g)
        self.curve.append((self.used, self.best))
        return self.used < self.n

    def left(self):
        return self.used < self.n


def _ev(g, bud, backend, rng, py, algo, gen):
    t0 = time.time()
    res = E.evaluate(g, bud.cache, backend, rng, py, algo, gen)
    dt = time.time() - t0
    prev = bud.used
    bud.spend(g, res)
    if bud.used != prev:                       # 真实评估才打印（缓存命中不打）
        bud.wall.append(dt)
        avg = sum(bud.wall) / len(bud.wall)
        eta = avg * max(0, bud.n - bud.used)
        detail = ("fit={:.4f} v1={:.1f}us".format(res["fitness"], res["v1_us"])
                  if res["ok"] else res["tag"])
        print("  [{:>3}/{}] {:>5.0f}s  {:<32} {:<26} best={:.4f}  剩余≈{:.0f}分钟".format(
            bud.used, bud.n, dt, G.label(g), detail, bud.best, eta / 60),
            flush=True)
    return res


# ---------------------------------------------------------------- 随机搜索
def random_search(bud, backend, rng, py):
    while bud.left():
        _ev(G.random_genome(rng), bud, backend, rng, py, "random", 0)


# ---------------------------------------------------------------- 局部搜索
def local_search(bud, backend, rng, py):
    cur = G.SEEDS["s1_div"]
    curf = _ev(cur, bud, backend, rng, py, "local", 0)["fitness"]
    while bud.left():
        improved = False
        for k, (vals, _, _) in G.GENES.items():
            for v in vals:
                if not bud.left():
                    return
                if v == cur[k]:
                    continue
                cand = G.repair(dict(cur, **{k: v}))
                f = _ev(cand, bud, backend, rng, py, "local", 0)["fitness"]
                if f > curf:
                    cur, curf, improved = cand, f, True
        if not improved:                     # 到局部最优就随机重启
            cur = G.random_genome(rng)
            curf = _ev(cur, bud, backend, rng, py, "local", 0)["fitness"]


# ---------------------------------------------------------------- 岛屿 GA
def island_ga(bud, backend, rng, py, pop_per_island=6, elite=1,
              p_cx=0.7, p_mut=0.15, migrate_every=5):
    tiles = G.GENES["tile_target"][0]
    n_isl = min(4, len(tiles))
    groups = [tiles[i::n_isl] for i in range(n_isl)]

    # 播种：已知强个体 + 随机
    seeds = list(G.SEEDS.values())
    islands = []
    for gi, grp in enumerate(groups):
        pop = []
        for s in seeds:
            if s["tile_target"] in grp and len(pop) < pop_per_island // 2:
                pop.append(G.repair(dict(s)))
        while len(pop) < pop_per_island:
            g = G.random_genome(rng)
            g["tile_target"] = rng.choice(grp)
            pop.append(G.repair(g))
        islands.append(pop)

    fit = [[0.0] * pop_per_island for _ in range(n_isl)]
    gen = 0
    stall = 0
    best_seen = 0.0
    while bud.left():
        for i, pop in enumerate(islands):
            for j, g in enumerate(pop):
                if not bud.left():
                    return
                fit[i][j] = _ev(g, bud, backend, rng, py, "ga", gen)["fitness"]

        if bud.best > best_seen + 1e-9:
            best_seen, stall = bud.best, 0
        else:
            stall += 1
        pm = p_mut * (2.0 if stall >= 5 else 1.0)     # 自适应变异率

        for i in range(n_isl):
            order = sorted(range(pop_per_island), key=lambda j: -fit[i][j])
            new = [islands[i][j] for j in order[:elite]]      # 精英保留
            while len(new) < pop_per_island:
                pa = max(rng.sample(range(pop_per_island), 3), key=lambda j: fit[i][j])
                pb = max(rng.sample(range(pop_per_island), 3), key=lambda j: fit[i][j])
                c1, c2 = G.crossover(islands[i][pa], islands[i][pb], rng, p_cx)
                new.append(G.mutate(c1, rng, pm))
                if len(new) < pop_per_island:
                    new.append(G.mutate(c2, rng, pm))
            islands[i] = new

        gen += 1
        if gen % migrate_every == 0 and n_isl > 1:        # 迁移精英
            elites = [max(range(pop_per_island), key=lambda j: fit[i][j])
                      for i in range(n_isl)]
            for i in range(n_isl):
                src = (i - 1) % n_isl
                islands[i][-1] = G.repair(dict(islands[src][elites[src]]))
        if stall >= 10:                                    # 重启最差一半
            for i in range(n_isl):
                for j in range(pop_per_island // 2, pop_per_island):
                    islands[i][j] = G.random_genome(rng)
            stall = 0


# ---------------------------------------------------------------- TPE
def tpe_search(bud, backend, rng, py):
    try:
        import optuna
    except ImportError:
        print("未安装 optuna，跳过 TPE（pip install optuna）")
        return
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        g = {k: trial.suggest_categorical(k, [str(x) for x in v[0]])
             for k, v in G.GENES.items()}
        g = G.repair({k: int(v) for k, v in g.items()})
        if not bud.left():
            trial.study.stop()
        return _ev(g, bud, backend, rng, py, "tpe", 0)["fitness"]

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=rng.randint(0, 10**6)))
    st.optimize(obj, n_trials=bud.n * 3, catch=(Exception,))


ALGOS = {"ga": island_ga, "random": random_search,
         "local": local_search, "tpe": tpe_search}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ga", choices=list(ALGOS))
    ap.add_argument("--backend", default="mock", choices=["mock", "real"])
    ap.add_argument("--budget", type=int, default=200, help="真实评估次数上限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    db = args.db or "ga/{}_{}_{}.sqlite".format(args.algo, args.backend, args.seed)
    partial = db + ".partial"
    for f in (db, partial):
        if os.path.exists(f):
            os.remove(f)
    cache = C.Cache(partial)
    rng = random.Random(args.seed)
    bud = Budget(args.budget, cache)

    est = 51 if args.backend == "real" else 0.001
    print("算法={}  后端={}  seed={}  预算={} 次真实评估".format(
        args.algo, args.backend, args.seed, args.budget))
    if args.backend == "real":
        print("单次评估约 {}s（cmake configure + build + 精度门禁 + 计时），"
              "预计 {:.0f} 分钟".format(est, args.budget * est / 60))
    print("进度（缓存命中不计入预算，也不打印）:", flush=True)

    t0 = time.time()
    try:
        ALGOS[args.algo](bud, args.backend, rng, args.python)
    except KeyboardInterrupt:
        print("\n已中断，保留部分结果于 {}".format(partial), flush=True)
    wall = time.time() - t0

    st = cache.stats()
    print()
    print("=" * 78)
    print("算法={}  后端={}  seed={}  预算={} 次真实评估".format(
        args.algo, args.backend, args.seed, args.budget))
    print("=" * 78)
    print("  最优 fitness      : {:.4f}".format(bud.best))
    print("  最优 v1           : {:.2f} us".format(
        cache.best(1)[0][1] if cache.best(1) else 0))
    print("  最优基因组        : {}".format(G.label(bud.best_g) if bud.best_g else "-"))
    print("  真实评估 / 缓存命中: {} / {}  (命中率 {:.0%})".format(
        st["misses"], st["hits"], st["hits"] / max(1, st["hits"] + st["misses"])))
    print("  各结局            : {}".format(st["by_tag"]))
    print("  耗时              : {:.1f}s  ({:.2f}s/次)".format(
        wall, wall / max(1, st["misses"])))
    cache.con.close()
    # 跑完才改成正式名字：中断留下的 .partial 不会让 run_all.sh 误判为已完成
    os.rename(partial, db)
    print("  收敛曲线已存入     : {}".format(db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
