"""个体评估：基因组 -> 编译 -> 精度门禁 -> 官方口径计时 -> fitness。

两个后端：
  real  真机。每个个体独立 build 目录 + 独立进程，超时即杀。
  mock  Mac 上可跑的代价模型，参数**全部来自真机实测**（见下方注释），
        用来在没有 NPU 时验证 GA 主循环本身。注意：mock 的最优点就是已知的
        s1d_arith，它只能验证机制，不能用来发现新配置。

fitness = BASELINE_US / (v1_us + LAMBDA * IQR)
  baseline 用 20 次实测的中位数钉死成常量 —— 实测跨进程漂移 ±11%，
  不钉死会让 2% 的真实差异淹没在噪声里。
"""

import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genome as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_US = 1563.0        # 20 次实测中位数，见 EXPERIMENT_PLAN §2.3
LAMBDA = 0.5
BUILD_TIMEOUT = 300
RUN_TIMEOUT = 240
HEADROOM_LIMIT = 0.5


# --------------------------------------------------------------------------
# mock 代价模型（参数来自真机实测）
# --------------------------------------------------------------------------
def mock_eval(g, rng):
    g = G.repair(g)
    a = 62.0                                     # launch+dispatch+sync 地板 M5
    a += 27.0 * (1.0 if g["copyout_mode"] == 0 else 1.2)   # CopyIn+CopyOut
    a += 11.7 if g["eps_mode"] == 0 else 0.3     # BuildEpsVector 实测差值
    a += 3.0 if g["use_max"] else 1.0            # Exp(+减max)
    if g["tile_target"] < 64:
        a += 23.0                                # 实测 block 数 >16 后的台阶
    if g["tiling_mode"] == 0:
        a += 75.0                                # P0 实测同步 memcpy
    if g["core_query"] == 0:
        a += 3.0

    b = 0.83 if g["colnorm_wide"] == 0 else 0.60  # 每次迭代，实测拟合值
    if g["barrier_mode"] == 1:
        b *= 0.75
    if g["div_mode"] == 0:
        b += 0.08 * g["nr_steps"]

    m1 = a + 10.0 * b
    # 62 = M1->v1 的经验偏移(43us) + mock 对 M1 的系统性低估(19us)，
    # 标定到 s1d_arith 的真机 v1=158.55us
    v1 = (m1 + 62.0) * (1.0 + rng.gauss(0, 0.015))

    # 精度门禁含 randn×32 用例：免减 max 时 exp(112)≈4e48 会溢出成 inf，
    # 进而产生 NaN。真机上这一档必挂，mock 必须如实反映，否则搜索会被误导。
    if g["use_max"] == 0:
        headroom = 9.9
    elif g["div_mode"] == 0 and g["nr_steps"] == 0:
        headroom = 0.133                        # 硬件 Reciprocal 只有 9 位
    elif g["div_mode"] == 0 and g["nr_steps"] == 1:
        headroom = 0.005
    else:
        headroom = 0.00002
    return v1, headroom, 0.02


# --------------------------------------------------------------------------
# 真机后端
# --------------------------------------------------------------------------
def _run(cmd, timeout, cwd=ROOT):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT"


def _json_line(out):
    m = None
    for line in out.splitlines():
        if line.startswith("JSON "):
            m = line[5:]
    return json.loads(m) if m else None


def real_eval(g, py, workdir, fast=True):
    g = G.repair(g)
    d = os.path.join(ROOT, workdir)
    rc, out = _run(["cmake", "-S", ".", "-B", d, "-DPython3_EXECUTABLE=" + py]
                   + G.cmake_args(g), BUILD_TIMEOUT)
    if rc != 0:
        return dict(ok=False, tag="CONFIG_FAIL", log=out[-2000:])
    rc, out = _run(["cmake", "--build", d, "--target",
                    "sinkhorn_normalize_ops", "-j4"], BUILD_TIMEOUT)
    if rc != 0:
        return dict(ok=False, tag="COMPILE_FAIL", log=out[-2000:])

    so = os.path.join(d, "libsinkhorn_normalize_ops.so")
    if not os.path.isfile(so):
        return dict(ok=False, tag="NO_ARTIFACT")

    seeds = "2" if fast else "5"
    rc, out = _run([py, "scripts/check_shapes.py", "--so", so,
                    "--seeds", seeds, "--json"], RUN_TIMEOUT)
    j = _json_line(out)
    if j is None:
        return dict(ok=False, tag="RUNTIME_FAIL", log=out[-2000:])
    if not j["passed"]:
        return dict(ok=False, tag="PRECISION_FAIL",
                    headroom=j.get("max_headroom", 9.9))

    env = dict(os.environ, SINKHORN_OPS_SO=so)
    try:
        p = subprocess.run([py, "scripts/bench_official.py",
                            "--module", "submission/model_new.py", "--device", "npu",
                            "--mode", "interleaved",
                            "--warmup", "60" if fast else "200",
                            "--repeat", "180" if fast else "500",
                            "--baseline-ms", str(BASELINE_US / 1000.0), "--json"],
                           cwd=ROOT, env=env, timeout=RUN_TIMEOUT,
                           capture_output=True, text=True)
        b = _json_line(p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return dict(ok=False, tag="BENCH_TIMEOUT")
    if b is None or not b.get("ok"):
        return dict(ok=False, tag="BENCH_FAIL")
    return dict(ok=True, tag="OK", v1_us=b["v1_us"],
                headroom=j.get("max_headroom", 0.0),
                cv=b.get("cv", 0.0), iqr_us=b.get("iqr_us", 0.0))


# --------------------------------------------------------------------------
def evaluate(g, cache, backend="mock", rng=random, py="python3",
             algo="", gen=-1, keep_build=False, fast=True):
    k = G.key(g)
    hit = cache.get(k)
    if hit is not None:
        return hit

    t0 = time.time()
    if backend == "mock":
        v1, headroom, cv = mock_eval(g, rng)
        if headroom >= HEADROOM_LIMIT:
            res = dict(ok=False, tag="PRECISION_FAIL", headroom=headroom)
        else:
            res = dict(ok=True, tag="OK", v1_us=v1, headroom=headroom,
                       cv=cv, iqr_us=v1 * cv)
    else:
        workdir = "build_ga_" + k
        res = real_eval(g, py, workdir, fast=fast)
        if not keep_build:
            subprocess.run(["rm", "-rf", os.path.join(ROOT, workdir)],
                           capture_output=True)

    if res["ok"]:
        denom = res["v1_us"] + LAMBDA * res.get("iqr_us", 0.0)
        res["fitness"] = BASELINE_US / denom
    else:
        res["fitness"] = 0.0
    res.setdefault("v1_us", 0.0)
    res.setdefault("headroom", 0.0)
    res.setdefault("cv", 0.0)
    cache.put(k, G.repair(g), res, time.time() - t0, algo, gen)
    return res


if __name__ == "__main__":
    import cache as C
    rng = random.Random(0)
    c = C.Cache("/tmp/ga_smoke.sqlite")
    print("mock 代价模型自检（参数来自真机实测）")
    print("=" * 74)
    print("{:<12} {:>10} {:>10} {:>10}  {}".format(
        "已知个体", "v1 (us)", "fitness", "裕度", "真机实测 v1"))
    print("-" * 74)
    truth = {"s1d_arith": 158.55, "s1c_gmeps": 164.14,
             "s1_div": 172.69, "s1b_nr": 181.74}
    for name, g in G.SEEDS.items():
        r = evaluate(g, c, "mock", rng)
        print("{:<12} {:>10.2f} {:>10.3f} {:>10.5f}  {:.2f}".format(
            name, r["v1_us"], r["fitness"], r["headroom"], truth[name]))
    print("-" * 74)
    print("缓存: {}".format(c.stats()))
