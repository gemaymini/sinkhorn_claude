#!/bin/bash
# =============================================================================
# 打包提交件
#
# 用法:  bash submission/package.sh "队伍名" "UID" [--skip-test]
# 产物:  dist/【2026KernelSwift算子创新大赛】-<队伍名>-Task03_sinkhorn-<UID>.tar.gz
#
# 提交件只包含规则 4.2 要求的内容：
#   算子优化代码 / README 文档 / 环境配置文件 / 运行脚本 / 性能测试结果
#
# 我们自己的验证（合规自检、精度门禁、时间拆解）照常跑，但**只输出到终端**，
# 不写进提交件——避免开发期的诊断日志混进去。
# 唯一落盘的是 results/performance.txt，那是规则 4.2 明确要求的「性能测试结果」。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
cd "${ROOT}"
source "${ROOT}/scripts/_build_lib.sh"

NAME="${1:-未填写队伍名}"
UID_STR="${2:-未填写UID}"
SKIP_TEST=0
for a in "$@"; do [ "$a" = "--skip-test" ] && SKIP_TEST=1; done

PKG="【2026KernelSwift算子创新大赛】-${NAME}-Task03_sinkhorn-${UID_STR}"
OUT="${ROOT}/dist/${PKG}"
rm -rf "${OUT}" && mkdir -p "${OUT}/results"

hr() { printf '%.0s-' {1..72}; echo; }

# ---------------------------------------------------------------- 0
echo "=== 0/5 确认写死版源码为最新 ==="
[ -d submission/src ] || { echo "!! submission/src 不存在，请先运行 python3 submission/gen_release_src.py"; exit 1; }
PY_GEN="${PYTHON:-python3}"
"${PY_GEN}" submission/verify_release_src.py || {
    echo "!! 写死版与开发版逻辑不一致，已中止"; exit 1; }

# ---------------------------------------------------------------- 1
echo
echo "=== 1/5 收集提交内容（仅规则 4.2 要求的文件）==="
cp -r submission/src/. "${OUT}/"                       # 算子优化代码 + CMakeLists
cp submission/README.md        "${OUT}/"               # README 文档
cp submission/requirements.txt "${OUT}/"               # 环境配置文件
cp submission/build.sh submission/run_auto_bench.sh "${OUT}/"   # 运行脚本
cp submission/model_ref.py submission/model_new.py  "${OUT}/"   # 评测入口
find "${OUT}" \( -name '*.orig' -o -name '*.log' -o -name '__pycache__' \) \
     -exec rm -rf {} + 2>/dev/null || true
echo "    已复制 $(find "${OUT}" -type f | wc -l | tr -d ' ') 个文件"

if [ ${SKIP_TEST} -eq 1 ]; then
    echo
    echo "=== 2-4/5 跳过编译与测试（--skip-test）==="
    : > "${OUT}/results/performance.txt"
else
    # ------------------------------------------------------------ 2
    echo
    echo "=== 2/5 在提交件目录内编译（验证自包含）==="
    ( cd "${OUT}" && bash build.sh ) || { echo "!! 编译失败"; exit 1; }
    SO="${OUT}/libsinkhorn_normalize_ops.so"
    [ -f "${SO}" ] || { echo "!! 未产出 .so"; exit 1; }

    PY="$(sn_require_torch_python)" || exit 1

    # ------------------------------------------------------------ 3
    echo
    echo "=== 3/5 自检（仅终端输出，不写入提交件）==="
    hr; echo "[合规自检]"
    "${PY}" submission/check_compliance.py --root "${OUT}" --so "${SO}" || {
        echo "!! 合规检查未通过，已中止打包"; exit 1; }
    hr; echo "[精度门禁] 12 形状 × 5 seed"
    "${PY}" scripts/check_shapes.py --so "${SO}" --seeds 5 \
        | tee /tmp/sn_prec.txt || { echo "!! 精度未通过"; exit 1; }
    hr; echo "[时间拆解] 每 tile 固定开销 vs 每次迭代开销"
    "${PY}" scripts/s1_scaling.py --so "${SO}" || true

    # ------------------------------------------------------------ 4
    echo
    echo "=== 4/5 官方口径性能测试 -> results/performance.txt ==="
    "${PY}" scripts/bench_official.py \
        --v0-file "${OUT}/model_ref.py" --v1-file "${OUT}/model_new.py" \
        --device npu --mode official | tee /tmp/sn_perf.txt

    "${PY}" - "${OUT}" /tmp/sn_perf.txt /tmp/sn_prec.txt <<'PYEOF' || true
import re, sys, datetime, platform
out, perf, prec = sys.argv[1], sys.argv[2], sys.argv[3]
p = open(perf, encoding="utf-8").read()
q = open(prec, encoding="utf-8").read()
sp = re.findall(r"Speedup = ([\d.]+)x\s+\(v0=([\d.]+)us, v1=([\d.]+)us\)", p)
hd = re.findall(r"裕度占用\s*:\s*([\d.]+)", p)
md = re.findall(r"最大绝对误差\s*:\s*([\d.eE+-]+)", p)
lines = [
    "SinkhornNormalize 性能测试结果",
    "=" * 60, "",
    "测试时间 : {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    "硬件     : 华为昇腾 Atlas A2 / 910B2 (dav-2201)",
    "CANN     : 9.0.0",
    "Python   : {}".format(platform.python_version()),
    "",
    "评测协议 (与官方 auto_bench.py 一致)",
    "-" * 60,
    "  warmup=200, repeat=500, 每次迭代 perf_counter 包住 forward 后 sync, 取 median",
    "  正确性 torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)",
    "  v0 = model_ref.py 的 Model (赛题参考实现)",
    "  v1 = model_new.py 的 ModelNew (本提交)",
    "",
    "性能",
    "-" * 60,
]
if sp:
    s, v0, v1 = sp[-1]
    lines += ["  v0 参考实现 : {} us".format(v0),
              "  v1 本提交   : {} us".format(v1),
              "  加速比      : {}x".format(s)]
lines += ["", "精度", "-" * 60]
if md:
    lines.append("  最大绝对误差 : {}".format(md[0]))
if hd:
    lines.append("  裕度占用     : {}   (= max|diff|/(atol+rtol*|ref|), <1 通过)".format(hd[0]))
m = re.search(r"结论: (.+)", q)
if m:
    lines.append("  多形状验证   : {}  (12 个形状 x 5 个随机种子)".format(m.group(1).strip()))
lines += ["", "复现方式", "-" * 60,
          "  bash build.sh          # 编译", "  bash run_auto_bench.sh # 用官方 auto_bench.py 评测", ""]
open(out + "/results/performance.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n已写入 results/performance.txt")
PYEOF
    rm -f /tmp/sn_perf.txt /tmp/sn_prec.txt
fi

# ---------------------------------------------------------------- 5
echo
echo "=== 5/5 打包 ==="
( cd "${ROOT}/dist" && tar czf "${PKG}.tar.gz" "${PKG}" )
echo
echo "提交件: ${ROOT}/dist/${PKG}.tar.gz"
du -sh "${ROOT}/dist/${PKG}.tar.gz" | sed 's/^/    /'
echo
echo "包内清单:"
( cd "${OUT}" && find . -type f | sed 's|^\./|    |' | sort )
echo
echo "邮件主题: 【2026KernelSwift算子创新大赛】-${NAME}-赛道二-${UID_STR}"
echo "收件邮箱: deeplink@pjlab.org.cn"
echo "PR 标题  : [KernelSwift算子优化]${NAME}-Task03_sinkhorn-${UID_STR}"
