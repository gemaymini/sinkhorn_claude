#!/bin/bash
# =============================================================================
# 用**官方原版** auto_bench.py 评测本提交件，并把结果写入 results/performance.txt
#
# 用法:
#   bash run_auto_bench.sh                                  # 自动下载 auto_bench.py 到 /tmp
#   bash run_auto_bench.sh /path/to/auto_bench.py           # 指定本地路径
#
# 官方评测协议：warmup=200，repeat=500，每次迭代 perf_counter 包住 forward 后 sync，
# 取 median；正确性 torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

BENCH="${1:-}"
if [ -z "${BENCH}" ]; then
    BENCH=/tmp/auto_bench.py
    URL=https://raw.githubusercontent.com/DeepLink-org/DLBlas/main/benchmarks/ks/auto_bench.py
    [ -f "${BENCH}" ] || curl -fsSL "${URL}" -o "${BENCH}" || {
        echo "自动下载失败，请手动指定路径："
        echo "  bash run_auto_bench.sh /path/to/auto_bench.py"; exit 1; }
fi
[ -f "${BENCH}" ] || { echo "找不到 auto_bench.py: ${BENCH}"; exit 1; }

SO="${HERE}/libsinkhorn_normalize_ops.so"
[ -f "${SO}" ] || { echo "未找到 ${SO}，请先运行 bash build.sh"; exit 1; }

PY=""
for p in "${PYTHON:-}" python3 /usr/local/python*/bin/python3 /usr/bin/python3.*; do
    [ -n "$p" ] || continue
    full="$(command -v "$p" 2>/dev/null)" || full="$p"
    [ -x "$full" ] && "$full" -c "import torch, torch_npu" >/dev/null 2>&1 && PY="$full" && break
done
[ -n "${PY}" ] || { echo "找不到同时装有 torch 和 torch_npu 的 python"; exit 1; }

echo "auto_bench : ${BENCH}"
echo "python     : ${PY}"
echo "v0 (参考)  : ${HERE}/model_ref.py"
echo "v1 (提交)  : ${HERE}/model_new.py"
echo "----------------------------------------------------------------------"

RAW=$(mktemp)
trap 'rm -f "${RAW}"' EXIT

# tee 让结果同时出现在终端和临时文件；PIPESTATUS 取 auto_bench 自身的退出码
"${PY}" "${BENCH}" \
    --v0_file "${HERE}/model_ref.py" \
    --v1_file "${HERE}/model_new.py" 2>&1 | tee "${RAW}"
RC=${PIPESTATUS[0]}

echo "----------------------------------------------------------------------"
mkdir -p "${HERE}/results"
"${PY}" - "${HERE}" "${RAW}" "${BENCH}" <<'PYEOF'
import datetime, os, platform, re, subprocess, sys

here, raw_path, bench = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(raw_path, encoding="utf-8", errors="replace").read()

m = re.search(r"PASS accuracy; v0=([\d.]+) ms, v1=([\d.]+) ms, speedup=([\d.]+)x", raw)
fail = re.search(r"^FAIL (.+)$", raw, re.M)


def probe(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip() or "未知"
    except Exception:                                   # noqa: BLE001
        return "未知"


torch_ver = probe(sys.executable + " -c \"import torch;print(torch.__version__)\"")
npu_ver = probe(sys.executable + " -c \"import torch_npu;print(torch_npu.__version__)\"")
cann = os.environ.get("ASCEND_HOME_PATH", "未设置")
soc = probe("npu-smi info -t board -i 0 2>/dev/null | awk -F: '/Chip Name|Name/{print $2; exit}'")

L = []
L.append("SinkhornNormalize 性能测试结果")
L.append("=" * 62)
L.append("")
L.append("由 run_auto_bench.sh 调用**官方原版** auto_bench.py 生成，非人工填写。")
L.append("")
L.append("测试时间   : {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
L.append("硬件       : 华为昇腾 Atlas A2 / 910B2 (dav-2201){}".format(
    "  " + soc if soc not in ("未知", "") else ""))
L.append("CANN       : {}".format(cann))
L.append("Python     : {}".format(platform.python_version()))
L.append("torch      : {}".format(torch_ver))
L.append("torch_npu  : {}".format(npu_ver))
L.append("评测脚本   : {}".format(bench))
L.append("")
L.append("评测协议（官方 auto_bench.py 默认值）")
L.append("-" * 62)
L.append("  warmup = 200,  repeat = 500,  取 median")
L.append("  每次迭代 time.perf_counter() 包住整个 forward() 后 sync_devices()")
L.append("  正确性 torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)")
L.append("  v0 = model_ref.py 的 Model      （赛题参考实现）")
L.append("  v1 = model_new.py 的 ModelNew   （本提交）")
L.append("")
L.append("结果")
L.append("-" * 62)
if m:
    v0_ms, v1_ms, sp = float(m.group(1)), float(m.group(2)), float(m.group(3))
    L.append("  正确性     : PASS")
    L.append("  v0 参考实现 : {:10.3f} us".format(v0_ms * 1000))
    L.append("  v1 本提交   : {:10.3f} us".format(v1_ms * 1000))
    L.append("  加速比      : {:10.3f}x".format(sp))
elif fail:
    L.append("  正确性     : FAIL")
    L.append("  失败原因   : {}".format(fail.group(1).strip()))
else:
    L.append("  未能从输出中解析出结果，原始输出见下方。")
L.append("")
L.append("复现方式")
L.append("-" * 62)
L.append("  bash build.sh            # 编译（不需要任何 -D 选项）")
L.append("  bash run_auto_bench.sh   # 用官方 auto_bench.py 评测并刷新本文件")
L.append("")
L.append("auto_bench.py 原始输出")
L.append("-" * 62)
for line in raw.rstrip().split("\n"):
    L.append("  " + line)
L.append("")

out = os.path.join(here, "results", "performance.txt")
open(out, "w", encoding="utf-8").write("\n".join(L))
print("结果已写入 results/performance.txt")
PYEOF

exit ${RC}
