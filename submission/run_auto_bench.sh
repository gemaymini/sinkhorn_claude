#!/bin/bash
# =============================================================================
# 用**官方原版** auto_bench.py 评测本提交件
#
# 用法:
#   bash run_auto_bench.sh /path/to/DLBlas/benchmarks/ks/auto_bench.py
#   bash run_auto_bench.sh          # 不传路径则尝试自动下载到 /tmp
#
# 与仓库内 scripts/bench_official.py 的区别：那是我们复刻的版本（多了交替测量、
# 固定 baseline 等调优用功能），这个脚本跑的是官方原件，用于最终确认。
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
SINKHORN_OPS_SO="${SO}" "${PY}" "${BENCH}" \
    --v0_file "${HERE}/model_ref.py" \
    --v1_file "${HERE}/model_new.py"
