#!/bin/bash
# =============================================================================
# 一键编译 SinkhornNormalize 自定义算子
#
# 产物: build/libsinkhorn_normalize_ops.so
# 编译完成后会把 .so 复制到本目录，使 model_new.py 无需额外配置即可找到它。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "=== 1/3 检查 CANN 环境 ==="
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for c in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-*; do
        [ -f "$c/set_env.sh" ] && export ASCEND_HOME_PATH="$c" && break
    done
fi
[ -n "${ASCEND_HOME_PATH:-}" ] || {
    echo "找不到 CANN。请先 source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1
echo "    CANN: ${ASCEND_HOME_PATH}"

echo "=== 2/3 定位装有 torch 的解释器 ==="
# CMake 的 find_package(Python3) 可能挑到没装 torch 的解释器，必须显式指定
PY=""
for p in "${PYTHON:-}" python3 python /usr/local/python*/bin/python3 \
         /usr/local/bin/python3 /usr/bin/python3.*; do
    [ -n "$p" ] || continue
    full="$(command -v "$p" 2>/dev/null)" || full="$p"
    [ -x "$full" ] || continue
    if "$full" -c "import torch, torch_npu" >/dev/null 2>&1; then PY="$full"; break; fi
done
[ -n "${PY}" ] || { echo "找不到同时装有 torch 和 torch_npu 的 python"; exit 1; }
echo "    Python: ${PY}  ($(${PY} -c 'import torch;print(torch.__version__)'))"

echo "=== 3/3 编译 ==="
cmake -S . -B build -DPython3_EXECUTABLE="${PY}" >/dev/null || { echo "cmake 配置失败"; exit 1; }
cmake --build build --target sinkhorn_normalize_ops -j4 || { echo "编译失败"; exit 1; }

cp -f build/libsinkhorn_normalize_ops.so ./ 2>/dev/null || true
echo
echo "完成: ${HERE}/libsinkhorn_normalize_ops.so"
echo "验证:  ${PY} scripts/check_shapes.py --so ${HERE}/libsinkhorn_normalize_ops.so"
echo "评测:  ${PY} scripts/bench_official.py --v0-file model_ref.py \\"
echo "           --v1-file model_new.py --device npu --mode official"
echo "官方:  bash run_auto_bench.sh"
