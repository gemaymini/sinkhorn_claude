#!/bin/bash
# =============================================================================
# S1 验证 + A/B：原 per-matrix kernel  vs  S1 padded-AoS 批量 kernel
#
#   变体 v0 : SN_KERNEL_VARIANT=0  （原实现，host 侧已优化）
#   变体 s1 : SN_KERNEL_VARIANT=1  （S1）
#
# 用法：  bash scripts/run_s1.sh [--skip-build] [--tile N]
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

SKIP_BUILD=0
TILE=64
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1 ;;
        --tile) shift; TILE="$1" ;;
    esac
    shift
done

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "请先 source set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1

PY="${PYTHON:-python3}"
SO_NAME="libsinkhorn_normalize_ops.so"
hr() { printf '%.0s=' {1..78}; echo; }

build() {   # $1=目录  $2=SN_KERNEL_VARIANT
    echo "--- 编译 $1  (SN_KERNEL_VARIANT=$2, SN_TILE_TARGET=${TILE}) ---"
    cmake -S . -B "$1" -DSN_KERNEL_VARIANT="$2" -DSN_TILE_TARGET="${TILE}" 2>&1 \
        | grep -E "SN_|Error|error" || true
    cmake --build "$1" --target sinkhorn_normalize_ops -j4 || {
        echo ">>> 编译失败: $1"; return 1; }
    [ -f "$1/${SO_NAME}" ] || { echo ">>> 产物缺失: $1/${SO_NAME}"; return 1; }
    echo "OK: $1/${SO_NAME}"
}

if [ ${SKIP_BUILD} -eq 0 ]; then
    hr; echo "编译两个变体"; hr
    build build_s1_v0 0 || exit 1
    build build_s1    1 || exit 1
fi

hr; echo "第 1 步：S1 精度门禁（多形状 × 多 seed，官方容差）"; hr
${PY} scripts/check_shapes.py --so "${ROOT}/build_s1/${SO_NAME}" --seeds 5
PREC_RC=$?
if [ ${PREC_RC} -ne 0 ]; then
    echo
    echo ">>> S1 精度未通过。若失败集中在"最大绝对误差"很大或有 NaN/Inf，"
    echo ">>>   最可能是 CopyOut 的批量 DataCopyPad 读取步长和预期不符，试试保守回退："
    echo ">>>   cmake -S . -B build_s1 -DSN_KERNEL_VARIANT=1 -DSN_S1_COPYOUT_MODE=1 && \\"
    echo ">>>   cmake --build build_s1 --target sinkhorn_normalize_ops -j4"
fi

hr; echo "第 2 步：时间拆解 A/B"; hr
for v in v0 s1; do
    D="build_s1_v0"; [ "$v" = "s1" ] && D="build_s1"
    SO="${ROOT}/${D}/${SO_NAME}"
    [ -f "${SO}" ] || continue
    echo
    echo "############ 变体: ${v} ############"
    ${PY} scripts/p0_host_breakdown.py --so "${SO}" --tag "${v}"
done

hr; echo "第 3 步：官方口径评测 A/B"; hr
for v in v0 s1; do
    D="build_s1_v0"; [ "$v" = "s1" ] && D="build_s1"
    SO="${ROOT}/${D}/${SO_NAME}"
    [ -f "${SO}" ] || continue
    echo
    echo "############ 变体: ${v} ############"
    SINKHORN_OPS_SO="${SO}" ${PY} scripts/bench_official.py \
        --module submission/model_new.py --device npu --mode both
done
hr
