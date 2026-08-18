#!/bin/bash
# =============================================================================
# S1 变体矩阵：定位精度问题 + 量化各修法的代价
#
#   v0       原 per-matrix kernel（对照，用满核）
#   s1_base  S1，裸 Reciprocal，不减 max      <- 上一轮实测: 6.88x 但精度 0.117 / big 失败
#   s1_nr    S1 + Newton-Raphson 修正倒数     <- 单独看 NR 能修多少
#   s1_full  S1 + NR + 减 max                <- 推荐配置
#   s1_div   S1 + 直接 Div + 减 max          <- 精度上限参考，看 Div 的时间代价
#
# 用法：  bash scripts/run_s1.sh [--skip-build] [--tile N] [--only 名字,名字]
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

SKIP_BUILD=0
TILE=64
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1 ;;
        --tile) shift; TILE="$1" ;;
        --only) shift; ONLY="$1" ;;
    esac
    shift
done

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "请先 source set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1

PY="${PYTHON:-python3}"
SO="libsinkhorn_normalize_ops.so"
hr() { printf '%.0s=' {1..78}; echo; }

# 名字 | cmake 选项
VARIANTS=(
  "v0|-DSN_KERNEL_VARIANT=0"
  "s1_base|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=0 -DSN_S1_NR_STEPS=0 -DSN_S1_USE_MAX=0"
  "s1_nr|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=0 -DSN_S1_NR_STEPS=2 -DSN_S1_USE_MAX=0"
  "s1_full|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=0 -DSN_S1_NR_STEPS=2 -DSN_S1_USE_MAX=1"
  "s1_div|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1"
)

selected() { [ -z "${ONLY}" ] && return 0; case ",${ONLY}," in *",$1,"*) return 0 ;; esac; return 1; }

if [ ${SKIP_BUILD} -eq 0 ]; then
    hr; echo "编译变体（SN_TILE_TARGET=${TILE}）"; hr
    for v in "${VARIANTS[@]}"; do
        name="${v%%|*}"; opts="${v#*|}"
        selected "${name}" || continue
        echo "--- ${name}: ${opts} ---"
        # shellcheck disable=SC2086
        cmake -S . -B "build_${name}" ${opts} -DSN_TILE_TARGET="${TILE}" 2>&1 \
            | grep -E "^-- SN_|error" || true
        cmake --build "build_${name}" --target sinkhorn_normalize_ops -j4 2>&1 \
            | grep -E "error|Error" && { echo ">>> ${name} 编译失败"; continue; }
        [ -f "build_${name}/${SO}" ] && echo "OK" || echo ">>> ${name} 产物缺失"
    done
fi

hr; echo "第 1 步：精度门禁（多形状 × 5 seed，官方容差 1e-2，要求裕度 < 0.5）"; hr
declare -A PREC
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    [ -f "build_${name}/${SO}" ] || continue
    echo
    echo "######## ${name} ########"
    if ${PY} scripts/check_shapes.py --so "${ROOT}/build_${name}/${SO}" --seeds 5; then
        PREC[${name}]=pass
    else
        PREC[${name}]=fail
    fi
done

hr; echo "第 2 步：时间拆解（只测精度通过的变体）"; hr
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    [ -f "build_${name}/${SO}" ] || continue
    if [ "${PREC[${name}]:-fail}" != "pass" ] && [ "${name}" != "s1_base" ]; then
        echo; echo "######## ${name}: 精度未通过，跳过计时 ########"
        continue
    fi
    echo
    echo "######## ${name} ########"
    ${PY} scripts/p0_host_breakdown.py --so "${ROOT}/build_${name}/${SO}" --tag "${name}" \
        | grep -E "M0|M1|M2|M5|M6|理论上限|当前 speedup|距离地板"
done

hr; echo "第 3 步：官方口径评测（精度通过的变体）"; hr
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    [ -f "build_${name}/${SO}" ] || continue
    [ "${PREC[${name}]:-fail}" = "pass" ] || continue
    echo
    echo "######## ${name} ########"
    SINKHORN_OPS_SO="${ROOT}/build_${name}/${SO}" ${PY} scripts/bench_official.py \
        --module submission/model_new.py --device npu --mode both \
        | grep -E "裕度占用|Speedup|interleaved|official|反 fallback"
done

hr
echo "精度门禁汇总:"
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    printf "  %-10s %s\n" "${name}" "${PREC[${name}]:-未构建}"
done
hr
