#!/bin/bash
# =============================================================================
# S1 变体矩阵：定位精度问题 + 量化各修法的代价
#
#   v0       原 per-matrix kernel（对照，用满核）           M1 ≈ 235us
#   s1_div   S1a：Div + 减 max，窄 ColNormalize            M1 ≈ 136us（上一轮最佳）
#   s1b_div  S1b：+ 宽 ColNormalize（8 矩阵/repeat）        <- 本轮主角
#   s1b_nr   S1b 但用 Reciprocal+NR 而非 Div               <- 宽路径下再比一次两者
#
# 判据用 M1（裸算子 + sync），不要用 speedup —— baseline 跨进程漂移 ±15%，
# 会把 2% 的真实差异淹没在 15% 的噪声里。
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

source "${SCRIPT_DIR}/_build_lib.sh"

PY="${PYTHON:-python3}"
SO="libsinkhorn_normalize_ops.so"
hr() { printf '%.0s=' {1..78}; echo; }

# 名字 | cmake 选项
VARIANTS=(
  "v0|-DSN_KERNEL_VARIANT=0"
  "s1_div|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=0"
  "s1b_div|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=1"
  "s1b_nr|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=0 -DSN_S1_NR_STEPS=2 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=1"
)

selected() { [ -z "${ONLY}" ] && return 0; case ",${ONLY}," in *",$1,"*) return 0 ;; esac; return 1; }

if [ ${SKIP_BUILD} -eq 0 ]; then
    hr; echo "编译变体（SN_TILE_TARGET=${TILE}）"; hr
    for v in "${VARIANTS[@]}"; do
        name="${v%%|*}"; opts="${v#*|}"
        selected "${name}" || continue
        echo "--- ${name}: ${opts} ---"
        # shellcheck disable=SC2086
        sn_configure "build_${name}" ${opts} -DSN_TILE_TARGET="${TILE}" || continue
        sn_build "build_${name}" sinkhorn_normalize_ops || continue
        [ -f "build_${name}/${SO}" ] && echo "    OK" || echo "    !! 产物缺失"
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
