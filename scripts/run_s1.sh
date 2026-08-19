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
SWEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1 ;;
        --tile) shift; TILE="$1" ;;
        --only) shift; ONLY="$1" ;;
        --sweep) SWEEP=1 ;;
    esac
    shift
done

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "请先 source set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1

source "${SCRIPT_DIR}/_build_lib.sh"

PY="$(sn_require_torch_python)" || exit 1
SO="libsinkhorn_normalize_ops.so"
hr() { printf '%.0s=' {1..78}; echo; }

# 名字 | cmake 选项
VARIANTS=(
  "s1c_gmeps|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=1 -DSN_S1_BARRIER_MODE=1 -DSN_S1_EPS_MODE=0"
  "s1d_arith|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=1 -DSN_S1_BARRIER_MODE=1 -DSN_S1_EPS_MODE=1"
  "diag_floor|-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=1 -DSN_S1_BARRIER_MODE=1 -DSN_S1_EPS_MODE=1 -DSN_S1_DIAG=1"
)

selected() { [ -z "${ONLY}" ] && return 0; case ",${ONLY}," in *",$1,"*) return 0 ;; esac; return 1; }


# ---------------------------------------------------------------------------
# tile 扫描：固定开销是否随「每核矩阵数」缩放？
#   每核矩阵数 n 决定 CopyIn/CopyOut/BuildEpsVector 的 MTE 描述符数（约 3n 个 16B 块），
#   同时决定用多少核（1024/n）。
#   随 n 减小而变快  => MTE 描述符是瓶颈，应继续减小 tile / 改布局减少描述符
#   基本不随 n 变化  => launch/dispatch 是瓶颈，kernel 里再优化都没用
# ---------------------------------------------------------------------------
if [ ${SWEEP} -eq 1 ]; then
    hr; echo "tile 扫描（配置 = 当前最好的 s1c）"; hr
    OPTS="-DSN_KERNEL_VARIANT=1 -DSN_S1_DIV_MODE=1 -DSN_S1_USE_MAX=1 -DSN_S1_COLNORM_WIDE=1 -DSN_S1_BARRIER_MODE=1"
    printf "%-8s %-10s %-14s %-12s %s\n" "tile" "核数" "MTE块数/核" "M1 (us)" "固定开销 a (us)"
    printf -- "----------------------------------------------------------------------\n"
    for t in 8 16 32 64 128; do
        d="build_sweep_${t}"
        # shellcheck disable=SC2086
        sn_configure "${d}" ${OPTS} -DSN_TILE_TARGET="${t}" >/dev/null 2>&1 || { echo "配置失败 tile=${t}"; continue; }
        sn_build "${d}" sinkhorn_normalize_ops >/dev/null 2>&1 || { echo "编译失败 tile=${t}"; continue; }
        m1=$(${PY} scripts/p0_host_breakdown.py --so "${ROOT}/${d}/${SO}" --repeat-torch 20 2>/dev/null \
             | awk '/M1  裸算子调用/ {print $NF}')
        a=$(${PY} scripts/s1_scaling.py --so "${ROOT}/${d}/${SO}" --iters 200 2>/dev/null \
             | awk '/每 tile 固定开销/ {print $(NF-2)}')
        cores=$(( (1024 + t - 1) / t ))
        printf "%-8s %-10s %-14s %-12s %s\n" "${t}" "${cores}" "$((3*t*4))" "${m1:-?}" "${a:-?}"
    done
    hr
    exit 0
fi

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
    case "${name}" in
        diag_*) echo "  诊断变体，结果不正确，跳过精度门禁"; PREC[${name}]=diag ;;
        *) if ${PY} scripts/check_shapes.py --so "${ROOT}/build_${name}/${SO}" --seeds 5; then
               PREC[${name}]=pass
           else
               PREC[${name}]=fail
           fi ;;
    esac
done

hr; echo "第 2 步：时间拆解（只测精度通过的变体）"; hr
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    [ -f "build_${name}/${SO}" ] || continue
    if [ "${PREC[${name}]:-fail}" = "fail" ]; then
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
    cp submission/model_new.py "build_${name}/"     # .so 必须在 model_new.py 旁边
    ${PY} scripts/bench_official.py \
        --v0-file submission/model_ref.py --v1-file "build_${name}/model_new.py" \
        --device npu --mode both \
        | grep -E "裕度占用|Speedup|interleaved|official|反 fallback"
done

hr; echo "第 4 步：迭代次数扫描（拆出每 tile 固定开销 vs 每次迭代开销）"; hr
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    [ "${PREC[${name}]:-fail}" = "fail" ] && continue
    echo
    echo "######## ${name} ########"
    ${PY} scripts/s1_scaling.py --so "${ROOT}/build_${name}/${SO}" | tail -16
done

hr
echo "精度门禁汇总:"
for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    selected "${name}" || continue
    printf "  %-10s %s\n" "${name}" "${PREC[${name}]:-未构建}"
done
hr
