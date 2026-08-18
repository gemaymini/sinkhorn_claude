#!/bin/bash
# =============================================================================
# P0 一键跑：host 侧优化 A/B + 提交形态验证 + 时间拆解
#
#   变体 orig : SN_TILING_MODE=0 SN_CORE_QUERY=0  （复刻原实现，对照组）
#   变体 opt  : SN_TILING_MODE=1 SN_CORE_QUERY=1  （静态缓存 tiling + 缓存核数）
#
# 两个 .so 注册同名 TORCH_LIBRARY 算子，**不能在同一进程内同时加载**，
# 因此每个变体都在独立的 python 进程里测。
#
# 用法：  bash scripts/run_p0.sh [--skip-build]
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

SKIP_BUILD=0
for a in "$@"; do [ "$a" = "--skip-build" ] && SKIP_BUILD=1; done

PY="${PYTHON:-python3}"
SO_NAME="libsinkhorn_normalize_ops.so"

hr() { printf '%.0s=' {1..78}; echo; }

hr; echo "P0 步骤 1/4：环境"; hr
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ASCEND_HOME_PATH 未设置，请先 source .../set_env.sh"; exit 1
fi
source "${ASCEND_HOME_PATH}/set_env.sh" || { echo "set_env.sh 失败"; exit 1; }
echo "CANN: ${ASCEND_HOME_PATH}"

hr; echo "P0 步骤 2/4：提交形态验证（AST 过滤）"; hr
${PY} scripts/p0_ast_check.py submission/model_new.py
AST_RC=$?
[ ${AST_RC} -ne 0 ] && echo ">>> AST 检查未通过，提交上去会跑不起来，先修这个。"

build_variant() {   # $1=目录  $2=SN_TILING_MODE  $3=SN_CORE_QUERY
    local dir="$1" tm="$2" cq="$3"
    echo "--- 编译 ${dir}  (SN_TILING_MODE=${tm}, SN_CORE_QUERY=${cq}) ---"
    cmake -S . -B "${dir}" -DSN_TILING_MODE="${tm}" -DSN_CORE_QUERY="${cq}" >/dev/null \
        || { echo "cmake 配置失败: ${dir}"; return 1; }
    cmake --build "${dir}" --target sinkhorn_normalize_ops -j4 >/dev/null \
        || { echo "编译失败: ${dir}"; return 1; }
    [ -f "${dir}/${SO_NAME}" ] || { echo "产物缺失: ${dir}/${SO_NAME}"; return 1; }
    echo "OK: ${dir}/${SO_NAME}"
}

hr; echo "P0 步骤 3/4：编译两个变体"; hr
if [ ${SKIP_BUILD} -eq 1 ]; then
    echo "跳过编译"
else
    build_variant build_p0_orig 0 0 || exit 1
    build_variant build_p0_opt  1 1 || exit 1
fi

hr; echo "P0 步骤 4/4：时间拆解 + 官方口径 A/B"; hr
for v in orig opt; do
    SO="${ROOT}/build_p0_${v}/${SO_NAME}"
    [ -f "${SO}" ] || { echo "跳过 ${v}：${SO} 不存在"; continue; }
    echo
    echo "#################### 变体: ${v} ####################"
    ${PY} scripts/p0_host_breakdown.py --so "${SO}" --tag "${v}"
    echo
    SINKHORN_OPS_SO="${SO}" ${PY} scripts/bench_official.py \
        --module submission/model_new.py --device npu
done

hr
echo "可选：采 msprof 拿纯 kernel 时间"
echo "  msprof --output=./prof_p0 --application=\"${PY} scripts/p0_msprof_target.py --so ${ROOT}/build_p0_opt/${SO_NAME}\""
echo "  然后在 prof_p0/**/op_summary_*.csv 里查 sinkhorn_normalize_kernel 的 Task Duration"
hr
