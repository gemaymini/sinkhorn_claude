#!/bin/bash
# =============================================================================
# 对照实验矩阵：4 个搜索算法 × 3 个随机种子，同预算
#
# 预算按「真实评估次数」计，缓存命中不计费 —— 这是唯一公平的口径，
# 否则缓存命中率高的算法会凭空多拿评估次数。
#
# 用法:  bash ga/run_all.sh [预算，默认 60]
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"
source "${ROOT}/scripts/_build_lib.sh"

BUDGET="${1:-60}"
PY="$(sn_require_torch_python)" || exit 1
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "请先 source set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1

mkdir -p ga/runs
for algo in ga random local tpe; do
    for seed in 0 1 2; do
        db="ga/runs/${algo}_s${seed}.sqlite"
        if [ -f "${db}" ]; then
            echo "跳过已完成: ${db}"
            continue
        fi
        echo "==================== ${algo} seed=${seed} 预算=${BUDGET} ===================="
        ${PY} ga/search.py --algo "${algo}" --backend real --budget "${BUDGET}" \
              --seed "${seed}" --db "${db}" --python "${PY}"
    done
done

echo
echo "==================== 汇总分析 ===================="
${PY} ga/analyze.py ga/runs
