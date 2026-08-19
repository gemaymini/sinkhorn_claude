#!/bin/bash
# =============================================================================
# GA 搜索：多个随机种子独立跑，结果汇总
#
# 多 seed 不是为了做对照，而是因为单次 GA 受随机初始化影响较大；
# 多跑几个种子取并集，既能提高找到好个体的概率，也能看出结果稳不稳。
# apply_best.py 会扫描全部结果库，所以每个 seed 的候选都会进入最终仲裁。
#
# 用法:
#   bash ga/run_ga.sh              # 预算 60，3 个 seed
#   bash ga/run_ga.sh 30           # 预算 30
#   bash ga/run_ga.sh 30 5         # 预算 30，5 个 seed
#
# 如果之后需要给报告补一个随机搜索对照：
#   python ga/search.py --algo random --backend real --budget 60 --seed 0 \
#          --db ga/runs/random_s0.sqlite
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"
source "${ROOT}/scripts/_build_lib.sh"

BUDGET="${1:-60}"
NSEED="${2:-3}"
PY="$(sn_require_torch_python)" || exit 1
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "请先 source set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1

EST=$(( BUDGET * 51 * NSEED / 60 ))
echo "=============================================================================="
echo "GA 搜索   预算=${BUDGET} 次真实评估/seed   seed 数=${NSEED}"
echo "单次评估约 51s，预计总耗时 ${EST} 分钟"
echo "已完成的 seed 会自动跳过；中断后重跑可续（未完成的留在 .partial）"
echo "另开终端可用  bash ga/monitor.sh  查看进度"
echo "=============================================================================="

mkdir -p ga/runs
for ((seed=0; seed<NSEED; seed++)); do
    db="ga/runs/ga_s${seed}.sqlite"
    if [ -f "${db}" ]; then
        echo
        echo "跳过已完成: ${db}"
        continue
    fi
    echo
    echo "==================== GA seed=${seed} ===================="
    ${PY} ga/search.py --algo ga --backend real --budget "${BUDGET}" \
          --seed "${seed}" --db "${db}" --python "${PY}"
done

echo
echo "==================== 汇总分析 ===================="
${PY} ga/analyze.py ga/runs

echo
echo "下一步："
echo "  ${PY} ga/apply_best.py            # 全档复验候选，看是否值得替换当前默认配置"
echo "  ${PY} ga/apply_best.py --apply    # 确认后写入 CMakeLists.txt"
