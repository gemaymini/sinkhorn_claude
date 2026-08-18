#!/bin/bash
# S1 前置 API 探针：编译 + 运行 + 解读
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "请先 source set_env.sh"; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || exit 1

source "${SCRIPT_DIR}/_build_lib.sh"

PY="$(sn_require_torch_python)" || exit 1
BUILD=build_probe

echo "=== 编译探针 ==="
sn_configure "${BUILD}" || exit 1
sn_build "${BUILD}" probe_sinkhorn || {
    echo
    echo ">>> 编译失败。把上面的完整报错发回来即可。"
    echo ">>> 如果只是 probe_gather_cost 的标量参数不被支持，"
    echo ">>>   把 probe/probe_kernel.asc 末尾的 KernelGatherCost / probe_gather_cost 整段注释掉，"
    echo ">>>   以及 probe/probe_main.asc 里第二部分（抽平面成本）整段注释掉，再跑一次。"
    exit 1
}

echo
echo "=== 运行探针 ==="
cd "${BUILD}"
./probe_sinkhorn || { echo "探针运行失败"; exit 1; }
cd "${ROOT}"

echo
echo "=== 解读结果 ==="
${PY} scripts/probe_report.py "${BUILD}/probe_layout.bin"
