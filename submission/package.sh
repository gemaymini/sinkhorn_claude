#!/bin/bash
# =============================================================================
# 打包提交件
#
# 用法:  bash submission/package.sh "队伍名" "UID" [--skip-test]
# 产物:  dist/【2026KernelSwift算子创新大赛】-<队伍名>-Task03_sinkhorn-<UID>.tar.gz
#
# 会先跑一遍精度门禁和官方口径评测，把结果写进 results/，确保提交的性能数据
# 就是这一份代码实际跑出来的。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
cd "${ROOT}"

NAME="${1:-未填写队伍名}"
UID_STR="${2:-未填写UID}"
SKIP_TEST=0
for a in "$@"; do [ "$a" = "--skip-test" ] && SKIP_TEST=1; done

PKG="【2026KernelSwift算子创新大赛】-${NAME}-Task03_sinkhorn-${UID_STR}"
OUT="${ROOT}/dist/${PKG}"
rm -rf "${OUT}" && mkdir -p "${OUT}/results"

echo "=== 0/4 确认生效的构建配置 ==="
# build.sh 不带任何 -D，用的就是 CMakeLists.txt 里的默认值。
# 若跑过 GA，先用 ga/apply_best.py 把最优基因写进默认值，再打包。
CFG=$(grep -E "^set\(SN_[A-Z0-9_]+[[:space:]]+[0-9]+" CMakeLists.txt \
      | sed -E 's/^set\((SN_[A-Z0-9_]+)[[:space:]]+([0-9]+).*/  \1=\2/')
echo "${CFG}"
mkdir -p "${OUT}/results"
{ echo "打包时生效的构建配置（来自 CMakeLists.txt 默认值）"; echo; echo "${CFG}"; } \
    > /tmp/sn_build_config.txt

echo "=== 1/4 收集源码 ==="
cp -r op_kernel op_extension op_host CMakeLists.txt "${OUT}/"
find "${OUT}" \( -name '*.orig' -o -name '*.log' -o -name '__pycache__' \) \
     -exec rm -rf {} + 2>/dev/null || true
cp submission/model_new.py submission/model_ref.py submission/build.sh \
   submission/requirements.txt submission/README.md \
   submission/run_auto_bench.sh submission/check_compliance.py "${OUT}/"
mkdir -p "${OUT}/scripts"
for f in check_shapes.py bench_official.py p0_ast_check.py s1_scaling.py \
         p0_host_breakdown.py golden.py; do
    [ -f "scripts/$f" ] && cp "scripts/$f" "${OUT}/scripts/"
done
# 不需要的诊断变体源文件一并带上原 kernel 作为对照，便于评委复现 A/B
cp op_kernel/sinkhorn_normalize_kernel.asc "${OUT}/op_kernel/" 2>/dev/null || true
cp /tmp/sn_build_config.txt "${OUT}/results/build_config.txt" 2>/dev/null || true
echo "    源码已复制"

if [ ${SKIP_TEST} -eq 1 ]; then
    echo "=== 2-3/4 跳过编译与测试（--skip-test）==="
else
    echo "=== 2/4 在提交件目录内编译（验证自包含）==="
    ( cd "${OUT}" && bash build.sh ) 2>&1 | tail -5
    SO="${OUT}/libsinkhorn_normalize_ops.so"
    [ -f "${SO}" ] || { echo "!! 编译未产出 .so，提交件不完整"; exit 1; }

    echo "=== 3/4 精度门禁 + 官方口径评测 ==="
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
    PY="$(command -v python3)"
    for p in /usr/local/python*/bin/python3; do
        [ -x "$p" ] && "$p" -c "import torch_npu" >/dev/null 2>&1 && PY="$p" && break
    done
    ( cd "${OUT}" && "${PY}" scripts/check_shapes.py --so "${SO}" --seeds 5 ) \
        | tee "${OUT}/results/precision.txt" | tail -4
    ( cd "${OUT}" && "${PY}" scripts/bench_official.py \
        --v0-file model_ref.py --v1-file model_new.py --device npu --mode both ) \
        | tee "${OUT}/results/performance.txt" | tail -6
    ( cd "${OUT}" && "${PY}" scripts/s1_scaling.py --so "${SO}" ) \
        > "${OUT}/results/scaling.txt" 2>&1 || true
    echo "    结果已写入 ${OUT}/results/"

    echo "=== 3.5/4 合规自检（打包的最后一道闸）==="
    if ! ( cd "${OUT}" && "${PY}" check_compliance.py --so "${SO}" ); then
        echo
        echo "!! 合规检查未通过，已中止打包。请修复上面标 [不符] 的项后重试。"
        exit 1
    fi
fi

echo "=== 4/4 打包 ==="
# .so 是平台相关的编译产物，保留可以让评委免编译直接跑；如需纯源码提交可删掉
( cd "${ROOT}/dist" && tar czf "${PKG}.tar.gz" "${PKG}" )
echo
echo "提交件: ${ROOT}/dist/${PKG}.tar.gz"
du -sh "${ROOT}/dist/${PKG}.tar.gz" | sed 's/^/    /'
echo
echo "邮件主题: 【2026KernelSwift算子创新大赛】-${NAME}-赛道二-${UID_STR}"
echo "收件邮箱: deeplink@pjlab.org.cn"
echo "PR 标题  : [KernelSwift算子优化]${NAME}-Task03_sinkhorn-${UID_STR}"
