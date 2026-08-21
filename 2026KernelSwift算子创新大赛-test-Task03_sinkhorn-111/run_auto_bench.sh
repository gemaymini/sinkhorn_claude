set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# ---------------------------------------------------------------- 默认值
# 下面五项与官方 auto_bench.py 的 argparse 默认值严格对应，改动前请先核对官方脚本。
DEF_SEED=42
DEF_ATOL=1e-2
DEF_RTOL=1e-2
DEF_WARMUP=200
DEF_REPEAT=500

SEED="${DEF_SEED}"
ATOL="${DEF_ATOL}"
RTOL="${DEF_RTOL}"
WARMUP="${DEF_WARMUP}"
REPEAT="${DEF_REPEAT}"

V0="${HERE}/model_ref.py"
V1="${HERE}/model_new.py"
BENCH="${AUTO_BENCH:-}"                       # 环境变量 AUTO_BENCH 可给默认路径
OUT="${HERE}/results/performance.txt"
PY="${PYTHON:-}"
FAIL_FAST=0
FULL_TB=0
NO_DOWNLOAD=0
NO_REPORT=0
EXTRA=()

BENCH_URL=https://raw.githubusercontent.com/DeepLink-org/DLBlas/main/benchmarks/ks/auto_bench.py

usage() {
    cat <<'USAGE'
用法: bash run_auto_bench.sh [选项] [-- 透传给 auto_bench.py 的其它参数]

评测参数（默认值与官方 auto_bench.py 一致）:
      --seed N            随机种子                      (默认 42)
      --atol F            allclose 绝对容差             (默认 1e-2)
      --rtol F            allclose 相对容差             (默认 1e-2)
      --warmup N          预热次数                      (默认 200)
      --repeat N          计时次数，取 median           (默认 500)
      --fail-fast         第一个用例失败即停止
      --full-traceback    加载/运行失败时打印完整 traceback
      --v0_file PATH      基准实现                      (默认 ./model_ref.py)
      --v1_file PATH      提交实现                      (默认 ./model_new.py)

脚本自身选项:
      --auto-bench PATH   官方 auto_bench.py 路径
                          (默认 $AUTO_BENCH，未设置则 /tmp/auto_bench.py；不存在时自动下载)
      --no-download       auto_bench.py 不存在时不自动下载，直接报错
      --python PATH       指定装有 torch/torch_npu 的解释器 (默认自动探测，也可用 $PYTHON)
      --output PATH       结果文件路径                  (默认 ./results/performance.txt)
      --no-report         只跑评测，不写结果文件
  -h, --help              显示本帮助

说明:
  * 评测参数的名称与官方 auto_bench.py 逐字一致，"--seed 7" 与 "--seed=7" 两种写法都支持。
  * "--" 之后的内容原样透传给 auto_bench.py，便于使用本脚本尚未包装的新选项。
  * 任何非默认参数都会在结果文件中被标注出来，避免与官方口径混淆。
USAGE
}
# ./run_auto_bench.sh --seed 43 --atol 1e-3 --rtol 1e-3  --fail-fast --full-traceback 

# ---------------------------------------------------------------- 参数解析
# 先把 "--opt=value" 拆成 "--opt" "value" 两个 token，后面就只需处理一种形式
NORM=()
for a in ${@+"$@"}; do
    case "${a}" in
        --*=*) NORM+=("${a%%=*}" "${a#*=}") ;;
        *)     NORM+=("${a}") ;;
    esac
done
set -- ${NORM[@]+"${NORM[@]}"}

# 取值型选项的公共检查：必须还有下一个 token，且它不像另一个选项
need_val() {
    if [ "$2" -lt 2 ]; then
        echo "错误: 选项 $1 需要一个值" >&2; exit 2
    fi
    case "$3" in
        -*) echo "错误: 选项 $1 的值缺失（读到了 $3）" >&2; exit 2 ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --seed)           need_val "$1" "$#" "${2:--}"; SEED="$2";   shift 2 ;;
        --atol)           need_val "$1" "$#" "${2:--}"; ATOL="$2";   shift 2 ;;
        --rtol)           need_val "$1" "$#" "${2:--}"; RTOL="$2";   shift 2 ;;
        --warmup)         need_val "$1" "$#" "${2:--}"; WARMUP="$2"; shift 2 ;;
        --repeat)         need_val "$1" "$#" "${2:--}"; REPEAT="$2"; shift 2 ;;
        --v0_file)        need_val "$1" "$#" "${2:--}"; V0="$2";     shift 2 ;;
        --v1_file)        need_val "$1" "$#" "${2:--}"; V1="$2";     shift 2 ;;
        --fail-fast)      FAIL_FAST=1; shift ;;
        --full-traceback) FULL_TB=1;   shift ;;
        --auto-bench)     need_val "$1" "$#" "${2:--}"; BENCH="$2";  shift 2 ;;
        --python)         need_val "$1" "$#" "${2:--}"; PY="$2";     shift 2 ;;
        --output)         need_val "$1" "$#" "${2:--}"; OUT="$2";    shift 2 ;;
        --no-download)    NO_DOWNLOAD=1; shift ;;
        --no-report)      NO_REPORT=1;   shift ;;
        --)               shift; while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done ;;
        *)                echo "错误: 未知参数 $1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

# 数值合法性：整数项必须是整数，容差必须是非负实数（支持 1e-2 写法）
is_int()   { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
is_float() { echo "$1" | grep -Eq '^[+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$'; }
is_int "${WARMUP}"   || { echo "错误: --warmup 必须是非负整数（读到 ${WARMUP}）" >&2; exit 2; }
is_int "${REPEAT}"   || { echo "错误: --repeat 必须是非负整数（读到 ${REPEAT}）" >&2; exit 2; }
echo "${SEED}" | grep -Eq '^-?[0-9]+$' || { echo "错误: --seed 必须是整数（读到 ${SEED}）" >&2; exit 2; }
is_float "${ATOL}"   || { echo "错误: --atol 必须是非负实数（读到 ${ATOL}）" >&2; exit 2; }
is_float "${RTOL}"   || { echo "错误: --rtol 必须是非负实数（读到 ${RTOL}）" >&2; exit 2; }
[ "${REPEAT}" -gt 0 ] || { echo "错误: --repeat 必须大于 0" >&2; exit 2; }

# ---------------------------------------------------------------- 环境准备
if [ -z "${BENCH}" ]; then
    BENCH=/tmp/auto_bench.py
fi
if [ ! -f "${BENCH}" ]; then
    if [ "${NO_DOWNLOAD}" = 1 ]; then
        echo "找不到 auto_bench.py: ${BENCH}（已指定 --no-download，不尝试下载）" >&2; exit 1
    fi
    curl -fsSL "${BENCH_URL}" -o "${BENCH}" || {
        echo "自动下载 auto_bench.py 失败，请手动指定路径：" >&2
        echo "  bash run_auto_bench.sh --auto-bench /path/to/auto_bench.py" >&2; exit 1; }
fi

for f in "${V0}" "${V1}"; do
    [ -f "${f}" ] || { echo "找不到模型文件: ${f}" >&2; exit 1; }
done

# .so 只在使用默认 v1（本提交件）时才检查；换了 --v1_file 就不该管
SO="${HERE}/libsinkhorn_normalize_ops.so"
if [ "${V1}" = "${HERE}/model_new.py" ] && [ ! -f "${SO}" ]; then
    echo "未找到 ${SO}，请先运行 bash build.sh" >&2; exit 1
fi

if [ -n "${PY}" ]; then
    "${PY}" -c "import torch, torch_npu" >/dev/null 2>&1 || {
        echo "指定的解释器缺少 torch 或 torch_npu: ${PY}" >&2; exit 1; }
else
    for p in python3 /usr/local/python*/bin/python3 /usr/bin/python3.*; do
        full="$(command -v "$p" 2>/dev/null)" || full="$p"
        [ -x "$full" ] && "$full" -c "import torch, torch_npu" >/dev/null 2>&1 && PY="$full" && break
    done
    [ -n "${PY}" ] || { echo "找不到同时装有 torch 和 torch_npu 的 python，可用 --python 指定" >&2; exit 1; }
fi

# ---------------------------------------------------------------- 组装并运行
BENCH_ARGS=(--v0_file "${V0}" --v1_file "${V1}"
            --seed "${SEED}" --atol "${ATOL}" --rtol "${RTOL}"
            --warmup "${WARMUP}" --repeat "${REPEAT}")
[ "${FAIL_FAST}" = 1 ] && BENCH_ARGS+=(--fail-fast)
[ "${FULL_TB}"   = 1 ] && BENCH_ARGS+=(--full-traceback)
BENCH_ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

# 结果文件里"复现方式"要给出可直接粘贴的命令，这里做最简单的 shell 引用
quote() { case "$1" in *[!A-Za-z0-9_./=-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
                       *) printf '%s' "$1" ;; esac; }
CMD="bash run_auto_bench.sh"
[ "${SEED}"   = "${DEF_SEED}" ]   || CMD="${CMD} --seed ${SEED}"
[ "${ATOL}"   = "${DEF_ATOL}" ]   || CMD="${CMD} --atol ${ATOL}"
[ "${RTOL}"   = "${DEF_RTOL}" ]   || CMD="${CMD} --rtol ${RTOL}"
[ "${WARMUP}" = "${DEF_WARMUP}" ] || CMD="${CMD} --warmup ${WARMUP}"
[ "${REPEAT}" = "${DEF_REPEAT}" ] || CMD="${CMD} --repeat ${REPEAT}"
[ "${V0}" = "${HERE}/model_ref.py" ] || CMD="${CMD} --v0_file $(quote "${V0}")"
[ "${V1}" = "${HERE}/model_new.py" ] || CMD="${CMD} --v1_file $(quote "${V1}")"
[ "${FAIL_FAST}" = 1 ] && CMD="${CMD} --fail-fast"
[ "${FULL_TB}"   = 1 ] && CMD="${CMD} --full-traceback"
if [ ${#EXTRA[@]} -gt 0 ]; then
    CMD="${CMD} --"
    for e in "${EXTRA[@]}"; do CMD="${CMD} $(quote "${e}")"; done
fi

echo "auto_bench : ${BENCH}"
echo "python     : ${PY}"
echo "v0 (参考)  : ${V0}"
echo "v1 (提交)  : ${V1}"
printf "评测参数   : seed=%s atol=%s rtol=%s warmup=%s repeat=%s" \
       "${SEED}" "${ATOL}" "${RTOL}" "${WARMUP}" "${REPEAT}"
[ "${FAIL_FAST}" = 1 ] && printf " --fail-fast"
[ "${FULL_TB}"   = 1 ] && printf " --full-traceback"
if [ "${SEED}" = "${DEF_SEED}" ] && [ "${ATOL}" = "${DEF_ATOL}" ] && \
   [ "${RTOL}" = "${DEF_RTOL}" ] && [ "${WARMUP}" = "${DEF_WARMUP}" ] && \
   [ "${REPEAT}" = "${DEF_REPEAT}" ] && [ ${#EXTRA[@]} -eq 0 ]; then
    printf "   (全部为官方默认值)\n"
else
    printf "\n             ^ 含非官方默认值，结果不可直接与官方口径比较\n"
fi
echo "----------------------------------------------------------------------"

RAW=$(mktemp)
trap 'rm -f "${RAW}"' EXIT

# tee 让结果同时出现在终端和临时文件；PIPESTATUS 取 auto_bench 自身的退出码
"${PY}" "${BENCH}" "${BENCH_ARGS[@]}" 2>&1 | tee "${RAW}"
RC=${PIPESTATUS[0]}

echo "----------------------------------------------------------------------"
if [ "${NO_REPORT}" = 1 ]; then
    echo "已指定 --no-report，未写结果文件"
    exit ${RC}
fi

mkdir -p "$(dirname "${OUT}")"
SN_HERE="${HERE}" SN_RAW="${RAW}" SN_BENCH="${BENCH}" SN_OUT="${OUT}" \
SN_SEED="${SEED}" SN_ATOL="${ATOL}" SN_RTOL="${RTOL}" \
SN_WARMUP="${WARMUP}" SN_REPEAT="${REPEAT}" \
SN_DEF="${DEF_SEED} ${DEF_ATOL} ${DEF_RTOL} ${DEF_WARMUP} ${DEF_REPEAT}" \
SN_V0="${V0}" SN_V1="${V1}" SN_CMD="${CMD}" \
SN_EXTRA="${EXTRA[*]+${EXTRA[*]}}" \
"${PY}" - <<'PYEOF'
import datetime, os, platform, re, subprocess, sys

env = os.environ
here, raw_path, bench = env["SN_HERE"], env["SN_RAW"], env["SN_BENCH"]
raw = open(raw_path, encoding="utf-8", errors="replace").read()

m = re.search(r"PASS accuracy; v0=([\d.]+) ms, v1=([\d.]+) ms, speedup=([\d.]+)x", raw)
fail = re.search(r"^FAIL (.+)$", raw, re.M)

d_seed, d_atol, d_rtol, d_warmup, d_repeat = env["SN_DEF"].split()
extra = env.get("SN_EXTRA", "").strip()


def mark(actual, default, numeric=float):
    """非官方默认值返回一段行尾标注，避免自定义参数跑出的数字被误当成官方口径。"""
    try:
        same = numeric(actual) == numeric(default)
    except ValueError:
        same = actual == default
    return "" if same else "   <== 非官方默认（官方 {}）".format(default)


custom = any(float(env[k]) != float(v) for k, v in
             (("SN_SEED", d_seed), ("SN_ATOL", d_atol), ("SN_RTOL", d_rtol),
              ("SN_WARMUP", d_warmup), ("SN_REPEAT", d_repeat))) or bool(extra)


def probe(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip() or "未知"
    except Exception:                                   # noqa: BLE001
        return "未知"


torch_ver = probe(sys.executable + " -c \"import torch;print(torch.__version__)\"")
npu_ver = probe(sys.executable + " -c \"import torch_npu;print(torch_npu.__version__)\"")
cann = env.get("ASCEND_HOME_PATH", "未设置")
soc = probe("npu-smi info -t board -i 0 2>/dev/null | awk -F: '/Chip Name|Name/{print $2; exit}'")

L = []
L.append("SinkhornNormalize 性能测试结果")
L.append("=" * 62)
L.append("")
L.append("由 run_auto_bench.sh 调用**官方原版** auto_bench.py 生成，非人工填写。")
L.append("")
L.append("测试时间   : {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
L.append("硬件       : 华为昇腾 Atlas A2 / 910B2 (dav-2201){}".format(
    "  " + soc if soc not in ("未知", "") else ""))
L.append("CANN       : {}".format(cann))
L.append("Python     : {}".format(platform.python_version()))
L.append("torch      : {}".format(torch_ver))
L.append("torch_npu  : {}".format(npu_ver))
L.append("评测脚本   : {}".format(bench))
L.append("")
L.append("评测协议{}".format(
    "（全部为官方 auto_bench.py 默认值）" if not custom
    else "（**含非官方默认参数**，不可直接与官方口径比较）"))
L.append("-" * 62)
L.append("  seed   = {}{}".format(env["SN_SEED"], mark(env["SN_SEED"], d_seed, int)))
L.append("  warmup = {}{}".format(env["SN_WARMUP"], mark(env["SN_WARMUP"], d_warmup, int)))
L.append("  repeat = {},  取 median{}".format(
    env["SN_REPEAT"], mark(env["SN_REPEAT"], d_repeat, int)))
L.append("  每次迭代 time.perf_counter() 包住整个 forward() 后 sync_devices()")
L.append("  正确性 torch.allclose(atol={}, rtol={}, equal_nan=True)".format(
    env["SN_ATOL"], env["SN_RTOL"]))
if float(env["SN_ATOL"]) != float(d_atol) or float(env["SN_RTOL"]) != float(d_rtol):
    L.append("         <== 非官方默认（官方 atol={}, rtol={}）".format(d_atol, d_rtol))
if extra:
    L.append("  额外透传参数 : {}".format(extra))
L.append("  v0 = {} 的 Model".format(os.path.basename(env["SN_V0"])))
L.append("  v1 = {} 的 ModelNew   （本提交）".format(os.path.basename(env["SN_V1"])))
L.append("")
L.append("结果")
L.append("-" * 62)
if m:
    v0_ms, v1_ms, sp = float(m.group(1)), float(m.group(2)), float(m.group(3))
    L.append("  正确性     : PASS")
    L.append("  v0 参考实现 : {:10.3f} us".format(v0_ms * 1000))
    L.append("  v1 本提交   : {:10.3f} us".format(v1_ms * 1000))
    L.append("  加速比      : {:10.3f}x".format(sp))
elif fail:
    L.append("  正确性     : FAIL")
    L.append("  失败原因   : {}".format(fail.group(1).strip()))
else:
    L.append("  未能从输出中解析出结果，原始输出见下方。")
L.append("")
L.append("复现方式")
L.append("-" * 62)
L.append("  bash build.sh            # 编译（不需要任何 -D 选项）")
L.append("  {}{}".format(
    env["SN_CMD"],
    "   # 官方默认口径" if not custom else "   # 本次使用的自定义参数"))
L.append("  bash run_auto_bench.sh --help    # 查看全部可调参数")
L.append("")
L.append("auto_bench.py 原始输出")
L.append("-" * 62)
for line in raw.rstrip().split("\n"):
    L.append("  " + line)
L.append("")

out = env["SN_OUT"]
open(out, "w", encoding="utf-8").write("\n".join(L))
print("结果已写入 {}".format(out))
PYEOF

exit ${RC}
