# 共用的编译辅助：日志落盘、失败时打印完整报错、自动清理过期的 CMakeCache
#
# 换容器 / 换机器后 build_*/CMakeCache.txt 里记的绝对路径会失效，
# 表现为 "Configuring incomplete, errors occurred!" 但看不到原因。
# 这里统一处理：配置失败 -> 删目录重试 -> 仍失败则打完整日志。

# ---------------------------------------------------------------------------
# 找到"真正装了 torch 的那个 python"。
# CMake 的 find_package(Python3) 会自己挑解释器，不同容器里可能挑到没装 torch 的那个
# （实测：新容器挑了 /usr/bin/python3.10，而 torch 装在 /usr/local/python3.11.15 下）。
# 编译和跑测试必须用同一个解释器，否则 .so 的 ABI 对不上、加载不了。
# ---------------------------------------------------------------------------
SN_TORCH_PY=""
sn_find_torch_python() {
    if [ -n "${SN_TORCH_PY}" ]; then echo "${SN_TORCH_PY}"; return 0; fi
    local cands=()
    [ -n "${PYTHON:-}" ] && cands+=("${PYTHON}")
    cands+=(python3 python)
    local g
    for g in /usr/local/python*/bin/python3 /usr/local/bin/python3 \
             /opt/*/bin/python3 /usr/bin/python3.*; do
        [ -x "$g" ] && cands+=("$g")
    done
    local p full
    for p in "${cands[@]}"; do
        full="$(command -v "$p" 2>/dev/null)" || full="$p"
        [ -x "$full" ] || continue
        if "$full" -c "import torch" >/dev/null 2>&1; then
            SN_TORCH_PY="$full"
            echo "$full"
            return 0
        fi
    done
    return 1
}

# 打印选中的解释器及其 torch / torch_npu 状态，找不到就报清楚
sn_require_torch_python() {
    local py
    if ! py="$(sn_find_torch_python)"; then
        echo "!! 找不到装了 torch 的 python 解释器。"
        echo "   试过: \$PYTHON, python3, python, /usr/local/python*/bin/python3,"
        echo "         /usr/local/bin/python3, /opt/*/bin/python3, /usr/bin/python3.*"
        echo "   请用 PYTHON=/路径/到/python3 bash scripts/xxx.sh 显式指定。"
        return 1
    fi
    echo "使用 Python: ${py}" >&2
    "${py}" -c "import torch;print('  torch     ', torch.__version__)" >&2 2>/dev/null || true
    "${py}" -c "import torch_npu;print('  torch_npu ', torch_npu.__version__)" >&2 2>/dev/null \
        || echo "  torch_npu  (导入失败，NPU 通路的脚本会跑不了)" >&2
    echo "${py}"
}

sn_configure() {   # $1=build目录  $2..=cmake 选项
    local dir="$1"; shift
    local log="${dir}.cmake.log"
    local pyopt=()
    local py
    if py="$(sn_find_torch_python)"; then
        pyopt=(-DPython3_EXECUTABLE="${py}")
    fi
    if cmake -S . -B "${dir}" "${pyopt[@]}" "$@" > "${log}" 2>&1; then
        grep -E "^-- SN_" "${log}" | sed 's/^/    /'
        return 0
    fi
    echo "    配置失败，清掉可能过期的缓存后重试…"
    rm -rf "${dir}"
    if cmake -S . -B "${dir}" "${pyopt[@]}" "$@" > "${log}" 2>&1; then
        grep -E "^-- SN_" "${log}" | sed 's/^/    /'
        return 0
    fi
    echo "    !! 配置仍然失败，完整日志（${log}）末尾 40 行："
    tail -40 "${log}" | sed 's/^/    | /'
    return 1
}

sn_build() {       # $1=build目录  $2=target
    local dir="$1" target="$2"
    local log="${dir}.build.log"
    if cmake --build "${dir}" --target "${target}" -j4 > "${log}" 2>&1; then
        return 0
    fi
    echo "    !! 编译失败，完整日志（${log}）末尾 60 行："
    tail -60 "${log}" | sed 's/^/    | /'
    return 1
}
