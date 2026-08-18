# 共用的编译辅助：日志落盘、失败时打印完整报错、自动清理过期的 CMakeCache
#
# 换容器 / 换机器后 build_*/CMakeCache.txt 里记的绝对路径会失效，
# 表现为 "Configuring incomplete, errors occurred!" 但看不到原因。
# 这里统一处理：配置失败 -> 删目录重试 -> 仍失败则打完整日志。

sn_configure() {   # $1=build目录  $2..=cmake 选项
    local dir="$1"; shift
    local log="${dir}.cmake.log"
    if cmake -S . -B "${dir}" "$@" > "${log}" 2>&1; then
        grep -E "^-- SN_" "${log}" | sed 's/^/    /'
        return 0
    fi
    echo "    配置失败，清掉可能过期的缓存后重试…"
    rm -rf "${dir}"
    if cmake -S . -B "${dir}" "$@" > "${log}" 2>&1; then
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
