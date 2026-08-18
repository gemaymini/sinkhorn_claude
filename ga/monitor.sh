#!/bin/bash
# 另开终端跑这个，看 GA 到底在不在干活
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
while true; do
    clear
    echo "===== $(date '+%H:%M:%S')  GA 运行监控 ====="
    echo
    echo "--- 已完成的评估数（各结果库）---"
    for db in ga/runs/*.sqlite*; do
        [ -f "$db" ] || continue
        n=$(sqlite3 "$db" "SELECT COUNT(*) FROM evals" 2>/dev/null || echo "?")
        ok=$(sqlite3 "$db" "SELECT COUNT(*) FROM evals WHERE ok=1" 2>/dev/null || echo "?")
        best=$(sqlite3 "$db" "SELECT ROUND(MAX(fitness),4) FROM evals WHERE ok=1" 2>/dev/null || echo "-")
        printf "  %-34s 共%-5s 成功%-5s 最优fitness=%s\n" "$(basename "$db")" "$n" "$ok" "$best"
    done
    echo
    echo "--- 当前正在编译的目录 ---"
    ls -d build_ga_* 2>/dev/null | head -3 | sed 's/^/  /' || echo "  （无）"
    echo
    echo "--- 相关进程 ---"
    ps -eo etime,comm,args --no-headers 2>/dev/null \
        | grep -E "cmake|bisheng|ccec|search\.py|check_shapes|bench_official" \
        | grep -v grep | head -6 \
        | awk '{printf "  %-10s %s\n", $1, substr($0, index($0,$2), 90)}' || echo "  （无）"
    echo
    echo "（Ctrl-C 退出监控，不影响 GA 运行）"
    sleep 10
done
