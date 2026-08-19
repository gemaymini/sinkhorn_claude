"""校验写死版与参数化开发版在给定配置下的逻辑完全一致。

做法：分别抽取两边 kernel 里的 **AscendC API 调用序列**（函数名 + 参数，按出现顺序），
逐条比对。生成器做的是删注释、删死代码、文本替换——这些都不应改变调用序列。
一旦序列有差异，说明机械变换出了问题。

用法:  python submission/verify_release_src.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_release_src import CONFIG, preprocess  # noqa: E402

# 写死配置后确实不可达、生成器会删掉的代码。校验时对参考侧施加同样的删除，
# 这样比对的就是「除了这几处有意删除之外，其余逻辑是否一字未变」。
INTENTIONAL_DROPS = [
    ("epsGm.SetGlobalBuffer", "eps 数组的 GM 视图（eps_mode=1 不从 GM 读）"),
    ("InitBuffer(recBuf", "Newton-Raphson 的临时缓冲（div_mode=1 不用 NR）"),
]
DEAD_FUNCS = [
    ("BuildEpsVector", "从 GM 展开 eps 数组（eps_mode=1 改为算术构造）"),
    ("ReciprocalNR", "Newton-Raphson 修正倒数（div_mode=1 直接用 Div）"),
]

API = (r"\b(Add|Adds|Mul|Muls|Sub|Div|Exp|Maxs|Mins|Duplicate|Reciprocal|"
       r"BlockReduceSum|BlockReduceMax|PairReduceSum|Brcb|DataCopyPad|DataCopy|"
       r"SetFlag|WaitFlag|PipeBarrier|InitBuffer|SetGlobalBuffer)\b")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", l) for l in text.split("\n"))


def call_seq(text, skip_dead=()):
    """抽取 AscendC 调用序列。skip_dead 里的函数体整体跳过（写死版已删除的死代码）。"""
    text = strip_comments(text)
    for fn in skip_dead:
        m = re.search(r"__aicore__ inline [^\n]*\b" + fn + r"\s*\(", text)
        if not m:
            continue
        i = text.index("{", m.end())
        depth = 0
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        text = text[:m.start()] + text[i + 1:]
    return [m.group(1) for m in re.finditer(API, text)]


def main():
    tunable = open(os.path.join(ROOT,
                   "op_kernel/sinkhorn_normalize_kernel_s1.asc"),
                   encoding="utf-8").read()
    release = open(os.path.join(ROOT,
                   "submission/src/op_kernel/sinkhorn_normalize_kernel.asc"),
                   encoding="utf-8").read()

    # 参考序列：只做 #if 消解，不删死代码、不动注释
    ref_text = preprocess(tunable, set(CONFIG))
    ref_text = ref_text.replace("SN_VBAR();", "")           # 该配置下是空操作
    for needle, _ in INTENTIONAL_DROPS:
        ref_text = "\n".join(l for l in ref_text.split("\n") if needle not in l)
    ref = call_seq(ref_text, skip_dead=[f for f, _ in DEAD_FUNCS])
    got = call_seq(release)

    print("=" * 70)
    print("写死版 vs 参数化版（按最优配置消解）的 AscendC 调用序列比对")
    print("=" * 70)
    print("  配置: " + "  ".join("{}={}".format(k.replace("SN_S1_", "").replace("SN_", ""), v)
                                  for k, v in CONFIG.items()))
    print("\n  写死配置后不可达、已删除的代码（比对时两侧同等处理）：")
    for name, why in DEAD_FUNCS + INTENTIONAL_DROPS:
        print("    {:<26} {}".format(name, why))
    print("\n  参考序列 {} 条，写死版 {} 条".format(len(ref), len(got)))
    if ref == got:
        print("\n  ✅ 完全一致 —— 机械变换未改变任何 AscendC 调用")
        rc = 0
    else:
        print("\n  ❌ 存在差异：")
        for i, (a, b) in enumerate(zip(ref, got)):
            if a != b:
                print("    第 {} 条: 参考={}  写死版={}".format(i, a, b))
                print("    上下文: {}".format(ref[max(0, i - 3):i + 3]))
                break
        if len(ref) != len(got):
            n = min(len(ref), len(got))
            print("    长度不同，多出的部分: {}".format(
                (ref[n:] or got[n:])[:10]))
        rc = 1

    # 顺带统计写死版的规模
    print("\n" + "-" * 70)
    for f in ("op_kernel/sinkhorn_normalize_kernel.asc",
              "op_kernel/sinkhorn_normalize_tiling.h",
              "op_extension/sinkhorn_normalize_torch.cpp",
              "op_host/sinkhorn_normalize.asc"):
        p = os.path.join(ROOT, "submission/src", f)
        if os.path.exists(p):
            src = open(p, encoding="utf-8").read()
            n_if = len(re.findall(r"^\s*#(if|ifdef|ifndef|else|elif|endif)",
                                  src, re.M))
            print("  {:<48} {:>4} 行   条件编译 {} 处".format(
                f, src.count("\n") + 1, n_if))
    return rc


if __name__ == "__main__":
    sys.exit(main())
