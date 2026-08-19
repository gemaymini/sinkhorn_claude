"""提交件合规自检：逐条对照官方规则与 auto_bench.py 的实际行为。

静态检查在任何机器上都能跑（Mac 亦可）；带 --so 时额外做运行时检查。

用法:
    python check_compliance.py                 # 静态检查
    python check_compliance.py --so ./libsinkhorn_normalize_ops.so   # 全量
"""

import argparse
import ast
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
# 三种布局都支持：
#   打包后        算子代码在包内根目录（HERE）
#   开发仓库      写死版在 submission/src/，开发版（带开关）在仓库根目录
# 顺序即优先级：先看包内，再看写死版，最后才是仓库根目录
ROOTS = [HERE, os.path.join(HERE, "src"), os.path.dirname(HERE)]


def _stub_torch_npu_if_absent():
    """本机（如 Mac）没有 torch_npu 时装个桩，好让静态检查照常进行。
    真机上有真模块就绝不覆盖——覆盖会让 torch 的 backend autoload 拿到假模块而报错。"""
    if "torch_npu" in sys.modules:
        return False
    try:
        import torch  # noqa: F401
    except Exception:                                      # noqa: BLE001
        pass
    try:
        import torch_npu  # noqa: F401
        return False
    except Exception:                                      # noqa: BLE001
        sys.modules["torch_npu"] = types.ModuleType("torch_npu")
        return True
OK, WARN, BAD = "[通过]", "[注意]", "[不符]"
_results = []


def check(cond, name, detail="", warn_only=False):
    tag = OK if cond else (WARN if warn_only else BAD)
    _results.append((tag, name, detail))
    print("  {} {:<46} {}".format(tag, name, detail))
    return cond


# ---------------------------------------------------------------- AST 过滤
def _is_safe_literal(node):
    """与官方 auto_bench.py 逐字一致。"""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all((k is None or _is_safe_literal(k)) and _is_safe_literal(v)
                   for k, v in zip(node.keys, node.values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_literal(node.operand)
    return False


def filtered_exec(path):
    """完全复刻官方 load_ks_module 的加载流程。"""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    kept, dropped = [], []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef,
                          ast.FunctionDef, ast.AsyncFunctionDef)):
            kept.append(n)
        elif isinstance(n, ast.Assign) and _is_safe_literal(n.value):
            kept.append(n)
        elif isinstance(n, ast.AnnAssign) and n.value is not None \
                and _is_safe_literal(n.value):
            kept.append(n)
        else:
            dropped.append(n)
    tree.body = kept
    ast.fix_missing_locations(tree)
    mod = types.ModuleType("_ks_" + os.path.basename(path))
    mod.__file__ = path                       # 官方确实设置了这一项
    exec(compile(tree, path, "exec"), mod.__dict__)   # noqa: S102
    return mod, dropped


def forward_ast(path, cls_name):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == cls_name:
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name == "forward":
                    return m
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=None, help="提供则额外做运行时检查")
    ap.add_argument("--root", default=None,
                    help="被检查的目录（默认本文件所在目录）。"
                         "打包流程用它检查包目录，从而无需把本脚本放进提交件")
    ap.add_argument("--preflight", action="store_true",
                    help="性能测试前预检：暂不要求 results/performance.txt，"
                         "该文件生成后必须再执行一次完整检查")
    args = ap.parse_args()

    global HERE, ROOTS
    if args.root:
        HERE = os.path.abspath(args.root)
        # 显式检查打包目录时不能回退到父目录，否则父目录中的同名文件会让
        # 一个内容不完整的提交件误通过。
        ROOTS = [HERE, os.path.join(HERE, "src")]

    v1 = os.path.join(HERE, "model_new.py")
    v0 = os.path.join(HERE, "model_ref.py")

    stubbed = _stub_torch_npu_if_absent()
    if stubbed:
        print("注意：本机无 torch_npu，已装桩；运行时检查请在真机上带 --so 重跑\n")

    print("=" * 78)
    print("规则 5.2 · 代码要求")
    print("=" * 78)
    for p in (v0, v1):
        try:
            open(p, encoding="utf-8").read()
            enc = True
        except (UnicodeDecodeError, OSError):
            enc = False
        check(enc, "文件编码为 UTF-8", os.path.basename(p))
    check(sys.version_info[:2] >= (3, 10), "Python >= 3.10",
          "当前 {}.{}".format(*sys.version_info[:2]), warn_only=True)

    print()
    print("=" * 78)
    print("auto_bench.py 兼容性（AST 过滤 + 符号 + 签名）")
    print("=" * 78)
    mods = {}
    # 官方 build_case(): Model 只从 v0 取，ModelNew 只从 v1 取，两个文件各自都要
    # get_init_inputs / get_inputs
    for path, role, need in ((v0, "v0", ["Model", "get_inputs", "get_init_inputs"]),
                             (v1, "v1", ["ModelNew", "get_inputs", "get_init_inputs"])):
        name = os.path.basename(path)
        try:
            mod, dropped = filtered_exec(path)
            mods[role] = mod
            check(True, "{} 过滤后可 exec".format(name))
        except Exception as e:                             # noqa: BLE001
            check(False, "{} 过滤后可 exec".format(name),
                  "{}: {}".format(type(e).__name__, e))
            continue
        check(all(hasattr(mod, s) for s in need),
              "{} 必需符号齐全".format(name), str(need))
        real = [n for n in dropped
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str))]
        check(not real, "{} 顶层无被丢弃的可执行语句".format(name),
              "docstring 被丢弃属正常" if dropped else "")
        for fn in ("get_inputs", "get_init_inputs"):
            if hasattr(mod, fn):
                v = getattr(mod, fn)()
                check(v is None or isinstance(v, (list, tuple)),
                      "{}.{}() 返回 list/tuple/None".format(name, fn),
                      type(v).__name__)

    if "v0" in mods and "v1" in mods:
        m, mn = mods["v0"].Model, mods["v1"].ModelNew
        for meth in ("__init__", "forward"):
            try:
                s0 = inspect.signature(getattr(m, meth))
                s1 = inspect.signature(getattr(mn, meth))
                same = list(s0.parameters) == list(s1.parameters)
                check(same, "ModelNew.{} 参数与 Model 一致".format(meth),
                      "{} vs {}".format(list(s0.parameters), list(s1.parameters)))
            except Exception as e:                         # noqa: BLE001
                check(False, "ModelNew.{} 签名比对".format(meth), str(e))
        d0 = mods["v0"].get_init_inputs()
        d1 = mods["v1"].get_init_inputs()
        check(list(d0 or []) == list(d1 or []),
              "两文件 get_init_inputs() 一致", "{} vs {}".format(d0, d1))

    print()
    print("=" * 78)
    print("规则 5.2 · 签名一致性与职责分离")
    print("=" * 78)
    # 规则 5.2 要求的是「与参考实现一致的 Model 定义，包括 __init__/forward 的参数」，
    # 即**签名一致**，不是把参考实现抄进提交文件。
    # auto_bench 的 build_case() 只从 v0 取 Model、只从 v1 取 ModelNew，互不干涉。
    v1_src = open(v1, encoding="utf-8").read()
    v1_tree = ast.parse(v1_src)
    v1_classes = [n.name for n in v1_tree.body if isinstance(n, ast.ClassDef)]
    check("Model" not in v1_classes,
          "提交文件不重复定义 Model", "顶层类: {}".format(v1_classes))
    check("softmax" not in v1_src,
          "提交文件不含 PyTorch 等价实现",
          "放一份可用的 torch 实现会构成规则 5.1 的嫌疑面")
    check("os.environ" not in v1_src and "getenv" not in v1_src,
          "提交文件不依赖环境变量", ".so 固定从本文件所在目录加载")

    print()
    print("=" * 78)
    print("规则 5.1 · 反 fallback（每条返回路径都必须走自定义算子）")
    print("=" * 78)
    fw = forward_ast(v1, "ModelNew")
    if fw is None:
        check(False, "找到 ModelNew.forward")
    else:
        rets = [n for n in ast.walk(fw) if isinstance(n, ast.Return)]
        def calls_custom(node):
            src = ast.unparse(node)
            return ("self._op" in src or "torch.ops.npu" in src)
        check(rets and all(calls_custom(r) for r in rets),
              "forward 的所有 return 都调用自定义算子",
              "{} 条返回路径".format(len(rets)))
        src = ast.unparse(fw)
        check("softmax" not in src and "sum(" not in src,
              "forward 内不含 PyTorch 内置实现的痕迹")
        check(not any(isinstance(n, (ast.Try, ast.ExceptHandler))
                      for n in ast.walk(fw)),
              "forward 内无异常捕获（规则明令禁止）")

    print()
    print("=" * 78)
    print("规则 4.2 · 提交内容完整性")
    print("=" * 78)
    need_files = {
        "算子优化代码": ["op_kernel", "op_extension", "op_host", "CMakeLists.txt"],
        "README 文档": ["README.md"],
        "环境配置文件": ["requirements.txt"],
        "运行脚本": ["build.sh", "run_auto_bench.sh"],
        "提交入口": ["model_new.py", "model_ref.py"],
        "性能测试结果": ["results/performance.txt"],
    }
    if args.preflight:
        need_files.pop("性能测试结果")
    def _find(f):
        for r in ROOTS:
            if os.path.exists(os.path.join(r, f)):
                return r
        return None

    for label, fs in need_files.items():
        miss = [f for f in fs if _find(f) is None]
        where = ""
        if not miss and fs:
            r = _find(fs[0])
            where = "（在上级目录，打包时复制进来）" if r != HERE else ""
        check(not miss, label,
              "缺少 {}".format(miss) if miss else ", ".join(fs) + where)
    if args.preflight:
        print("  [延后] 性能测试结果                               "
              "将在性能测试完成后执行完整检查")
    # 提交件不应包含开发期的诊断日志
    res = os.path.join(HERE, "results")
    extra = []
    if os.path.isdir(res):
        allow = {"performance.txt", "precision.txt"}
        extra = [f for f in os.listdir(res) if f not in allow]
    check(not extra, "results/ 只含官方要求的测试结果",
          "多余文件 {}".format(extra) if extra else "performance.txt / precision.txt")
    check(not os.path.isdir(os.path.join(HERE, "scripts")),
          "提交件不含开发工具脚本", "scripts/ 目录应只存在于开发仓库")

    print()
    print("=" * 78)
    print("构建自包含性（CMakeLists 引用的源文件是否都在包内）")
    print("=" * 78)
    cml = None
    for r in ROOTS:
        c = os.path.join(r, "CMakeLists.txt")
        if os.path.exists(c):
            cml = c
            break
    if cml is None:
        check(False, "找到 CMakeLists.txt")
    else:
        import re
        txt = open(cml, encoding="utf-8").read()
        base = os.path.dirname(cml)
        # set(VAR path) 形式的变量，用于解析 ${VAR}
        vs = dict(re.findall(r"set\((SN_[A-Z_]*SRC)\s+([^\s)]+)\)", txt))
        # if(EXISTS ...) 保护块内的引用不算硬依赖
        guarded = set(re.findall(r"if\(EXISTS\s+\$\{CMAKE_CURRENT_SOURCE_DIR\}/([^\s)]+)", txt))
        missing = []
        for m in re.finditer(r"add_(?:executable|library)\s*\(([^)]*)\)", txt, re.S):
            for tok in m.group(1).split():
                if tok.startswith("${"):
                    tok = vs.get(tok[2:-1], tok)
                if not re.search(r"\.(asc|cpp|cc|c)$", tok):
                    continue
                if tok in guarded:
                    continue
                if not os.path.exists(os.path.join(base, tok)):
                    missing.append(tok)
        check(not missing, "CMakeLists 引用的源文件均存在",
              "缺少 {}".format(missing) if missing
              else "已跳过 if(EXISTS) 保护的 {} 项".format(len(guarded)))

    print()
    print("=" * 78)
    print("构建可复现性（提交件不应含任何编译期配置开关）")
    print("=" * 78)
    import re as _re
    srcs, sw = [], []
    for r in ROOTS:
        for sub in ("op_kernel", "op_extension", "op_host"):
            d = os.path.join(r, sub)
            if os.path.isdir(d):
                srcs += [os.path.join(d, f) for f in os.listdir(d)]
        c = os.path.join(r, "CMakeLists.txt")
        if os.path.exists(c):
            srcs.append(c)
        if srcs:
            break
    for f in srcs:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:                                  # noqa: BLE001
            continue
        for m in _re.finditer(r"\bSN_(?:S1_[A-Z_]+|KERNEL_VARIANT|TILING_MODE|"
                              r"CORE_QUERY|DIAG)\b", t):
            sw.append("{}:{}".format(os.path.basename(f), m.group(0)))
    check(not sw, "源码与 CMakeLists 无编译期开关",
          "残留 {}".format(sorted(set(sw))[:4]) if sw
          else "算子配置已固化，cmake 无需任何 -D")

    print()
    print("=" * 78)
    print("规则 4.3 · README 必备章节")
    print("=" * 78)
    rd = os.path.join(HERE, "README.md")
    text = open(rd, encoding="utf-8").read() if os.path.exists(rd) else ""
    for sec in ("作品说明", "优化方案", "性能测试结果", "原创声明"):
        check(sec in text, "README 含「{}」".format(sec))

    if args.so:
        print()
        print("=" * 78)
        print("运行时检查（需要 NPU）")
        print("=" * 78)
        check(os.path.isfile(args.so), ".so 存在", args.so)
        check(os.path.dirname(os.path.abspath(args.so)) == HERE,
              ".so 与 model_new.py 同目录", "这是唯一的加载路径，不同目录会直接失败")
        try:
            import torch
            import torch_npu  # noqa: F401
            torch.ops.load_library(args.so)
            check(hasattr(torch.ops.npu, "sinkhorn_normalize"), "算子注册成功")
            mod, _ = filtered_exec(v1)
            m = mod.ModelNew()
            x = torch.randn(1, 1024, 4, 4).npu()
            before = int(torch.ops.npu.sinkhorn_normalize_call_count())
            y = m.forward(x)
            after = int(torch.ops.npu.sinkhorn_normalize_call_count())
            check(after > before, "forward 真实进入自定义 kernel",
                  "计数 {} -> {}".format(before, after))
            ref = mods["v0"].Model().npu()
            r = ref.forward(x)
            d = (r - y).abs().max().item()
            tol = 1e-2 + 1e-2 * r.abs().max().item()
            check(torch.allclose(r, y, atol=1e-2, rtol=1e-2, equal_nan=True),
                  "官方容差下结果一致",
                  "max_diff={:.3e}  裕度占用={:.5f}".format(d, d / tol))
        except Exception as e:                             # noqa: BLE001
            check(False, "运行时检查", "{}: {}".format(type(e).__name__, e))

    bad = [r for r in _results if r[0] == BAD]
    warn = [r for r in _results if r[0] == WARN]
    print()
    print("=" * 78)
    print("合计 {} 项检查：{} 通过，{} 注意，{} 不符".format(
        len(_results), len(_results) - len(bad) - len(warn), len(warn), len(bad)))
    for t, n, d in bad:
        print("  {} {} {}".format(t, n, d))
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
