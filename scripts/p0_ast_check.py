"""P0-①：验证提交文件能扛住 auto_bench.py 的 AST 过滤。

复刻 auto_bench.py 的 _filter_module_ast：只保留
    Import / ImportFrom / ClassDef / FunctionDef / AsyncFunctionDef
    + 值为"安全字面量"的 Assign / AnnAssign

本脚本在 Mac / 纯 CPU 上即可运行，不需要 NPU。
用法：  python scripts/p0_ast_check.py [submission/model_new.py]
"""

import ast
import os
import sys
import types

DEFAULT_TARGET = os.path.join("submission", "model_new.py")
REQUIRED_NAMES = ["Model", "ModelNew", "get_inputs", "get_init_inputs"]


def _is_safe_literal(node):
    """auto_bench 未公开该函数实现，这里用最自然的等价写法（literal_eval 可解析即安全）。
    真实实现可能更严格——本检查偏保守，通过这里不代表一定通过官方。"""
    try:
        ast.literal_eval(node)
        return True
    except Exception:                                    # noqa: BLE001
        return False


def filter_module_ast(tree):
    kept, dropped = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            kept.append(node)
        elif isinstance(node, ast.Assign) and _is_safe_literal(node.value):
            kept.append(node)
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
              and _is_safe_literal(node.value)):
            kept.append(node)
        else:
            dropped.append(node)
    new_tree = ast.Module(body=kept, type_ignores=[])
    return new_tree, kept, dropped


def describe(node):
    name = getattr(node, "name", None)
    if name:
        return "{} {}".format(type(node).__name__, name)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "{} {}".format(type(node).__name__,
                              ", ".join(a.name for a in node.names))
    try:
        src = ast.unparse(node)
    except Exception:                                    # noqa: BLE001
        src = "<无法反解析>"
    return "{}: {}".format(type(node).__name__, src[:80])


def install_npu_stub():
    """仅在环境里**真的没有** torch_npu 时才塞桩（Mac 本地）。

    真机上 torch_npu 是存在的：如果这里把 sys.modules["torch_npu"] 覆盖成假模块，
    torch 的 device backend autoload 会拿到不完整的模块并抛
    "Failed to load the backend extension: torch_npu"。
    返回 True 表示装了桩。"""
    if "torch_npu" in sys.modules:
        return False
    try:
        import torch  # noqa: F401
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import torch_npu  # noqa: F401
        return False
    except Exception:                                    # noqa: BLE001
        sys.modules["torch_npu"] = types.ModuleType("torch_npu")
        return True


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    if not os.path.isfile(target):
        print("找不到文件: {}".format(target))
        return 1

    src = open(target, encoding="utf-8").read()
    tree = ast.parse(src, filename=target)
    new_tree, kept, dropped = filter_module_ast(tree)

    print("=" * 72)
    print("AST 过滤检查: {}".format(target))
    print("=" * 72)
    print("\n[保留] {} 个顶层节点".format(len(kept)))
    for n in kept:
        print("   + {}".format(describe(n)))

    print("\n[丢弃] {} 个顶层节点".format(len(dropped)))
    for n in dropped:
        print("   - 第{}行  {}".format(n.lineno, describe(n)))
    if not dropped:
        print("   （无）")
    else:
        print("   注：模块 docstring 也是 Expr 节点，被丢弃属正常，无副作用")

    problems = []
    for n in dropped:
        # 模块/类 docstring 也是 Expr，会被丢弃，但无害——跳过
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, str):
            continue
        txt = ""
        try:
            txt = ast.unparse(n)
        except Exception:                                # noqa: BLE001
            pass
        problems.append("第{}行的顶层可执行语句被丢弃：{}".format(n.lineno, txt[:80]))

    # ---- 用过滤后的 AST 真正 exec 一遍 ----
    print("\n" + "-" * 72)
    print("在隔离命名空间中 exec 过滤后的代码（不提供 __file__，模拟最坏情况）")
    print("-" * 72)
    stubbed = install_npu_stub()
    print("torch_npu: {}".format("使用桩（本机无 torch_npu）" if stubbed else "使用真实模块"))
    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, filename="<filtered>", mode="exec")
    ns = {"__name__": "submitted_module"}                 # 故意不放 __file__
    try:
        exec(code, ns)                                    # noqa: S102
        print("exec 成功")
    except Exception as e:                                # noqa: BLE001
        print("exec 失败: {}: {}".format(type(e).__name__, e))
        return 1

    missing = [n for n in REQUIRED_NAMES if n not in ns]
    print("\n必需符号检查: {}".format(
        "全部存在" if not missing else "缺失 {}".format(missing)))
    if missing:
        problems.append("过滤后缺失符号: {}".format(missing))

    # ---- 参考 Model 在 CPU 上应可直接跑 ----
    try:
        import torch
        torch.manual_seed(0)
        inputs = ns["get_inputs"]()
        m = ns["Model"](*ns["get_init_inputs"]())
        y = m(*inputs)
        print("参考 Model 前向: OK, 输出 shape={}, dtype={}".format(
            tuple(y.shape), y.dtype))
    except Exception as e:                                # noqa: BLE001
        print("参考 Model 前向失败: {}: {}".format(type(e).__name__, e))
        problems.append("参考 Model 跑不起来")

    # ---- 缺 .so 时必须响亮报错，绝不能静默 fallback（规则 5.1）----
    print("\n" + "-" * 72)
    print("反 fallback 检查：.so 不存在时 ModelNew() 必须抛异常")
    print("-" * 72)
    saved = {k: os.environ.pop(k) for k in ("SINKHORN_OPS_SO", "SINKHORN_OPS_DIR")
             if k in os.environ}
    try:
        ns["ModelNew"]()
        print("危险：没有 .so 也构造成功了 —— 存在静默 fallback 的风险")
        problems.append("ModelNew 在缺少 .so 时未报错")
    except RuntimeError as e:
        first = str(e).splitlines()[0]
        print("正确抛出 RuntimeError: {}".format(first))
    except Exception as e:                                # noqa: BLE001
        print("抛出了 {}（不是 RuntimeError，但至少没有静默 fallback）: {}".format(
            type(e).__name__, str(e).splitlines()[0]))
    finally:
        os.environ.update(saved)

    print("\n" + "=" * 72)
    if problems:
        print("结论：有 {} 个问题需要处理".format(len(problems)))
        for p in problems:
            print("  ! {}".format(p))
        return 1
    print("结论：提交文件可以扛住 AST 过滤")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
