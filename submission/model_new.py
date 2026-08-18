"""SinkhornNormalize —— KernelSwift 赛道二 Task03 提交文件。

【重要】官方评测脚本 auto_bench.py 会对本文件做 AST 过滤后再 exec，
只保留 Import / ImportFrom / ClassDef / FunctionDef / AsyncFunctionDef / 字面量赋值。
=> 顶层的 `torch.ops.load_library(...)` 这类函数调用会被**静默丢弃**。
因此 .so 的加载全部放在函数体 / __init__ 内部，本文件顶层不含任何可执行语句。

.so 路径查找顺序：
  1. 环境变量 SINKHORN_OPS_SO   （指向 .so 文件本身，最可靠）
  2. 环境变量 SINKHORN_OPS_DIR  （指向所在目录）
  3. 当前工作目录及其 build/ 子目录
  4. 本文件所在目录及其 build/ 子目录（若 __file__ 可用）
  5. sys.path 各条目及其 build/ 子目录
"""

import os
import sys
import torch
import torch.nn as nn
import torch_npu


def _so_name():
    return "libsinkhorn_normalize_ops.so"


def _candidate_paths():
    """返回所有待尝试的 .so 绝对路径（按优先级）。"""
    name = _so_name()
    cands = []

    explicit = os.environ.get("SINKHORN_OPS_SO")
    if explicit:
        cands.append(explicit)

    dirs = []
    env_dir = os.environ.get("SINKHORN_OPS_DIR")
    if env_dir:
        dirs.append(env_dir)

    cwd = os.getcwd()
    dirs += [cwd, os.path.join(cwd, "build")]

    # AST 过滤 + exec 后 __file__ 未必存在，必须防御性获取
    here = globals().get("__file__")
    if here:
        d = os.path.dirname(os.path.abspath(here))
        dirs += [d, os.path.join(d, "build"), os.path.join(os.path.dirname(d), "build")]

    for p in sys.path:
        if p:
            dirs += [p, os.path.join(p, "build")]

    seen = set()
    for d in dirs:
        full = os.path.join(d, name)
        if full not in seen:
            seen.add(full)
            cands.append(full)
    return cands


def _op_is_registered():
    try:
        return hasattr(torch.ops.npu, "sinkhorn_normalize")
    except Exception:                                    # noqa: BLE001
        return False


def _load_custom_op():
    """加载自定义算子 .so。幂等。失败时抛异常——绝不静默 fallback 到内置算子。"""
    if getattr(_load_custom_op, "_done", False):
        return
    if _op_is_registered():
        _load_custom_op._done = True
        return

    tried = []
    for path in _candidate_paths():
        if not os.path.isfile(path):
            tried.append(path + "  [不存在]")
            continue
        try:
            torch.ops.load_library(path)
            if _op_is_registered():
                _load_custom_op._done = True
                _load_custom_op._path = path
                return
            tried.append(path + "  [已加载但未注册 npu::sinkhorn_normalize]")
        except Exception as e:                           # noqa: BLE001
            tried.append("{}  [{}: {}]".format(path, type(e).__name__, e))

    raise RuntimeError(
        "无法加载自定义算子 {}。请设置环境变量 SINKHORN_OPS_SO 指向该文件。\n"
        "已尝试的路径：\n  {}".format(_so_name(), "\n  ".join(tried))
    )


class Model(nn.Module):
    """赛题给定的参考实现（保留以满足提交规范 5.2）。"""

    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.softmax(-1) + self.eps
        x = x / (x.sum(-2, keepdim=True) + self.eps)
        for _ in range(self.repeat - 1):
            x = x / (x.sum(-1, keepdim=True) + self.eps)
            x = x / (x.sum(-2, keepdim=True) + self.eps)
        return x


class ModelNew(nn.Module):
    """昇腾 AscendC 自定义算子实现。__init__ / forward 签名与 Model 完全一致。"""

    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps
        # 在这里加载，而不是模块顶层——顶层调用会被 AST 过滤删掉
        _load_custom_op()
        self._op = torch.ops.npu.sinkhorn_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 快路径：已在 NPU 上，直接调用，不做任何多余操作
        if x.device.type == "npu":
            return self._op(x, self.repeat, self.eps)
        # 防御路径：评测框架若喂 CPU 张量，搬到 NPU 计算后搬回原设备
        dev = x.device
        return self._op(x.npu().contiguous(), self.repeat, self.eps).to(dev)


def get_inputs():
    n0 = 1
    n1 = 1024
    mhc = 4
    x = torch.randn(n0, n1, mhc, mhc)
    return [x]


def get_init_inputs():
    return []
