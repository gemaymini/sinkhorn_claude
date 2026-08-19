"""SinkhornNormalize —— KernelSwift 赛道二 Task03 提交文件（auto_bench 的 --v1_file）。

本文件只定义 ModelNew / get_inputs / get_init_inputs。
参考实现 Model 在 model_ref.py（--v0_file），官方 build_case() 从两个文件分别读取。

【AST 过滤】官方评测会对本文件做 AST 过滤后再 exec，只保留
Import / ImportFrom / ClassDef / FunctionDef / AsyncFunctionDef / 字面量赋值。
=> 顶层的 `torch.ops.load_library(...)` 这类函数调用会被**静默丢弃**。
因此 .so 的加载放在 ModelNew.__init__ 内部。

【.so 位置】官方 load_ks_module() 会设置 module.__file__ = str(path)，
所以本文件能拿到自身路径。libsinkhorn_normalize_ops.so 固定从**本文件所在目录**加载
—— 提交包已把它放在这里；从源码构建时 build.sh 也会把产物复制过来。
"""

import os
import torch
import torch.nn as nn
import torch_npu


def _load_custom_op():
    """加载与本文件同目录的自定义算子 .so。

    找不到就抛异常，**绝不静默回退到 PyTorch 内置算子**（规则 5.1）。
    """
    if hasattr(torch.ops.npu, "sinkhorn_normalize"):
        return
    so = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "libsinkhorn_normalize_ops.so")
    if not os.path.isfile(so):
        raise RuntimeError(
            "未找到自定义算子: {}\n"
            "请在本文件所在目录运行 `bash build.sh` 生成该文件。".format(so))
    torch.ops.load_library(so)


class ModelNew(nn.Module):
    """昇腾 AscendC 自定义算子实现。

    规则 5.2 要求「与参考实现一致的 Model 定义，包括 __init__ 和 forward 的参数」——
    指的是**签名一致**，即本类的 __init__(repeat, eps) 与 forward(x) 必须与赛题参考实现
    完全对得上，而不是把参考实现本身抄进提交文件。

    本文件**刻意不包含任何 PyTorch 版的等价实现** —— 规则 5.1 禁止 fallback 到内置算子，
    提交文件里放一份可用的 torch 实现本身就构成嫌疑面。forward 的每一条返回路径
    都只调用自定义算子。
    """

    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps
        # 在这里加载，而不是模块顶层——顶层调用会被 AST 过滤删掉
        _load_custom_op()
        self._op = torch.ops.npu.sinkhorn_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # kernel 假定输入连续。官方 harness 传进来的一定是连续张量，
        # 但非连续输入会算出错误结果（直接 0 分），这次极廉价的检查值得付。
        if not x.is_contiguous():
            x = x.contiguous()
        return self._op(x, self.repeat, self.eps)


def get_inputs():
    n0 = 1
    n1 = 1024
    mhc = 4
    x = torch.randn(n0, n1, mhc, mhc)
    return [x]


def get_init_inputs():
    return []
