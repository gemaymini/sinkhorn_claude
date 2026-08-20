import os
import torch
import torch.nn as nn
import torch_npu


def _load_custom_op():
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
    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps
        _load_custom_op()
        self._op = torch.ops.npu.sinkhorn_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
