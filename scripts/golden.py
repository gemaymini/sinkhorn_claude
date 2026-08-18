"""Golden computation for SinkhornNormalize, matching the reference PyTorch op."""

import torch
import torch.nn as nn


class SinkhornNormalize(nn.Module):
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


def compute_golden(x, repeat=10, eps=1e-6):
    """Compute the golden SinkhornNormalize output for input x.

    x can be a torch.Tensor or numpy array; we convert to torch for the
    reference implementation.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    model = SinkhornNormalize(repeat=repeat, eps=eps)
    with torch.no_grad():
        y = model(x)
    return y
