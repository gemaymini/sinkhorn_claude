# SinkhornNormalize Ascend C Operator

## Reference PyTorch operator

```python
class Model(nn.Module):
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
```

Reference shape: `x = torch.randn(1, 1024, 4, 4)` (float32).

## Target hardware

- Huawei Ascend 910B2 (Atlas A2 series, `dav-2201`)
- CANN 9.0.0

## Directory layout

```
workspace/
├── op_kernel/                       NPU compute layer
│   ├── sinkhorn_normalize_tiling.h     Tiling constants + struct (kernel+host shared)
│   └── sinkhorn_normalize_kernel.asc    Pure kernel: KernelSinkhorn class + kernel entry
├── op_host/                         Host direct-invoke layer
│   ├── sinkhorn_normalize.asc          Host + main entry (includes kernel.asc)
│   └── data_utils.h                    File I/O helpers (copied from add_custom template)
├── op_extension/                    PyTorch bindings
│   ├── sinkhorn_normalize_torch.cpp    torch op host impl (Tiling compute + launch)
│   ├── register.cpp                    TORCH_LIBRARY registration (+Meta impl)
│   └── ops.h                           Function declaration
├── scripts/
│   ├── golden.py                       Reference SinkhornNormalize torch op
│   ├── gen_data.py                      Generate input_x.bin + golden.bin
│   ├── verify_result.py                Verify direct-invoke output against golden
│   ├── test_torch.py                    Verify PyTorch path on multiple shapes
│   └── perf_test.py                    Baseline vs kernel perf comparison
├── CMakeLists.txt                   Two targets: executable + libsinkhorn_normalize_ops.so
├── run.sh                           Full pipeline: build → gen → run → verify → torch test
└── README.md                        (this file)
```

## Build and test

```bash
# Set up CANN env (only needed once per shell)
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# Full pipeline: build + gen data + run kernel + verify + torch test
bash run.sh

# Skip build (reuse prior build/)
bash run.sh --skip-build

# Run only the PyTorch path (requires prior build)
bash run.sh --torch

# Performance test (after build)
cd build && python3 ../scripts/perf_test.py
```

## Design highlights

- Per-matrix UB layout: 4 row-groups × 8 fp32 = 32 elements (4 valid + 4 padding per row).
- Row reductions: `BlockReduceSum`/`BlockReduceMax` with `mask=4` reduce only the first 4 elements per 8-element block, giving 4 row sums/maxes at positions 0, 8, 16, 24. `Brcb` broadcasts each result to its 8-element block; `Div` then applies row-wise normalization. Padding never contaminates row reductions because of the mask.
- Column reductions: three strided `Add` calls compute col sums at `tmp[0..3]`; padding sums land in `tmp[4..7]` (unused). The divisor tensor is built by copying `[col_sum+eps, col_sum+eps, col_sum+eps, col_sum+eps, 1.0, 1.0, 1.0, 1.0]` to each row group (so padding positions divide by 1.0 = unchanged).
- All 10 sinkhorn iterations run in UB; only the initial load and final store touch GM per matrix.
