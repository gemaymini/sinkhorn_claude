"""msprof 的采集目标：只跑自定义算子，尽量少的无关噪声。

    msprof --output=./prof_p0 --application="python3 scripts/p0_msprof_target.py --so <path>"

采完后在 prof_p0/**/op_summary_*.csv 里找 sinkhorn_normalize_kernel 的
Task Duration，那才是**纯 kernel 时间**；把它和 p0_host_breakdown.py 的
M1 / M2 对上，就能得到 kernel / host / launch 的完整拆解。
"""

import argparse
import torch
import torch_npu  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    torch.ops.load_library(args.so)
    op = torch.ops.npu.sinkhorn_normalize

    torch.manual_seed(42)
    x = torch.randn(1, 1024, 4, 4, dtype=torch.float32).npu()

    with torch.no_grad():
        for _ in range(args.warmup):
            op(x, 10, 1e-6)
        torch.npu.synchronize()
        for _ in range(args.iters):
            op(x, 10, 1e-6)
        torch.npu.synchronize()
    print("done: {} iters".format(args.iters))


if __name__ == "__main__":
    main()
