"""Generate test data for SinkhornNormalize kernel."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Reference shape: [1, 1024, 4, 4] float32
N0 = 1
N1 = 1024
MHC = 4
total_mats = N0 * N1
total_elements = total_mats * MHC * MHC
dtype = np.float32

REPEAT = 10
EPS = 1e-6

x = np.random.randn(total_elements).astype(dtype).reshape(N0, N1, MHC, MHC)

x.tofile("input/input_x.bin")

golden = compute_golden(x, repeat=REPEAT, eps=EPS)
golden_np = golden.numpy() if isinstance(golden, __import__('torch').Tensor) else golden
golden_np = np.ascontiguousarray(golden_np, dtype=dtype)
golden_np.tofile("output/golden.bin")

print(f"Generated test data: shape=({N0},{N1},{MHC},{MHC}), totalMats={total_mats}, dtype={dtype}")
print(f"  input/input_x.bin: {x.shape}, {x.dtype}")
print(f"  output/golden.bin: {golden_np.shape}, {golden_np.dtype}")
