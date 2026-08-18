"""Verify SinkhornNormalize kernel output against golden."""

import numpy as np
import sys

dtype = np.float32
rtol = 1e-4
atol = 1e-5


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    abs_diff = np.abs(output - golden)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {max_diff}, Mean diff: {mean_diff}")
        return True
    else:
        print(f"Verification FAILED!")
        print(f"Max diff: {max_diff}, Mean diff: {mean_diff}")
        rel_diff = abs_diff / (np.abs(golden) + 1e-12)
        print(f"Max rel diff: {np.max(rel_diff)}")
        mismatches = np.where(abs_diff > atol + rtol * np.abs(golden))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            idx = mismatches[0]
            print(f"First mismatch at idx {idx}: output={output[idx]}, golden={golden[idx]}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
