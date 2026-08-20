#!/usr/bin/env python3
"""在 NPU 上按官方 auto_bench.py 口径做 SinkhornNormalize 多随机种子精度测试。

本脚本复用官方 auto_bench.py 的模块加载、设备选择、输入克隆和
compare_values，因此 PASS/FAIL 判定与官方的
torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True) 一致。

每个种子只执行精度 forward，不做 warmup/repeat 性能循环。默认测试 0~99
共 100 个种子，并将逐种子误差、最大容差占用率写入 JSON 报告。

示例：
  python run_auto_bench_mulseed.py
  python run_auto_bench_mulseed.py --seeds 0-999
  python run_auto_bench_mulseed.py --seeds 0-99,2026 --fail-fast
  python run_auto_bench_mulseed.py --auto-bench /path/to/auto_bench.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch


OFFICIAL_AUTO_BENCH_URL = (
    "https://raw.githubusercontent.com/DeepLink-org/DLBlas/"
    "main/benchmarks/ks/auto_bench.py"
)
DEFAULT_SEEDS = "0-99"
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def parse_seed_spec(spec: str) -> list[int]:
    """解析 ``0-99,2026`` 格式，去重后保留原顺序。"""
    seeds: list[int] = []
    seen: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"无效种子片段 {part!r}；请使用 42 或 0-99,2026 格式"
            )
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise argparse.ArgumentTypeError(f"种子范围必须递增: {part!r}")
        for seed in range(first, last + 1):
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError("至少需要一个随机种子")
    return seeds


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="在 NPU 上按官方 auto_bench 口径做 SinkhornNormalize 多种子精度测试"
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_spec,
        default=parse_seed_spec(DEFAULT_SEEDS),
        metavar="SPEC",
        help=f"种子列表/范围，如 0-99,2026（默认 {DEFAULT_SEEDS}）",
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument(
        "--auto-bench",
        type=Path,
        default=Path(os.environ.get("AUTO_BENCH", "/tmp/auto_bench.py")),
        help="官方 auto_bench.py 路径（默认 /tmp/auto_bench.py）",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="auto_bench.py 不存在时不自动下载",
    )
    parser.add_argument("--v0-file", type=Path, default=here / "model_ref.py")
    parser.add_argument("--v1-file", type=Path, default=here / "model_new.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "results" / "multiseed_accuracy.json",
        help="JSON 报告路径",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印每个种子的明细，仅打印失败项和汇总",
    )
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol 和 --rtol 必须大于等于 0")
    return args


def ensure_official_auto_bench(path: Path, no_download: bool) -> Path:
    if path.is_file():
        return path.resolve()
    if no_download:
        raise FileNotFoundError(f"找不到官方 auto_bench.py: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"未找到 {path}，正在下载官方 auto_bench.py ...")
    try:
        urllib.request.urlretrieve(OFFICIAL_AUTO_BENCH_URL, path)
    except Exception as exc:  # noqa: BLE001 - 保留原始下载错误
        raise RuntimeError(
            f"下载官方 auto_bench.py 失败: {exc}\n"
            "可手动下载后用 --auto-bench /path/to/auto_bench.py 指定。"
        ) from exc
    return path.resolve()


def load_official_auto_bench(path: Path) -> ModuleType:
    name = "_official_auto_bench_multiseed"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载官方 auto_bench.py: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses 等标准库会根据 __module__ 查找 sys.modules。
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    required = (
        "build_case",
        "_detect_target_device",
        "_move_to_device",
        "clone_value",
        "run_forward",
        "compare_values",
    )
    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise RuntimeError(
            f"{path} 缺少所需接口: {', '.join(missing)}；请更新官方 auto_bench.py"
        )
    return module


def tensor_pairs(
    lhs: Any, rhs: Any, path: str = "output"
) -> Iterable[tuple[str, torch.Tensor, torch.Tensor]]:
    """递归枚举嵌套输出中的 Tensor 对；结构正确性仍由官方函数判定。"""
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        yield path, lhs, rhs
        return
    if isinstance(lhs, (tuple, list)) and isinstance(rhs, type(lhs)):
        for index, (left_item, right_item) in enumerate(zip(lhs, rhs)):
            yield from tensor_pairs(left_item, right_item, f"{path}[{index}]")
        return
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        for key in sorted(set(lhs).intersection(rhs), key=repr):
            yield from tensor_pairs(lhs[key], rhs[key], f"{path}[{key!r}]")


def measure_outputs(lhs: Any, rhs: Any, atol: float, rtol: float) -> dict[str, Any]:
    """在官方 PASS/FAIL 之外，计算误差和容差余量。"""
    element_count = 0
    mismatch_count = 0
    reference_nonfinite_count = 0
    candidate_nonfinite_count = 0
    both_nan_count = 0
    abs_sum = 0.0
    max_abs_diff = 0.0
    max_tolerance_ratio = 0.0
    worst: dict[str, Any] | None = None

    for output_path, lhs_tensor, rhs_tensor in tensor_pairs(lhs, rhs):
        left = lhs_tensor.detach().float().cpu()
        right = rhs_tensor.detach().to(lhs_tensor.device).float().cpu()
        if left.shape != right.shape:
            continue

        close = torch.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
        raw_diff = (left - right).abs()
        # 相同的 +/-inf 在 allclose 中通过，但 inf-inf 会产生 NaN。
        diff = torch.where(close & torch.isnan(raw_diff), 0.0, raw_diff)
        # 官方调用 allclose(lhs, rhs)，公式的相对容差基于 rhs。
        tolerance = atol + rtol * right.abs()
        ratio = diff / tolerance
        ratio = torch.where((diff == 0) & (tolerance == 0), 0.0, ratio)
        ratio = torch.where(close & torch.isnan(ratio), 0.0, ratio)

        count = left.numel()
        element_count += count
        mismatch_count += int((~close).sum().item())
        reference_nonfinite_count += int((~torch.isfinite(left)).sum().item())
        candidate_nonfinite_count += int((~torch.isfinite(right)).sum().item())
        both_nan_count += int((torch.isnan(left) & torch.isnan(right)).sum().item())
        if count == 0:
            continue

        safe_diff = torch.nan_to_num(diff, nan=math.inf, posinf=math.inf, neginf=math.inf)
        safe_ratio = torch.nan_to_num(ratio, nan=math.inf, posinf=math.inf, neginf=math.inf)
        local_abs = float(safe_diff.max().item())
        local_ratio, flat_index_tensor = safe_ratio.reshape(-1).max(dim=0)
        local_ratio_value = float(local_ratio.item())
        abs_sum += float(safe_diff.sum().item())
        max_abs_diff = max(max_abs_diff, local_abs)
        if worst is None or local_ratio_value > max_tolerance_ratio:
            flat_index = int(flat_index_tensor.item())
            max_tolerance_ratio = local_ratio_value
            worst = {
                "output_path": output_path,
                "flat_index": flat_index,
                "reference": float(left.reshape(-1)[flat_index].item()),
                "candidate": float(right.reshape(-1)[flat_index].item()),
                "abs_diff": float(safe_diff.reshape(-1)[flat_index].item()),
                "tolerance_ratio": local_ratio_value,
            }

    return {
        "element_count": element_count,
        "mismatch_count": mismatch_count,
        "reference_nonfinite_count": reference_nonfinite_count,
        "candidate_nonfinite_count": candidate_nonfinite_count,
        "both_nan_count": both_nan_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": abs_sum / element_count if element_count else 0.0,
        "max_tolerance_ratio": max_tolerance_ratio,
        "worst_element": worst,
    }


def run_seed(
    bench: ModuleType,
    v0_path: Path,
    v1_path: Path,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """复用官方 compare_case 的精度路径，仅跳过性能循环。"""
    model, model_new, v0_inputs, v1_inputs = bench.build_case(v0_path, v1_path, seed)
    target_device = bench._detect_target_device(model, model_new, v0_inputs, v1_inputs)
    if target_device.type != "npu":
        raise RuntimeError(f"检测到的设备是 {target_device}，而不是 NPU")

    try:
        model_new.load_state_dict(model.state_dict())
    except Exception:
        pass
    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model_new, "to"):
        model_new = model_new.to(target_device)
    v0_inputs = bench._move_to_device(v0_inputs, target_device)
    # 官方脚本最终也是把 v0 输入克隆给 v1，确保两者逐 bit 同输入。
    v1_inputs = bench.clone_value(v0_inputs)
    v0_output = bench.run_forward(model, v0_inputs, seed, f"seed={seed}: v0")
    v1_output = bench.run_forward(model_new, v1_inputs, seed, f"seed={seed}: v1")

    passed = True
    message = ""
    try:
        bench.compare_values(v0_output, v1_output, "output", atol, rtol)
    except Exception as exc:  # noqa: BLE001 - 官方会抛出自定义异常
        passed = False
        message = str(exc)

    return {
        "seed": seed,
        "passed": passed,
        "message": message,
        "device": str(target_device),
        **measure_outputs(v0_output, v1_output, atol, rtol),
    }


def json_safe(value: Any) -> Any:
    """避免 JSON 中出现非标准 Infinity/NaN 字面量。"""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    seeds: list[int] = args.seeds
    started = time.perf_counter()

    bench_path = ensure_official_auto_bench(args.auto_bench, args.no_download)
    bench = load_official_auto_bench(bench_path)
    v0_path = args.v0_file.resolve()
    v1_path = args.v1_file.resolve()
    for path, label in ((v0_path, "v0"), (v1_path, "v1")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 文件不存在: {path}")

    print("=" * 78)
    print("SinkhornNormalize NPU 多随机种子精度测试")
    print("=" * 78)
    print(f"种子       : {len(seeds)} 个 ({seeds[0]} ... {seeds[-1]})")
    print(f"判定口径   : torch.allclose(atol={args.atol:g}, rtol={args.rtol:g}, equal_nan=True)")
    print(f"auto_bench : {bench_path}")
    print(f"v0         : {v0_path}")
    print(f"v1         : {v1_path}")
    print("-" * 78)

    cases: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        runtime_error = False
        try:
            case = run_seed(bench, v0_path, v1_path, seed, args.atol, args.rtol)
        except Exception as exc:  # noqa: BLE001 - 记录种子级运行失败
            runtime_error = True
            case = {
                "seed": seed,
                "passed": False,
                "message": f"{type(exc).__name__}: {exc}",
                "device": "npu",
                "element_count": 0,
                "mismatch_count": 0,
                "reference_nonfinite_count": 0,
                "candidate_nonfinite_count": 0,
                "both_nan_count": 0,
                "max_abs_diff": math.inf,
                "mean_abs_diff": math.inf,
                "max_tolerance_ratio": math.inf,
                "worst_element": None,
            }
        cases.append(case)
        if not args.quiet or not case["passed"]:
            status = "PASS" if case["passed"] else "FAIL"
            print(
                f"[{index:>{len(str(len(seeds)))}}/{len(seeds)}] "
                f"seed={seed:<10} {status}  "
                f"max_abs={case['max_abs_diff']:.3e}  "
                f"tol_ratio={case['max_tolerance_ratio']:.3e}"
            )
            if case["message"]:
                print(f"  {case['message']}")
        # 环境、加载或 forward 异常通常与种子无关，继续 100 次只会重复同一错误。
        if runtime_error:
            print("遇到运行异常，已停止后续种子测试。")
            break
        if args.fail_fast and not case["passed"]:
            break

    passed_cases = [case for case in cases if case["passed"]]
    failed_cases = [case for case in cases if not case["passed"]]
    worst_case = max(cases, key=lambda case: case["max_tolerance_ratio"])
    max_abs_case = max(cases, key=lambda case: case["max_abs_diff"])
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "test_type": "real_npu_custom_operator_multiseed_accuracy",
        "all_passed": not failed_cases and len(cases) == len(seeds),
        "requested_seed_count": len(seeds),
        "tested_seed_count": len(cases),
        "passed_seed_count": len(passed_cases),
        "failed_seed_count": len(failed_cases),
        "reference_nonfinite_count": sum(
            case["reference_nonfinite_count"] for case in cases
        ),
        "candidate_nonfinite_count": sum(
            case["candidate_nonfinite_count"] for case in cases
        ),
        "both_nan_count": sum(case["both_nan_count"] for case in cases),
        "seeds": seeds,
        "atol": args.atol,
        "rtol": args.rtol,
        "equal_nan": True,
        "worst_seed_by_tolerance_ratio": worst_case["seed"],
        "max_tolerance_ratio": worst_case["max_tolerance_ratio"],
        "worst_seed_by_absolute_error": max_abs_case["seed"],
        "max_abs_diff": max_abs_case["max_abs_diff"],
        "elapsed_seconds": elapsed,
        "auto_bench_path": str(bench_path),
        "auto_bench_url": OFFICIAL_AUTO_BENCH_URL,
        "v0_file": str(v0_path),
        "v1_file": str(v1_path),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 78)
    print(
        f"汇总       : {'PASS' if report['all_passed'] else 'FAIL'}  "
        f"{len(passed_cases)}/{len(cases)} 个已测种子通过"
    )
    print(
        f"最大绝对误差: {max_abs_case['max_abs_diff']:.6e} "
        f"(seed={max_abs_case['seed']})"
    )
    print(
        f"最大容差占用: {worst_case['max_tolerance_ratio']:.6e} "
        f"(seed={worst_case['seed']}; < 1 才通过)"
    )
    print(f"耗时       : {elapsed:.3f} s")
    print(f"报告       : {args.output.resolve()}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
