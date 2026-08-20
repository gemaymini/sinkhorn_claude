#!/usr/bin/env python3
"""在 NPU 上按官方 auto_bench.py 口径做 SinkhornNormalize 多随机种子精度测试。

本脚本复用官方 auto_bench.py 的模块加载、设备选择、输入克隆和
compare_values，因此 PASS/FAIL 判定与官方的
torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True) 一致。

每个种子只执行精度 forward，不做 warmup/repeat 性能循环。种子从
--start-seed 开始逐一递增，无上限地持续测试，直到用户按 Ctrl+C。
逐种子结果立即追加到 JSONL，汇总结果定期原子更新到 JSON，
支持通过 --resume 从下一个种子断点续跑。

示例：
  python run_auto_bench_mulseed.py
  python run_auto_bench_mulseed.py --start-seed 1000000
  python run_auto_bench_mulseed.py --auto-bench /path/to/auto_bench.py
  python run_auto_bench_mulseed.py --quiet --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import urllib.request
from datetime import datetime
from itertools import count
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch


OFFICIAL_AUTO_BENCH_URL = (
    "https://raw.githubusercontent.com/DeepLink-org/DLBlas/"
    "main/benchmarks/ks/auto_bench.py"
)
DEFAULT_START_SEED = 0
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="在 NPU 上按官方 auto_bench 口径做 SinkhornNormalize 多种子精度测试"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_START_SEED,
        help=f"无限递增测试的起始种子（默认 {DEFAULT_START_SEED}）",
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
        help="汇总 JSON 报告路径",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="逐种子 JSONL 路径（默认为汇总文件名加 _cases.jsonl）",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        metavar="N",
        help="每 N 个新种子刷新汇总 JSON 并 fsync（默认 100）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有 JSONL 明细中恢复，跳过已测种子",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印每个种子的明细，仅打印失败项和汇总",
    )
    args = parser.parse_args()
    if args.start_seed < 0:
        parser.error("--start-seed 必须大于等于 0")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol 和 --rtol 必须大于等于 0")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every 必须大于 0")
    if args.details_output is None:
        args.details_output = args.output.with_name(
            args.output.stem + "_cases.jsonl"
        )
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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def metric_value(value: Any) -> float:
    """把 JSON 恢复后的 inf/-inf/nan 字符串还原为可比较数值。"""
    if isinstance(value, (int, float)):
        return float(value)
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if value == "nan":
        return math.nan
    raise ValueError(f"无效的数值指标: {value!r}")


def new_stats() -> dict[str, Any]:
    return {
        "tested_seed_count": 0,
        "passed_seed_count": 0,
        "failed_seed_count": 0,
        "runtime_error_count": 0,
        "last_runtime_error": None,
        "reference_nonfinite_count": 0,
        "candidate_nonfinite_count": 0,
        "both_nan_count": 0,
        "total_case_seconds": 0.0,
        "average_case_seconds": None,
        "min_case_seconds": None,
        "max_case_seconds": None,
        "first_tested_seed": None,
        "last_tested_seed": None,
        "max_tolerance_ratio": None,
        "worst_case_by_tolerance_ratio": None,
        "max_abs_diff": None,
        "worst_case_by_absolute_error": None,
        "failed_seeds_preview": [],
    }


def update_stats(stats: dict[str, Any], case: dict[str, Any]) -> None:
    case_snapshot = {
        key: value
        for key, value in case.items()
        if key not in ("record_type", "recorded_at")
    }
    seed = int(case["seed"])
    stats["tested_seed_count"] += 1
    if case["passed"]:
        stats["passed_seed_count"] += 1
    else:
        stats["failed_seed_count"] += 1
        if len(stats["failed_seeds_preview"]) < 100:
            stats["failed_seeds_preview"].append(seed)
    stats["reference_nonfinite_count"] += int(case["reference_nonfinite_count"])
    stats["candidate_nonfinite_count"] += int(case["candidate_nonfinite_count"])
    stats["both_nan_count"] += int(case["both_nan_count"])
    duration = metric_value(case["duration_seconds"])
    stats["total_case_seconds"] += duration
    stats["average_case_seconds"] = (
        stats["total_case_seconds"] / stats["tested_seed_count"]
    )
    if stats["min_case_seconds"] is None or duration < stats["min_case_seconds"]:
        stats["min_case_seconds"] = duration
    if stats["max_case_seconds"] is None or duration > stats["max_case_seconds"]:
        stats["max_case_seconds"] = duration
    if stats["first_tested_seed"] is None:
        stats["first_tested_seed"] = seed
    stats["last_tested_seed"] = seed

    tolerance_ratio = metric_value(case["max_tolerance_ratio"])
    if (
        stats["max_tolerance_ratio"] is None
        or tolerance_ratio > stats["max_tolerance_ratio"]
    ):
        stats["max_tolerance_ratio"] = tolerance_ratio
        stats["worst_case_by_tolerance_ratio"] = case_snapshot

    max_abs_diff = metric_value(case["max_abs_diff"])
    if stats["max_abs_diff"] is None or max_abs_diff > stats["max_abs_diff"]:
        stats["max_abs_diff"] = max_abs_diff
        stats["worst_case_by_absolute_error"] = case_snapshot


def update_runtime_error_stats(stats: dict[str, Any], error: dict[str, Any]) -> None:
    stats["runtime_error_count"] += 1
    stats["last_runtime_error"] = {
        key: value
        for key, value in error.items()
        if key not in ("record_type", "recorded_at")
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """同目录临时文件 + os.replace，避免汇总 JSON 被写一半。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(
    args: argparse.Namespace,
    bench_path: Path,
    v0_path: Path,
    v1_path: Path,
) -> dict[str, Any]:
    custom_op_path = v1_path.parent / "libsinkhorn_normalize_ops.so"
    return {
        "test_type": "real_npu_custom_operator_multiseed_accuracy",
        "run_mode": "until_keyboard_interrupt",
        "start_seed": args.start_seed,
        "requested_seed_count": None,
        "atol": args.atol,
        "rtol": args.rtol,
        "equal_nan": True,
        "auto_bench_path": str(bench_path),
        "auto_bench_url": OFFICIAL_AUTO_BENCH_URL,
        "auto_bench_sha256": sha256_file(bench_path),
        "v0_file": str(v0_path),
        "v0_sha256": sha256_file(v0_path),
        "v1_file": str(v1_path),
        "v1_sha256": sha256_file(v1_path),
        "custom_op_file": str(custom_op_path),
        "custom_op_sha256": sha256_file(custom_op_path),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def write_journal_metadata(journal: Any, config: dict[str, Any]) -> None:
    record = {
        "record_type": "metadata",
        "schema_version": 1,
        "created_at": now_iso(),
        "config": config,
    }
    journal.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    journal.flush()
    os.fsync(journal.fileno())


def append_case_record(journal: Any, case: dict[str, Any]) -> None:
    record = {
        "record_type": "runtime_error" if case.get("runtime_error") else "case",
        "recorded_at": now_iso(),
        **case,
    }
    journal.write(
        json.dumps(json_safe(record), ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    # 每个种子都刷到操作系统；fsync 由 checkpoint 批量完成。
    journal.flush()


def load_journal(
    path: Path, expected_config: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """恢复 JSONL；若最后一行因意外中断不完整，安全截断该行。"""
    next_seed = int(expected_config["start_seed"])
    stats = new_stats()
    metadata_seen = False
    repaired = False

    with path.open("rb+") as raw:
        file_size = os.fstat(raw.fileno()).st_size
        line_number = 0
        while True:
            line_start = raw.tell()
            line = raw.readline()
            if not line:
                break
            line_number += 1
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if raw.tell() == file_size:
                    raw.truncate(line_start)
                    repaired = True
                    break
                raise RuntimeError(
                    f"JSONL 第 {line_number} 行损坏，无法恢复: {path}"
                ) from exc

            if record.get("record_type") == "metadata":
                if metadata_seen or line_number != 1:
                    raise RuntimeError(f"JSONL metadata 位置异常: {path}")
                metadata_seen = True
                if record.get("config") != expected_config:
                    raise RuntimeError(
                        "--resume 的参数/环境与已有 JSONL 不一致；"
                        "请使用原参数，或去掉 --resume 开始新测试。"
                    )
                continue
            if record.get("record_type") == "runtime_error":
                update_runtime_error_stats(stats, record)
                continue
            if record.get("record_type") != "case":
                raise RuntimeError(f"JSONL 第 {line_number} 行类型无效: {path}")
            seed = int(record["seed"])
            if seed != next_seed:
                raise RuntimeError(
                    f"JSONL 种子序列不连续：期望 seed={next_seed}，实际 seed={seed}: {path}"
                )
            update_stats(stats, record)
            next_seed += 1

    if not metadata_seen:
        raise RuntimeError(f"JSONL 缺少 metadata: {path}")
    if repaired:
        print(f"已截断 JSONL 末尾的不完整记录: {path}")
    return next_seed, stats


def build_summary(
    config: dict[str, Any],
    stats: dict[str, Any],
    *,
    status: str,
    elapsed_seconds: float,
    details_path: Path,
    stop_reason: str = "",
) -> dict[str, Any]:
    all_tested_passed = (
        stats["tested_seed_count"] > 0 and stats["failed_seed_count"] == 0
    )
    return {
        "schema_version": 2,
        "status": status,
        "all_passed": all_tested_passed,
        "next_seed": config["start_seed"] + stats["tested_seed_count"],
        "updated_at": now_iso(),
        "elapsed_seconds": elapsed_seconds,
        "stop_reason": stop_reason,
        "details_file": str(details_path),
        "details_format": "JSON Lines; first record is metadata, remaining records are per-seed cases or runtime errors",
        "config": config,
        "summary": stats,
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    bench_path = ensure_official_auto_bench(args.auto_bench, args.no_download)
    bench = load_official_auto_bench(bench_path)
    v0_path = args.v0_file.resolve()
    v1_path = args.v1_file.resolve()
    for path, label in ((v0_path, "v0"), (v1_path, "v1")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 文件不存在: {path}")

    output_path = args.output.resolve()
    details_path = args.details_output.resolve()
    if output_path == details_path:
        raise ValueError("--output 和 --details-output 不能是同一文件")
    config = build_config(args, bench_path, v0_path, v1_path)

    journal_resumed = (
        args.resume and details_path.is_file() and details_path.stat().st_size > 0
    )
    previous_elapsed = 0.0
    if journal_resumed and output_path.is_file():
        try:
            previous_summary = json.loads(output_path.read_text(encoding="utf-8"))
            previous_elapsed = float(previous_summary.get("elapsed_seconds", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            previous_elapsed = 0.0

    details_path.parent.mkdir(parents=True, exist_ok=True)
    if journal_resumed:
        next_seed, stats = load_journal(details_path, config)
        journal = details_path.open("a", encoding="utf-8")
    else:
        next_seed = args.start_seed
        stats = new_stats()
        journal = details_path.open("w", encoding="utf-8")
        write_journal_metadata(journal, config)

    print("=" * 78)
    print("SinkhornNormalize NPU 多随机种子精度测试")
    print("=" * 78)
    print(f"种子       : 从 {next_seed} 开始无限递增，按 Ctrl+C 停止")
    print(f"判定口径   : torch.allclose(atol={args.atol:g}, rtol={args.rtol:g}, equal_nan=True)")
    print(f"auto_bench : {bench_path}")
    print(f"v0         : {v0_path}")
    print(f"v1         : {v1_path}")
    print(f"汇总 JSON  : {output_path}")
    print(f"明细 JSONL : {details_path}")
    if stats["tested_seed_count"]:
        print(
            f"断点恢复   : 已完成 {stats['tested_seed_count']} 个种子，"
            f"从 seed={next_seed} 继续"
        )
    print("-" * 78)

    initial_report = build_summary(
        config,
        stats,
        status="running",
        elapsed_seconds=previous_elapsed,
        details_path=details_path,
    )
    atomic_write_json(output_path, initial_report)

    runtime_error = False
    interrupted = False
    stopped_by_fail_fast = False
    stop_reason = ""
    new_case_count = 0
    try:
        for seed in count(next_seed):
            runtime_error = False
            case_started = time.perf_counter()
            try:
                case = run_seed(bench, v0_path, v1_path, seed, args.atol, args.rtol)
                case["runtime_error"] = False
            except Exception as exc:  # noqa: BLE001 - 记录后停止无意义重试
                runtime_error = True
                case = {
                    "seed": seed,
                    "passed": False,
                    "runtime_error": True,
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
            case["duration_seconds"] = time.perf_counter() - case_started

            append_case_record(journal, case)
            if runtime_error:
                update_runtime_error_stats(stats, case)
            else:
                update_stats(stats, case)
            new_case_count += 1

            tested_count = stats["tested_seed_count"]
            if not args.quiet or not case["passed"]:
                case_status = "PASS" if case["passed"] else "FAIL"
                print(
                    f"[已测 {tested_count}] "
                    f"seed={seed:<10} {case_status}  "
                    f"max_abs={case['max_abs_diff']:.3e}  "
                    f"tol_ratio={case['max_tolerance_ratio']:.3e}"
                )
                if case["message"]:
                    print(f"  {case['message']}")

            if new_case_count % args.checkpoint_every == 0:
                os.fsync(journal.fileno())
                elapsed = previous_elapsed + time.perf_counter() - started
                checkpoint = build_summary(
                    config,
                    stats,
                    status="running",
                    elapsed_seconds=elapsed,
                    details_path=details_path,
                )
                atomic_write_json(output_path, checkpoint)

            if runtime_error:
                stop_reason = case["message"]
                print("遇到运行异常，已停止后续种子测试。")
                break
            if args.fail_fast and not case["passed"]:
                stopped_by_fail_fast = True
                stop_reason = f"--fail-fast: seed={seed} 精度失败"
                break
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "用户中断"
        print("\n用户中断，正在保存检查点……", file=sys.stderr)
    finally:
        journal.flush()
        os.fsync(journal.fileno())
        journal.close()

    elapsed = previous_elapsed + time.perf_counter() - started
    if interrupted:
        final_status = "interrupted"
    elif runtime_error:
        final_status = "error"
    elif stopped_by_fail_fast:
        final_status = "stopped"
    else:
        final_status = "stopped"
    report = build_summary(
        config,
        stats,
        status=final_status,
        elapsed_seconds=elapsed,
        details_path=details_path,
        stop_reason=stop_reason,
    )
    atomic_write_json(output_path, report)

    print("-" * 78)
    print(
        f"汇总       : {'PASS' if report['all_passed'] else 'FAIL'}  "
        f"{stats['passed_seed_count']}/{stats['tested_seed_count']} 个已测种子通过"
    )
    max_abs_case = stats["worst_case_by_absolute_error"]
    worst_ratio_case = stats["worst_case_by_tolerance_ratio"]
    if max_abs_case is not None:
        print(
            f"最大绝对误差: {metric_value(max_abs_case['max_abs_diff']):.6e} "
            f"(seed={max_abs_case['seed']})"
        )
    if worst_ratio_case is not None:
        print(
            f"最大容差占用: {metric_value(worst_ratio_case['max_tolerance_ratio']):.6e} "
            f"(seed={worst_ratio_case['seed']}; < 1 才通过)"
        )
    print(f"耗时       : {elapsed:.3f} s")
    print(f"汇总 JSON  : {output_path}")
    print(f"明细 JSONL : {details_path}")
    if interrupted:
        return 130
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
