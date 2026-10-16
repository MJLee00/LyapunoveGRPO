#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any


DEFAULT_ROLLOUT_ROOT = Path("/mnt/ws-jfs/dsh-jfs/lrz/verl/val_results/CMDP/v2_qwen3-8b-base_filter")
DEFAULT_TRAIN_FILE = Path("/mnt/ws-jfs/dsh-jfs/lrz/verl/data/train/dapo_train.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate rollout accuracy by prompt index from validation JSONL files, "
            "then filter dapo_train.parquet to keep samples with 0.2 < accuracy < 0.8."
        )
    )
    parser.add_argument(
        "-r", "--rollout-root",
        type=Path,
        default=DEFAULT_ROLLOUT_ROOT,
        help="Root directory containing step subdirectories such as 0/0.jsonl, 1/1.jsonl, ...",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="Path to dapo_train.parquet.",
    )
    parser.add_argument(
        "--lower",
        type=float,
        default=0.2,
        help="Lower accuracy bound to keep, inclusive.",
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=0.8,
        help="Upper accuracy bound to keep, inclusive.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output path prefix. The saved row count will be appended before the suffix.",
    )
    parser.add_argument(
        "--strict-missing-accuracy",
        action="store_true",
        help="Raise an error if any training sample cannot find a matching accuracy.",
    )
    return parser.parse_args()


def _normalize_index(value: Any) -> str:
    if value is None:
        raise ValueError("index is None")
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError(f"index must be finite, got {value}")
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if not text:
        raise ValueError("index is empty")

    try:
        number = float(text)
    except ValueError:
        return text

    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _maybe_parse_extra_info(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_train_index(record: dict[str, Any], row_idx: int) -> tuple[str, bool]:
    extra_info = record.get("extra_info")
    extra_info_dict = _maybe_parse_extra_info(extra_info)
    if isinstance(extra_info_dict, dict) and "index" in extra_info_dict:
        try:
            return _normalize_index(extra_info_dict["index"]), False
        except (TypeError, ValueError):
            pass
    if "index" in record:
        try:
            return _normalize_index(record["index"]), False
        except (TypeError, ValueError):
            pass
    return str(row_idx), True


def _extract_train_index_with_original(record: dict[str, Any], row_idx: int) -> tuple[str, bool, Any]:
    extra_info = record.get("extra_info")
    extra_info_dict = _maybe_parse_extra_info(extra_info)
    if isinstance(extra_info_dict, dict) and "index" in extra_info_dict:
        raw_index = extra_info_dict["index"]
        try:
            return _normalize_index(raw_index), False, _to_python_scalar(raw_index)
        except (TypeError, ValueError):
            pass
    if "index" in record:
        raw_index = record["index"]
        try:
            return _normalize_index(raw_index), False, _to_python_scalar(raw_index)
        except (TypeError, ValueError):
            pass
    return str(row_idx), True, row_idx


def _extract_rollout_index(record: dict[str, Any]) -> str:
    if "index" not in record:
        raise KeyError("rollout record missing 'index' field")
    return _normalize_index(record["index"])


def _score_to_is_correct(score: Any) -> bool:
    if score is None:
        return False
    return float(score) > 0.0


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return numeric
    return value


def discover_rollout_jsonl_files(rollout_root: Path) -> list[Path]:
    if not rollout_root.exists():
        raise FileNotFoundError(f"Rollout root does not exist: {rollout_root}")
    if not rollout_root.is_dir():
        raise NotADirectoryError(f"Rollout root is not a directory: {rollout_root}")

    step_paths: list[Path] = []
    for child in rollout_root.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        jsonl_path = child / f"{child.name}.jsonl"
        if jsonl_path.is_file():
            step_paths.append(jsonl_path)

    if step_paths:
        return sorted(step_paths, key=lambda path: int(path.stem))

    direct_paths = [path for path in rollout_root.glob("*.jsonl") if path.stem.isdigit()]
    if direct_paths:
        return sorted(direct_paths, key=lambda path: int(path.stem))

    raise FileNotFoundError(
        "No rollout JSONL files found. Expected directories like 0/0.jsonl, 1/1.jsonl under "
        f"{rollout_root}"
    )


def aggregate_accuracy(rollout_files: list[Path]) -> tuple[dict[str, float], dict[str, tuple[int, int]]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for rollout_file in rollout_files:
        with rollout_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {rollout_file}:{line_no}") from exc

                index = _extract_rollout_index(record)
                is_correct = _score_to_is_correct(record.get("score"))
                counts[index][0] += 1
                counts[index][1] += int(is_correct)

    accuracy_by_index = {
        index: correct / total
        for index, (total, correct) in counts.items()
        if total > 0
    }
    counts_summary = {
        index: (total, correct)
        for index, (total, correct) in counts.items()
    }
    return accuracy_by_index, counts_summary


def build_default_output_path(train_file: Path, lower: float, upper: float, rollout_root) -> Path:
    lower_pct = int(round(lower * 100))
    upper_pct = int(round(upper * 100))
    model_spec = str(rollout_root).split("/")[-1].split("_")[1]
    return train_file.with_name(f"{train_file.stem}_{model_spec}_{lower_pct}_{upper_pct}{train_file.suffix}")


def _append_saved_count_to_output_path(output_file: Path, saved_count: int) -> Path:
    return output_file.with_name(f"{output_file.stem}_{saved_count}{output_file.suffix}")


def _import_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading parquet requires pandas and a parquet engine such as pyarrow. "
            "Please install them in the runtime environment before running this script."
        ) from exc
    return pd


def _parse_extra_info_for_write(value: Any, row_idx: int) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    if value is None:
        return {}

    if isinstance(value, float) and math.isnan(value):
        return {}

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Row {row_idx} has non-JSON string extra_info, cannot rewrite index") from exc
        if not isinstance(parsed, dict):
            raise TypeError(f"Row {row_idx} extra_info must decode to a dict, got {type(parsed)}")
        return parsed

    raise TypeError(f"Row {row_idx} extra_info must be a dict, JSON string, or null, got {type(value)}")


def _rewrite_filtered_extra_info_indices(dataframe, original_indices: list[Any], accuracies: list[float]) -> None:
    if len(dataframe) != len(original_indices) or len(dataframe) != len(accuracies):
        raise ValueError("Filtered dataframe and metadata lengths do not match")

    updated_extra_infos: list[dict[str, Any]] = []
    for new_index, extra_info in enumerate(dataframe["extra_info"].tolist()):
        updated = _parse_extra_info_for_write(extra_info, new_index)
        updated["index"] = new_index
        updated["original_index"] = original_indices[new_index]
        updated["accuracy"] = float(accuracies[new_index])
        updated_extra_infos.append(updated)
    dataframe.loc[:, "extra_info"] = updated_extra_infos


def _filter_train_parquet(
    train_file: Path,
    output_file: Path,
    accuracy_by_index: dict[str, float],
    lower: float,
    upper: float,
    strict_missing_accuracy: bool,
) -> dict[str, int]:
    pd = _import_pandas()
    dataframe = pd.read_parquet(train_file)
    records = dataframe.to_dict(orient="records")

    total_rows = len(records)
    kept_rows = 0
    missing_accuracy = 0
    kept_with_fallback_index = 0
    skipped_missing_accuracy_examples: list[str] = []
    keep_mask: list[bool] = []
    kept_original_indices: list[Any] = []
    kept_accuracies: list[float] = []

    for row_idx, record in enumerate(records):
        sample_index, used_fallback_index, original_index = _extract_train_index_with_original(record, row_idx)
        accuracy = accuracy_by_index.get(sample_index)

        if accuracy is None:
            missing_accuracy += 1
            keep_mask.append(False)
            if len(skipped_missing_accuracy_examples) < 5:
                skipped_missing_accuracy_examples.append(sample_index)
            continue

        keep = lower < accuracy < upper
        keep_mask.append(keep)
        if keep:
            kept_rows += 1
            kept_original_indices.append(original_index)
            kept_accuracies.append(float(accuracy))
            if used_fallback_index:
                kept_with_fallback_index += 1

    if strict_missing_accuracy and missing_accuracy > 0:
        sample_text = ", ".join(skipped_missing_accuracy_examples)
        raise ValueError(
            f"Missing accuracy for {missing_accuracy} training samples. "
            f"Example indices: {sample_text}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_dataframe = dataframe.loc[keep_mask].copy().reset_index(drop=True)
    if "extra_info" not in filtered_dataframe.columns:
        filtered_dataframe["extra_info"] = [{} for _ in range(len(filtered_dataframe))]
    _rewrite_filtered_extra_info_indices(filtered_dataframe, kept_original_indices, kept_accuracies)
    actual_output_file = _append_saved_count_to_output_path(output_file, kept_rows)
    if train_file.expanduser().resolve() == actual_output_file.expanduser().resolve():
        raise ValueError("output_file must be different from train_file after appending saved row count")
    filtered_dataframe.to_parquet(actual_output_file, index=False)

    return {
        "total_lines": total_rows,
        "kept_lines": kept_rows,
        "missing_accuracy": missing_accuracy,
        "kept_with_fallback_index": kept_with_fallback_index,
        "output_file": str(actual_output_file),
    }


def filter_train_file(
    train_file: Path,
    output_file: Path,
    accuracy_by_index: dict[str, float],
    lower: float,
    upper: float,
    strict_missing_accuracy: bool,
) -> dict[str, int]:
    if not train_file.exists():
        raise FileNotFoundError(f"Train file does not exist: {train_file}")
    if train_file.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet train files are supported, got: {train_file}")
    if output_file.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet output files are supported, got: {output_file}")
    return _filter_train_parquet(
        train_file=train_file,
        output_file=output_file,
        accuracy_by_index=accuracy_by_index,
        lower=lower,
        upper=upper,
        strict_missing_accuracy=strict_missing_accuracy,
    )


def main() -> None:
    args = parse_args()

    if args.lower > args.upper:
        raise ValueError(f"lower must be <= upper, got lower={args.lower}, upper={args.upper}")

    rollout_files = discover_rollout_jsonl_files(args.rollout_root)
    accuracy_by_index, counts_summary = aggregate_accuracy(rollout_files)

    output_file = args.output_file
    if output_file is None:
        output_file = build_default_output_path(args.train_file, args.lower, args.upper, args.rollout_root)

    filter_stats = filter_train_file(
        train_file=args.train_file,
        output_file=output_file,
        accuracy_by_index=accuracy_by_index,
        lower=args.lower,
        upper=args.upper,
        strict_missing_accuracy=args.strict_missing_accuracy,
    )

    total_rollout_records = sum(total for total, _ in counts_summary.values())
    print(f"rollout_files={len(rollout_files)}")
    print(f"rollout_records={total_rollout_records}")
    print(f"prompt_count_with_accuracy={len(accuracy_by_index)}")
    print(f"train_total={filter_stats['total_lines']}")
    print(f"train_kept_{int(round(args.lower * 100))}_{int(round(args.upper * 100))}={filter_stats['kept_lines']}")
    print(f"train_missing_accuracy={filter_stats['missing_accuracy']}")
    print(f"train_kept_with_fallback_index={filter_stats['kept_with_fallback_index']}")
    print(f"output_file={filter_stats['output_file']}")


if __name__ == "__main__":
    main()
