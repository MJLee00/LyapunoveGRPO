#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_TOP_K_TOKENS = 50
DEFAULT_LAGRANGIAN_C = 0.00050
DEFAULT_LAGRANGIAN_LAMBDAS = (0.1, 0.3, 0.4, 0.5, 1.0)
DEFAULT_LAGRANGIAN_RATIO = "seq_geo_no_exp"
DEFAULT_LAGRANGIAN_BUFFER_MODE = "p80"
DEFAULT_NORMALIZE_MODE = "sigmoid"


def _flatten_sample_records(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not samples:
        return {}

    all_keys: list[str] = []
    seen_keys: set[str] = set()
    for sample in samples:
        for key in sample.keys():
            if key not in seen_keys:
                seen_keys.add(key)
                all_keys.append(key)

    flattened = {key: [] for key in all_keys}
    for sample in samples:
        for key in all_keys:
            flattened[key].append(sample.get(key, None))
    return flattened


def _load_jsonl(path: Path) -> dict[str, Any]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return _flatten_sample_records(samples)


def _resolve_rollout_jsonl_path(rollout_data_dir: Path, step: int) -> Path:
    jsonl_path = rollout_data_dir / f"{step}.jsonl"
    if jsonl_path.exists():
        return jsonl_path

    raise FileNotFoundError(f"Cannot find {step}.jsonl under {rollout_data_dir}")


def _values_match(lhs: Any, rhs: Any) -> bool:
    if torch.is_tensor(lhs) and torch.is_tensor(rhs):
        return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and torch.equal(lhs, rhs)
    try:
        return lhs == rhs
    except Exception:
        return False


def _merge_dicts(json_data: Any, pt_data: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            merged[key] = value
    else:
        merged["samples"] = json_data

    for key, value in pt_data.items():
        if key not in merged:
            merged[key] = value
            continue

        existing = merged[key]
        if type(existing) is type(value) and _values_match(existing, value):
            continue

        merged[f"json_{key}"] = existing
        merged[f"pt_{key}"] = value
        merged.pop(key, None)

    return merged


def load_rollout_step_data(rollout_data_dir: str | Path, step: int) -> dict[str, Any]:
    rollout_dir = Path(rollout_data_dir)
    jsonl_path = _resolve_rollout_jsonl_path(rollout_dir, step)
    pt_path = rollout_dir / f"{step}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Cannot find {pt_path}")

    json_data = _load_jsonl(jsonl_path)
    pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(pt_data, dict):
        raise TypeError(f"Expected {pt_path} to contain a dict, got {type(pt_data)}")

    merged = _merge_dicts(json_data=json_data, pt_data=pt_data)
    merged["_meta"] = {
        "step": step,
        "rollout_data_dir": str(rollout_dir),
        "jsonl_path": str(jsonl_path),
        "pt_path": str(pt_path),
    }
    return merged


def _require_tensor(merged: dict[str, Any], key: str) -> torch.Tensor:
    if key not in merged:
        raise KeyError(f"Missing key '{key}' in merged rollout data")
    value = merged[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected merged['{key}'] to be a torch.Tensor, got {type(value)}")
    return value.detach().cpu()


def _get_merged_value(merged: dict[str, Any], key: str) -> Any:
    if key in merged:
        return merged[key]
    json_key = f"json_{key}"
    if json_key in merged:
        return merged[json_key]
    pt_key = f"pt_{key}"
    if pt_key in merged:
        return merged[pt_key]
    raise KeyError(f"Missing key '{key}' in merged rollout data")


def _broadcast_to_response_shape(tensor: torch.Tensor, response_mask: torch.Tensor, key: str) -> torch.Tensor:
    if tensor.shape == response_mask.shape:
        return tensor
    if tensor.dim() == 1 and tensor.shape[0] == response_mask.shape[0]:
        return tensor.unsqueeze(-1).expand_as(response_mask)
    if tensor.dim() == 2 and tensor.shape[0] == response_mask.shape[0] and tensor.shape[1] == 1:
        return tensor.expand_as(response_mask)
    raise ValueError(
        f"Cannot broadcast key '{key}' with shape {tuple(tensor.shape)} to response_mask shape {tuple(response_mask.shape)}"
    )


def _masked_flatten(tensor: torch.Tensor, response_mask: torch.Tensor, key: str) -> torch.Tensor:
    tensor = _broadcast_to_response_shape(tensor.float(), response_mask, key)
    mask = response_mask.bool()
    return tensor[mask].reshape(-1)


def _parse_float_csv(csv_text: str) -> list[float]:
    values = []
    for part in csv_text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one lambda value")
    return values


def _build_grpo_uid_array(merged: dict[str, Any], batch_size: int) -> np.ndarray:
    for key in ("uid", "input"):
        try:
            value = _get_merged_value(merged, key)
        except KeyError:
            continue
        if isinstance(value, list) and len(value) == batch_size:
            return np.asarray([str(item) for item in value], dtype=object)

    prompt_token_ids = _require_tensor(merged, "prompt_token_ids").long()
    if prompt_token_ids.shape[0] != batch_size:
        raise ValueError(f"prompt_token_ids batch size {prompt_token_ids.shape[0]} does not match response batch size {batch_size}")

    prompt_lengths = None
    try:
        prompt_lengths = _get_merged_value(merged, "prompt_length")
    except KeyError:
        prompt_lengths = None

    uids: list[str] = []
    for row_idx in range(batch_size):
        row = prompt_token_ids[row_idx].tolist()
        if isinstance(prompt_lengths, list) and len(prompt_lengths) == batch_size and prompt_lengths[row_idx] is not None:
            prompt_len = int(prompt_lengths[row_idx])
            row = row[-prompt_len:] if prompt_len > 0 else []
        uids.append("prompt_ids:" + ",".join(map(str, row)))
    return np.asarray(uids, dtype=object)


def _build_token_level_scores_from_merged(merged: dict[str, Any], response_mask: torch.Tensor) -> torch.Tensor:
    batch_size = response_mask.shape[0]
    try:
        token_level_scores = _require_tensor(merged, "token_level_scores").float()
        return _broadcast_to_response_shape(token_level_scores, response_mask, "token_level_scores")
    except KeyError:
        pass

    score_values = _get_merged_value(merged, "score")
    if not isinstance(score_values, list) or len(score_values) != batch_size:
        raise ValueError(
            f"Expected merged['score'] to be a list with batch size {batch_size}, got {type(score_values)}"
        )

    seq_scores = torch.tensor(
        [0.0 if value is None else float(value) for value in score_values],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    last_idx = response_mask.float().sum(dim=-1).long() - 1
    last_idx = last_idx.clamp(min=0)
    row_idx = torch.arange(batch_size, dtype=torch.long)
    token_level_scores[row_idx, last_idx] = seq_scores
    return token_level_scores


def _masked_sequence_mean(token_tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    denom = response_mask.float().sum(dim=-1).clamp_min(1.0)
    return (token_tensor.float() * response_mask.float()).sum(dim=-1) / denom


def _compute_grpo_sequence_advantage_details(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid_array: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> dict[str, Any]:
    seq_rewards = token_level_rewards.sum(dim=-1).detach().cpu().float()
    uid_list = [str(item) for item in uid_array.tolist()]

    group_to_indices: dict[str, list[int]] = {}
    for row_idx, uid in enumerate(uid_list):
        group_to_indices.setdefault(uid, []).append(row_idx)

    group_mean = torch.zeros_like(seq_rewards)
    group_std = torch.zeros_like(seq_rewards)
    advantage = torch.zeros_like(seq_rewards)

    for uid, row_indices in group_to_indices.items():
        group_scores = seq_rewards[row_indices]
        if len(row_indices) == 1:
            mean_value = torch.tensor(0.0, dtype=seq_rewards.dtype)
            std_value = torch.tensor(1.0, dtype=seq_rewards.dtype)
        else:
            mean_value = group_scores.mean()
            std_value = group_scores.std()

        for row_idx in row_indices:
            group_mean[row_idx] = mean_value
            group_std[row_idx] = std_value
            centered = seq_rewards[row_idx] - mean_value
            if norm_adv_by_std_in_grpo:
                advantage[row_idx] = centered / (std_value + epsilon)
            else:
                advantage[row_idx] = centered

    return {
        "seq_rewards": seq_rewards,
        "group_mean": group_mean,
        "group_std": group_std,
        "advantage_per_seq": advantage,
        "group_to_indices": group_to_indices,
        "uid_list": uid_list,
    }


def _get_optional_list_field(merged: dict[str, Any], key: str, expected_len: int) -> list[Any] | None:
    try:
        value = _get_merged_value(merged, key)
    except KeyError:
        return None
    if isinstance(value, list) and len(value) == expected_len:
        return value
    return None


def _filter_merged_by_row_indices(merged: dict[str, Any], row_indices: list[int]) -> dict[str, Any]:
    response_mask = _require_tensor(merged, "response_mask")
    batch_size = response_mask.shape[0]
    index_tensor = torch.tensor(row_indices, dtype=torch.long)
    filtered: dict[str, Any] = {}
    for key, value in merged.items():
        if key == "_meta":
            filtered[key] = dict(value)
            continue
        if key == "_analysis":
            continue
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == batch_size:
            filtered[key] = value.index_select(0, index_tensor)
            continue
        if isinstance(value, list) and len(value) == batch_size:
            filtered[key] = [value[idx] for idx in row_indices]
            continue
        if isinstance(value, np.ndarray) and value.shape[0] == batch_size:
            filtered[key] = value[row_indices]
            continue
        filtered[key] = value
    if "_meta" in filtered:
        filtered["_meta"]["filtered_row_indices"] = row_indices
        filtered["_meta"]["filtered_batch_size"] = len(row_indices)
    return filtered


def select_mixed_prompt_subset(merged: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    response_mask = _require_tensor(merged, "response_mask")
    batch_size = response_mask.shape[0]
    score_list = _get_optional_list_field(merged, "score", batch_size)
    if score_list is None:
        raise ValueError("Mixed-prompt analysis requires jsonl 'score' field")

    uid_array = _build_grpo_uid_array(merged, batch_size=batch_size)
    group_to_indices: dict[str, list[int]] = {}
    for row_idx, uid in enumerate(uid_array.tolist()):
        group_to_indices.setdefault(str(uid), []).append(row_idx)

    mixed_row_indices: list[int] = []
    all_correct_prompt_count = 0
    all_wrong_prompt_count = 0
    mixed_prompt_count = 0

    for uid, row_indices in group_to_indices.items():
        group_scores = [0.0 if score_list[idx] is None else float(score_list[idx]) for idx in row_indices]
        group_correct = [score > 0 for score in group_scores]
        if all(group_correct):
            all_correct_prompt_count += 1
        elif not any(group_correct):
            all_wrong_prompt_count += 1
        else:
            mixed_prompt_count += 1
            mixed_row_indices.extend(row_indices)

    mixed_row_indices = sorted(mixed_row_indices)
    subset = _filter_merged_by_row_indices(merged, mixed_row_indices)
    summary = {
        "total_prompt_count": len(group_to_indices),
        "all_correct_prompt_count": all_correct_prompt_count,
        "all_wrong_prompt_count": all_wrong_prompt_count,
        "mixed_prompt_count": mixed_prompt_count,
        "total_response_count": batch_size,
        "mixed_response_count": len(mixed_row_indices),
        "criterion": "score > 0 => correct, score <= 0 => wrong",
    }
    return subset, summary


def compute_lagrangian_lambda_sweep(
    merged: dict[str, Any],
    lambdas: list[float] | tuple[float, ...] = DEFAULT_LAGRANGIAN_LAMBDAS,
    constraint_budget_c: float = DEFAULT_LAGRANGIAN_C,
    ratio_type: str = DEFAULT_LAGRANGIAN_RATIO,
    buffer_mode: str = DEFAULT_LAGRANGIAN_BUFFER_MODE,
    normalize_mode: str = DEFAULT_NORMALIZE_MODE,
) -> dict[str, Any]:
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.trainer.ppo.grpo_lagrangian import GRPOLagrangianOperator
    from verl.trainer.ppo.rollout_corr_helper import compute_raw_penalty_per_sequence, normalize_penalty_per_sequence

    response_mask = _require_tensor(merged, "response_mask").float()
    old_log_probs = _broadcast_to_response_shape(_require_tensor(merged, "old_log_probs").float(), response_mask, "old_log_probs")
    rollout_log_probs = _broadcast_to_response_shape(
        _require_tensor(merged, "rollout_log_probs").float(), response_mask, "rollout_log_probs"
    )
    token_level_scores = _build_token_level_scores_from_merged(merged, response_mask)
    uid_array = _build_grpo_uid_array(merged, batch_size=response_mask.shape[0])

    base_batch = DataProto.from_dict(
        tensors={
            "old_log_probs": old_log_probs.clone(),
            "rollout_log_probs": rollout_log_probs.clone(),
            "response_mask": response_mask.clone(),
            "token_level_scores": token_level_scores.clone(),
            "token_level_rewards": token_level_scores.clone(),
        },
        non_tensors={"uid": uid_array},
    )

    raw_penalty_per_seq = compute_raw_penalty_per_sequence(
        batch=base_batch,
        ratio_type=ratio_type,
        penalty_cost_start_step=None,
        global_step=None,
        clip_penalty_low=None,
        clip_penalty_high=None,
    )
    normalized_penalty_per_seq = normalize_penalty_per_sequence(
        raw_per_seq=raw_penalty_per_seq,
        max_ratio=None,
        normalize_mode=normalize_mode,
        normalize_tanh_k=None,
        penalty_truncated=None,
    )

    results: dict[str, Any] = {}
    lambda_values = [float(value) for value in lambdas]
    lambda_max = max([1.0, *lambda_values])
    seq_scores_before_penalty = token_level_scores.sum(dim=-1).detach().cpu().float()
    last_response_token_idx = response_mask.float().sum(dim=-1).long().clamp(min=1) - 1
    for lambda_value in lambda_values:
        lagrangian_operator = GRPOLagrangianOperator(
            {
                "constraint_budget_c": float(constraint_budget_c),
                "lambda_init": float(lambda_value),
                "lambda_lr": 1.0,
                "lambda_min": 0.0,
                "lambda_max": float(lambda_max),
                "operator_update_every_z": 1,
                "buffer_size": 1,
                "buffer_mode": buffer_mode,
                "ratio_type": ratio_type,
                "enable_lambda2": False,
            }
        )
        batch = DataProto.from_dict(
            tensors={
                "old_log_probs": old_log_probs.clone(),
                "rollout_log_probs": rollout_log_probs.clone(),
                "response_mask": response_mask.clone(),
                "token_level_scores": token_level_scores.clone(),
                "token_level_rewards": token_level_scores.clone(),
            },
            non_tensors={"uid": uid_array.copy()},
        )
        lagrangian_operator.apply_to_batch(
            batch=batch,
            penalty_cost_start_step=None,
            global_step=None,
            clip_penalty_low=None,
            clip_penalty_high=None,
            normalize_mode=normalize_mode,
            normalize_tanh_k=None,
            penalty_truncated=None,
        )
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            index=batch.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=True,
        )

        weighted_penalty_token = batch.batch["rollout_penalty"].detach().cpu()
        weighted_penalty_per_seq = weighted_penalty_token.sum(dim=-1)
        advantage_token = advantages.detach().cpu()
        advantage_per_seq = _masked_sequence_mean(advantage_token, response_mask)
        grpo_details = _compute_grpo_sequence_advantage_details(
            token_level_rewards=batch.batch["token_level_rewards"].detach().cpu(),
            response_mask=response_mask,
            uid_array=uid_array,
            norm_adv_by_std_in_grpo=True,
        )
        computed_advantage = grpo_details["advantage_per_seq"]
        if not torch.allclose(advantage_per_seq, computed_advantage, atol=1e-6, rtol=1e-6):
            raise ValueError("Offline GRPO advantage reconstruction mismatch with compute_grpo_outcome_advantage")

        lambda_key = f"{lambda_value:g}"
        results[lambda_key] = {
            "lambda": float(lambda_value),
            "weighted_penalty_token": weighted_penalty_token,
            "weighted_penalty_per_seq": weighted_penalty_per_seq,
            "seq_reward_after_penalty": grpo_details["seq_rewards"],
            "group_mean": grpo_details["group_mean"],
            "group_std": grpo_details["group_std"],
            "advantage_token": advantage_token,
            "advantage_per_seq": advantage_per_seq,
            "returns_token": returns.detach().cpu(),
            "summary": {
                "weighted_penalty_mean": float(weighted_penalty_per_seq.mean().item()),
                "weighted_penalty_std": float(weighted_penalty_per_seq.std().item()),
                "weighted_penalty_min": float(weighted_penalty_per_seq.min().item()),
                "weighted_penalty_max": float(weighted_penalty_per_seq.max().item()),
                "advantage_mean": float(advantage_per_seq.mean().item()),
                "advantage_std": float(advantage_per_seq.std().item()),
                "advantage_min": float(advantage_per_seq.min().item()),
                "advantage_max": float(advantage_per_seq.max().item()),
            },
        }

    return {
        "config": {
            "constraint_budget_c": float(constraint_budget_c),
            "ratio_type": ratio_type,
            "buffer_mode": buffer_mode,
            "normalize_mode": normalize_mode,
            "norm_adv_by_std_in_grpo": True,
        },
        "lambdas": lambda_values,
        "uid_list": [str(item) for item in uid_array.tolist()],
        "seq_score_before_penalty": seq_scores_before_penalty,
        "last_response_token_idx": last_response_token_idx.detach().cpu(),
        "raw_penalty_per_seq": raw_penalty_per_seq.detach().cpu(),
        "normalized_penalty_per_seq": normalized_penalty_per_seq.detach().cpu() if normalized_penalty_per_seq is not None else None,
        "results": results,
    }


def analyze_lambda_difference_cases(
    merged: dict[str, Any],
    lambda_sweep: dict[str, Any],
    lambda_a: float = 0.5,
    lambda_b: float = 1.0,
    max_cases: int = 3,
    diff_tolerance: float = 1e-8,
) -> dict[str, Any]:
    lambda_a_key = f"{float(lambda_a):g}"
    lambda_b_key = f"{float(lambda_b):g}"
    if lambda_a_key not in lambda_sweep["results"]:
        raise KeyError(f"Lambda {lambda_a} not found in computed sweep results")
    if lambda_b_key not in lambda_sweep["results"]:
        raise KeyError(f"Lambda {lambda_b} not found in computed sweep results")

    result_a = lambda_sweep["results"][lambda_a_key]
    result_b = lambda_sweep["results"][lambda_b_key]
    advantage_a = result_a["advantage_per_seq"].float()
    advantage_b = result_b["advantage_per_seq"].float()
    abs_diff = (advantage_b - advantage_a).abs()
    valid_mask = torch.isfinite(abs_diff) & (abs_diff > diff_tolerance)
    differing_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)

    uid_list: list[str] = list(lambda_sweep["uid_list"])
    group_to_indices: dict[str, list[int]] = {}
    for row_idx, uid in enumerate(uid_list):
        group_to_indices.setdefault(uid, []).append(row_idx)

    batch_size = advantage_a.shape[0]
    input_list = _get_optional_list_field(merged, "input", batch_size)
    output_list = _get_optional_list_field(merged, "output", batch_size)
    score_list = _get_optional_list_field(merged, "score", batch_size)
    pred_list = _get_optional_list_field(merged, "pred", batch_size)
    prompt_length_list = _get_optional_list_field(merged, "prompt_length", batch_size)
    response_length_list = _get_optional_list_field(merged, "response_length", batch_size)

    selected_cases: list[dict[str, Any]] = []
    if differing_indices.numel() > 0:
        topk = min(max_cases, differing_indices.numel())
        top_positions = torch.topk(abs_diff[differing_indices], k=topk, largest=True).indices.tolist()
        top_indices = differing_indices[top_positions].tolist()
        raw_penalty_per_seq = lambda_sweep["raw_penalty_per_seq"].float()
        normalized_penalty_per_seq = lambda_sweep["normalized_penalty_per_seq"]
        if normalized_penalty_per_seq is not None:
            normalized_penalty_per_seq = normalized_penalty_per_seq.float()
        seq_score_before_penalty = lambda_sweep["seq_score_before_penalty"].float()
        last_response_token_idx = lambda_sweep["last_response_token_idx"].long()

        for row_idx in top_indices:
            uid = uid_list[row_idx]
            group_indices = group_to_indices[uid]

            group_cases = []
            for group_row_idx in group_indices:
                group_cases.append(
                    {
                        "row_idx": int(group_row_idx),
                        "seq_score_before_penalty": float(seq_score_before_penalty[group_row_idx].item()),
                        "raw_penalty_per_seq": float(raw_penalty_per_seq[group_row_idx].item()),
                        "normalized_penalty_per_seq": None
                        if normalized_penalty_per_seq is None
                        else float(normalized_penalty_per_seq[group_row_idx].item()),
                        f"weighted_penalty_lambda_{lambda_a_key}": float(result_a["weighted_penalty_per_seq"][group_row_idx].item()),
                        f"seq_reward_after_penalty_lambda_{lambda_a_key}": float(result_a["seq_reward_after_penalty"][group_row_idx].item()),
                        f"group_mean_lambda_{lambda_a_key}": float(result_a["group_mean"][group_row_idx].item()),
                        f"group_std_lambda_{lambda_a_key}": float(result_a["group_std"][group_row_idx].item()),
                        f"advantage_lambda_{lambda_a_key}": float(result_a["advantage_per_seq"][group_row_idx].item()),
                        f"weighted_penalty_lambda_{lambda_b_key}": float(result_b["weighted_penalty_per_seq"][group_row_idx].item()),
                        f"seq_reward_after_penalty_lambda_{lambda_b_key}": float(result_b["seq_reward_after_penalty"][group_row_idx].item()),
                        f"group_mean_lambda_{lambda_b_key}": float(result_b["group_mean"][group_row_idx].item()),
                        f"group_std_lambda_{lambda_b_key}": float(result_b["group_std"][group_row_idx].item()),
                        f"advantage_lambda_{lambda_b_key}": float(result_b["advantage_per_seq"][group_row_idx].item()),
                    }
                )

            case_payload = {
                "row_idx": int(row_idx),
                "uid": uid,
                "input_preview": None if input_list is None else _format_text_preview(input_list[row_idx]),
                "output_preview": None if output_list is None else _format_text_preview(output_list[row_idx]),
                "pred_preview": None if pred_list is None else _format_text_preview(pred_list[row_idx]),
                "score": None if score_list is None else score_list[row_idx],
                "prompt_length": None if prompt_length_list is None else prompt_length_list[row_idx],
                "response_length": None if response_length_list is None else response_length_list[row_idx],
                "last_response_token_idx": int(last_response_token_idx[row_idx].item()),
                "seq_score_before_penalty": float(seq_score_before_penalty[row_idx].item()),
                "raw_penalty_per_seq": float(raw_penalty_per_seq[row_idx].item()),
                "normalized_penalty_per_seq": None
                if normalized_penalty_per_seq is None
                else float(normalized_penalty_per_seq[row_idx].item()),
                f"weighted_penalty_lambda_{lambda_a_key}": float(result_a["weighted_penalty_per_seq"][row_idx].item()),
                f"seq_reward_after_penalty_lambda_{lambda_a_key}": float(result_a["seq_reward_after_penalty"][row_idx].item()),
                f"group_mean_lambda_{lambda_a_key}": float(result_a["group_mean"][row_idx].item()),
                f"group_std_lambda_{lambda_a_key}": float(result_a["group_std"][row_idx].item()),
                f"advantage_lambda_{lambda_a_key}": float(result_a["advantage_per_seq"][row_idx].item()),
                f"weighted_penalty_lambda_{lambda_b_key}": float(result_b["weighted_penalty_per_seq"][row_idx].item()),
                f"seq_reward_after_penalty_lambda_{lambda_b_key}": float(result_b["seq_reward_after_penalty"][row_idx].item()),
                f"group_mean_lambda_{lambda_b_key}": float(result_b["group_mean"][row_idx].item()),
                f"group_std_lambda_{lambda_b_key}": float(result_b["group_std"][row_idx].item()),
                f"advantage_lambda_{lambda_b_key}": float(result_b["advantage_per_seq"][row_idx].item()),
                f"advantage_abs_diff_{lambda_a_key}_vs_{lambda_b_key}": float(abs_diff[row_idx].item()),
                "group_cases": group_cases,
            }
            selected_cases.append(case_payload)

    return {
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "diff_tolerance": float(diff_tolerance),
        "num_differing_cases": int(differing_indices.numel()),
        "num_total_cases": int(advantage_a.shape[0]),
        "selected_cases": selected_cases,
    }


def plot_lagrangian_lambda_sweep(
    lambda_sweep: dict[str, Any],
    output_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambda_values = [float(value) for value in lambda_sweep["lambdas"]]
    results = lambda_sweep["results"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)

    for lambda_value in lambda_values:
        result = results[f"{lambda_value:g}"]
        penalty = result["weighted_penalty_per_seq"].float()
        advantage = result["advantage_per_seq"].float()

        penalty = penalty[torch.isfinite(penalty)]
        advantage = advantage[torch.isfinite(advantage)]

        if penalty.numel() > 0:
            penalty_sorted = torch.sort(penalty).values.numpy()
            penalty_rank = np.linspace(0.0, 1.0, penalty_sorted.shape[0], endpoint=False)
            axes[0, 0].plot(penalty_rank, penalty_sorted, linewidth=1.5, label=f"lambda={lambda_value:g}")
            axes[0, 1].hist(
                penalty.numpy(),
                bins=80,
                density=True,
                histtype="step",
                linewidth=1.2,
                label=f"lambda={lambda_value:g}",
            )

        if advantage.numel() > 0:
            advantage_sorted = torch.sort(advantage).values.numpy()
            advantage_rank = np.linspace(0.0, 1.0, advantage_sorted.shape[0], endpoint=False)
            axes[1, 0].plot(advantage_rank, advantage_sorted, linewidth=1.5, label=f"lambda={lambda_value:g}")
            axes[1, 1].hist(
                advantage.numpy(),
                bins=80,
                density=True,
                histtype="step",
                linewidth=1.2,
                label=f"lambda={lambda_value:g}",
            )

    axes[0, 0].set_title("Weighted Penalty Per Sequence")
    axes[0, 0].set_xlabel("Sorted Rank Fraction")
    axes[0, 0].set_ylabel("Weighted Penalty")
    axes[0, 1].set_title("Weighted Penalty Distribution")
    axes[0, 1].set_xlabel("Weighted Penalty")
    axes[0, 1].set_ylabel("Density")
    axes[1, 0].set_title("GRPO Advantage Per Sequence")
    axes[1, 0].set_xlabel("Sorted Rank Fraction")
    axes[1, 0].set_ylabel("Advantage")
    axes[1, 1].set_title("GRPO Advantage Distribution")
    axes[1, 1].set_xlabel("Advantage")
    axes[1, 1].set_ylabel("Density")

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Lagrangian Lambda Sweep "
        f"(ratio={lambda_sweep['config']['ratio_type']}, normalize={lambda_sweep['config']['normalize_mode']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _prepare_scatter_xy(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    return x, y


def _crop_by_x_quantile(
    x: torch.Tensor,
    y: torch.Tensor,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.numel() < 2:
        return x, y

    q = torch.tensor([lower_q, upper_q], dtype=torch.float32, device=x.device)
    x_low, x_high = torch.quantile(x.float(), q).tolist()
    keep = (x >= x_low) & (x <= x_high)
    return x[keep], y[keep]


def _format_token_str(token_str: str, max_len: int = 32) -> str:
    token_str = token_str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    if token_str == "":
        token_str = "<empty>"
    if len(token_str) > max_len:
        token_str = token_str[: max_len - 3] + "..."
    return token_str


def _format_text_preview(value: Any, max_len: int = 512) -> Any:
    if value is None:
        return None
    text = str(value).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _build_token_info_decoder(tokenizer_path: str | None):
    if not tokenizer_path:
        return lambda token_id: {
            "token_piece": None,
            "token_text": None,
            "token_piece_visible": None,
            "token_text_visible": None,
        }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _decode(token_id: int) -> dict[str, str | None]:
        token_piece = tokenizer.convert_ids_to_tokens(int(token_id))
        token_text = tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        return {
            "token_piece": None if token_piece is None else str(token_piece),
            "token_text": token_text,
            "token_piece_visible": None if token_piece is None else _format_token_str(str(token_piece), max_len=256),
            "token_text_visible": _format_token_str(token_text, max_len=256),
        }

    return _decode


def compute_old_minus_rollout_token_extremes(
    merged: dict[str, Any],
    tokenizer_path: str | None = None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
) -> dict[str, list[dict[str, Any]]]:
    response_mask = _require_tensor(merged, "response_mask").bool()
    old_log_probs = _broadcast_to_response_shape(_require_tensor(merged, "old_log_probs").float(), response_mask, "old_log_probs")
    rollout_log_probs = _broadcast_to_response_shape(
        _require_tensor(merged, "rollout_log_probs").float(), response_mask, "rollout_log_probs"
    )
    response_token_ids = _broadcast_to_response_shape(_require_tensor(merged, "response_token_ids").long(), response_mask, "response_token_ids")

    valid_mask = response_mask & torch.isfinite(old_log_probs) & torch.isfinite(rollout_log_probs)
    token_ids = response_token_ids[valid_mask].reshape(-1)
    old_vals = old_log_probs[valid_mask].reshape(-1)
    rollout_vals = rollout_log_probs[valid_mask].reshape(-1)
    diff_vals = old_vals - rollout_vals

    if diff_vals.numel() == 0:
        return {"largest": [], "smallest": []}

    top_k = min(top_k, diff_vals.numel())
    largest_indices = torch.topk(diff_vals, k=top_k, largest=True).indices
    smallest_indices = torch.topk(diff_vals, k=top_k, largest=False).indices
    decode_token_info = _build_token_info_decoder(tokenizer_path)

    def _gather(indices: torch.Tensor) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for rank, idx in enumerate(indices.tolist(), start=1):
            token_id = int(token_ids[idx].item())
            token_info = decode_token_info(token_id)
            entries.append(
                {
                    "rank": rank,
                    "token_id": token_id,
                    **token_info,
                    "old_log_prob": float(old_vals[idx].item()),
                    "rollout_log_prob": float(rollout_vals[idx].item()),
                    "old_minus_rollout": float(diff_vals[idx].item()),
                }
            )
        return entries

    return {
        "largest": _gather(largest_indices),
        "smallest": _gather(smallest_indices),
    }


def save_old_minus_rollout_token_extremes_json(
    token_extremes: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(token_extremes, f, ensure_ascii=False, indent=2)
    return output_path


def _build_rollout_step_scatter_plot_specs(
    merged: dict[str, Any],
) -> list[tuple[str, str, torch.Tensor, str, torch.Tensor]]:
    response_mask = _require_tensor(merged, "response_mask")
    old_log_probs = _masked_flatten(_require_tensor(merged, "old_log_probs"), response_mask, "old_log_probs")
    rollout_log_probs = _masked_flatten(_require_tensor(merged, "rollout_log_probs"), response_mask, "rollout_log_probs")
    penalty = _masked_flatten(_require_tensor(merged, "penalty"), response_mask, "penalty")
    advantages = _masked_flatten(_require_tensor(merged, "advantages"), response_mask, "advantages")

    exp_old = torch.exp(old_log_probs)
    exp_rollout = torch.exp(rollout_log_probs)
    exp_old_minus_exp_rollout = exp_old - exp_rollout
    old_minus_rollout = old_log_probs - rollout_log_probs

    return [
        ("exp(old) vs penalty", "exp(old)", exp_old, "penalty", penalty),
        ("exp(old) vs advantages", "exp(old)", exp_old, "advantages", advantages),
        ("exp(old) vs exp(rollout)", "exp(old)", exp_old, "exp(rollout)", exp_rollout),
        ("exp(old)-exp(rollout) vs penalty", "exp(old)-exp(rollout)", exp_old_minus_exp_rollout, "penalty", penalty),
        ("exp(old)-exp(rollout) vs advantages", "exp(old)-exp(rollout)", exp_old_minus_exp_rollout, "advantages", advantages),
        ("exp(old)-exp(rollout) vs exp(old)", "exp(old)-exp(rollout)", exp_old_minus_exp_rollout, "exp(old)", exp_old),
        ("old-rollout vs penalty", "old-rollout", old_minus_rollout, "penalty", penalty),
        ("old-rollout vs advantages", "old-rollout", old_minus_rollout, "advantages", advantages),
        ("old-rollout vs exp(old)", "old-rollout", old_minus_rollout, "exp(old)", exp_old),
    ]


def plot_rollout_step_scatter_grid(
    merged: dict[str, Any],
    output_path: str | Path,
    crop_extreme_x_quantiles: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = _build_rollout_step_scatter_plot_specs(merged)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=200)
    for idx, (ax, (title, xlabel, x, ylabel, y)) in enumerate(zip(axes.flat, plot_specs, strict=True)):
        x_plot, y_plot = _prepare_scatter_xy(x, y)
        if crop_extreme_x_quantiles and idx >= 3:
            x_plot, y_plot = _crop_by_x_quantile(x_plot, y_plot, lower_q=0.01, upper_q=0.99)
        if x_plot.numel() == 0:
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.scatter(
                x_plot.numpy(),
                y_plot.numpy(),
                s=8,
                alpha=0.3,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rollout_step_scatter_grid_raw(
    merged: dict[str, Any],
    output_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = _build_rollout_step_scatter_plot_specs(merged)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=200)
    for ax, (title, xlabel, x, ylabel, y) in zip(axes.flat, plot_specs, strict=True):
        x_plot, y_plot = _prepare_scatter_xy(x, y)
        if x_plot.numel() == 0:
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.scatter(
                x_plot.numpy(),
                y_plot.numpy(),
                s=8,
                alpha=0.3,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and merge rollout step JSONL and PT data.")
    parser.add_argument("--base_rollout_data_dir", default="/mnt/ws-jfs/dsh-jfs/lrz/verl/rollout/CMDP", help="Directory containing rollout step files.")
    parser.add_argument("--rollout_data_dir", "-d", default="", help="Directory containing rollout step files.")
    parser.add_argument("--step", "-s", type=int, default=1, help="Rollout step to load, for example 1.")
    parser.add_argument(
        "--output_pt",
        default=None,
        help="Optional path to save the merged dict with torch.save(...).",
    )
    parser.add_argument(
        "--output_figure",
        default=None,
        help="Optional path to save the 3x3 scatter figure. Defaults to {rollout_data_dir}/{step}_scatter.png.",
    )
    parser.add_argument(
        "--output_raw_figure",
        default=None,
        help="Optional path to save the raw 3x3 scatter figure. Defaults to {rollout_data_dir}/{step}_scatter_raw.png.",
    )
    parser.add_argument(
        "--crop_scatter_x_quantiles",
        action="store_true",
        help="If set, crop x-axis extremes to the 1%%-99%% quantile range for scatter subplots 4-9.",
    )
    parser.add_argument(
        "--output_token_json",
        default=None,
        help="Optional path to save the old-rollout token extreme json. Defaults to {rollout_data_dir}/{step}_mismatch_top_tokens.json.",
    )
    parser.add_argument(
        "--output_lambda_figure",
        default=None,
        help="Optional path to save the Lagrangian lambda sweep figure. Defaults to {rollout_data_dir}/{step}_lag_lambda_sweep.png.",
    )
    parser.add_argument(
        "--output_lambda_compare_json",
        default=None,
        help="Optional path to save detailed lambda comparison cases. Defaults to {rollout_data_dir}/{step}_lambda_compare_0.5_vs_1.0.json.",
    )
    parser.add_argument(
        "--output_mixed_lambda_figure",
        default=None,
        help="Optional path to save the mixed-prompt-only Lagrangian lambda sweep figure. Defaults to {rollout_data_dir}/{step}_lag_lambda_sweep_mixed.png.",
    )
    parser.add_argument(
        "--output_mixed_lambda_compare_json",
        default=None,
        help="Optional path to save detailed mixed-prompt lambda comparison cases. Defaults to {rollout_data_dir}/{step}_lambda_compare_0.5_vs_1.0_mixed.json.",
    )
    parser.add_argument(
        "--lagrangian_lambdas",
        default="0.1,0.3,0.4,0.5,1.0",
        help="Comma-separated lambda values for the offline Lagrangian sweep.",
    )
    parser.add_argument(
        "--compare_lambda_a",
        type=float,
        default=0.5,
        help="First lambda used for detailed advantage-difference comparison.",
    )
    parser.add_argument(
        "--compare_lambda_b",
        type=float,
        default=1.0,
        help="Second lambda used for detailed advantage-difference comparison.",
    )
    parser.add_argument(
        "--max_compare_cases",
        type=int,
        default=3,
        help="Maximum number of differing cases to print in detail.",
    )
    parser.add_argument(
        "--grpo_lagrangian_c",
        type=float,
        default=DEFAULT_LAGRANGIAN_C,
        help="Constraint budget c used to initialize the offline Lagrangian operator.",
    )
    parser.add_argument(
        "--grpo_lagrangian_ratio",
        default=DEFAULT_LAGRANGIAN_RATIO,
        help="Penalty ratio_type used for offline Lagrangian analysis.",
    )
    parser.add_argument(
        "--grpo_lagrangian_buffer_mode",
        default=DEFAULT_LAGRANGIAN_BUFFER_MODE,
        help="Buffer mode used to initialize the offline Lagrangian operator.",
    )
    parser.add_argument(
        "--normalize_mode",
        default=DEFAULT_NORMALIZE_MODE,
        help="normalize_mode used for offline Lagrangian analysis.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/mnt/ws-jfs/dsh-jfs/lrz/hf_models/Qwen3-8B",
        help="Optional tokenizer path used to decode response token ids into token strings.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if "/mnt" not in args.rollout_data_dir:
        args.rollout_data_dir = os.path.join(args.base_rollout_data_dir, args.rollout_data_dir)
    merged = load_rollout_step_data(args.rollout_data_dir, args.step)

    token_extremes = compute_old_minus_rollout_token_extremes(
        merged,
        tokenizer_path=args.tokenizer_path,
        top_k=DEFAULT_TOP_K_TOKENS,
    )
    lagrangian_lambda_sweep = compute_lagrangian_lambda_sweep(
        merged,
        lambdas=_parse_float_csv(args.lagrangian_lambdas),
        constraint_budget_c=args.grpo_lagrangian_c,
        ratio_type=args.grpo_lagrangian_ratio,
        buffer_mode=args.grpo_lagrangian_buffer_mode,
        normalize_mode=args.normalize_mode,
    )
    lambda_compare = analyze_lambda_difference_cases(
        merged,
        lagrangian_lambda_sweep,
        lambda_a=args.compare_lambda_a,
        lambda_b=args.compare_lambda_b,
        max_cases=args.max_compare_cases,
    )
    mixed_merged, mixed_prompt_summary = select_mixed_prompt_subset(merged)
    mixed_lagrangian_lambda_sweep = None
    mixed_lambda_compare = None
    if mixed_prompt_summary["mixed_response_count"] > 0:
        mixed_lagrangian_lambda_sweep = compute_lagrangian_lambda_sweep(
            mixed_merged,
            lambdas=_parse_float_csv(args.lagrangian_lambdas),
            constraint_budget_c=args.grpo_lagrangian_c,
            ratio_type=args.grpo_lagrangian_ratio,
            buffer_mode=args.grpo_lagrangian_buffer_mode,
            normalize_mode=args.normalize_mode,
        )
        mixed_lambda_compare = analyze_lambda_difference_cases(
            mixed_merged,
            mixed_lagrangian_lambda_sweep,
            lambda_a=args.compare_lambda_a,
            lambda_b=args.compare_lambda_b,
            max_cases=args.max_compare_cases,
        )
    merged["_analysis"] = {
        "old_minus_rollout_token_extremes": token_extremes,
        "lagrangian_lambda_sweep": lagrangian_lambda_sweep,
        "lambda_compare": lambda_compare,
        "mixed_prompt_summary": mixed_prompt_summary,
        "mixed_lagrangian_lambda_sweep": mixed_lagrangian_lambda_sweep,
        "mixed_lambda_compare": mixed_lambda_compare,
    }

    if args.output_pt:
        output_path = Path(args.output_pt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, output_path)
        print(f"Saved merged rollout data to {output_path}")

    output_figure = args.output_figure
    if output_figure is None:
        output_figure = os.path.join(args.rollout_data_dir, f"{args.step}_scatter.png")
    figure_path = plot_rollout_step_scatter_grid(
        merged,
        output_figure,
        crop_extreme_x_quantiles=args.crop_scatter_x_quantiles,
    )
    print(f"Saved scatter figure to {figure_path}")

    output_raw_figure = args.output_raw_figure
    if output_raw_figure is None:
        output_raw_figure = os.path.join(args.rollout_data_dir, f"{args.step}_scatter_raw.png")
    raw_figure_path = plot_rollout_step_scatter_grid_raw(merged, output_raw_figure)
    print(f"Saved raw scatter figure to {raw_figure_path}")

    output_token_json = args.output_token_json
    if output_token_json is None:
        output_token_json = os.path.join(args.rollout_data_dir, f"{args.step}_mismatch_top_tokens.json")
    token_json_path = save_old_minus_rollout_token_extremes_json(token_extremes, output_token_json)
    print(f"Saved token extreme json to {token_json_path}")

    output_lambda_figure = args.output_lambda_figure
    if output_lambda_figure is None:
        output_lambda_figure = os.path.join(args.rollout_data_dir, f"{args.step}_lag_lambda_sweep.png")
    lambda_figure_path = plot_lagrangian_lambda_sweep(lagrangian_lambda_sweep, output_lambda_figure)
    print(f"Saved Lagrangian lambda sweep figure to {lambda_figure_path}")

    compare_lambda_a_key = f"{args.compare_lambda_a:g}"
    compare_lambda_b_key = f"{args.compare_lambda_b:g}"
    output_lambda_compare_json = args.output_lambda_compare_json
    if output_lambda_compare_json is None:
        output_lambda_compare_json = os.path.join(
            args.rollout_data_dir,
            f"{args.step}_lambda_compare_{compare_lambda_a_key}_vs_{compare_lambda_b_key}.json",
        )
    output_lambda_compare_json = Path(output_lambda_compare_json)
    output_lambda_compare_json.parent.mkdir(parents=True, exist_ok=True)
    with output_lambda_compare_json.open("w", encoding="utf-8") as f:
        json.dump(lambda_compare, f, ensure_ascii=False, indent=2)
    print(f"Saved lambda comparison json to {output_lambda_compare_json}")

    print(
        f"Lambda comparison {compare_lambda_a_key} vs {compare_lambda_b_key}: "
        f"{lambda_compare['num_differing_cases']} / {lambda_compare['num_total_cases']} cases differ"
    )
    if lambda_compare["selected_cases"]:
        print("Top differing cases:")
        print(json.dumps(lambda_compare["selected_cases"], ensure_ascii=False, indent=2))
    else:
        print("No differing cases found under the configured tolerance.")

    print(
        "Mixed prompt summary: "
        f"{mixed_prompt_summary['mixed_prompt_count']} mixed / "
        f"{mixed_prompt_summary['total_prompt_count']} total prompts, "
        f"{mixed_prompt_summary['mixed_response_count']} mixed responses"
    )
    if mixed_lagrangian_lambda_sweep is not None and mixed_lambda_compare is not None:
        output_mixed_lambda_figure = args.output_mixed_lambda_figure
        if output_mixed_lambda_figure is None:
            output_mixed_lambda_figure = os.path.join(args.rollout_data_dir, f"{args.step}_lag_lambda_sweep_mixed.png")
        mixed_lambda_figure_path = plot_lagrangian_lambda_sweep(mixed_lagrangian_lambda_sweep, output_mixed_lambda_figure)
        print(f"Saved mixed-prompt Lagrangian lambda sweep figure to {mixed_lambda_figure_path}")

        output_mixed_lambda_compare_json = args.output_mixed_lambda_compare_json
        if output_mixed_lambda_compare_json is None:
            output_mixed_lambda_compare_json = os.path.join(
                args.rollout_data_dir,
                f"{args.step}_lambda_compare_{compare_lambda_a_key}_vs_{compare_lambda_b_key}_mixed.json",
            )
        output_mixed_lambda_compare_json = Path(output_mixed_lambda_compare_json)
        output_mixed_lambda_compare_json.parent.mkdir(parents=True, exist_ok=True)
        with output_mixed_lambda_compare_json.open("w", encoding="utf-8") as f:
            json.dump(mixed_lambda_compare, f, ensure_ascii=False, indent=2)
        print(f"Saved mixed-prompt lambda comparison json to {output_mixed_lambda_compare_json}")

        print(
            f"Mixed prompt lambda comparison {compare_lambda_a_key} vs {compare_lambda_b_key}: "
            f"{mixed_lambda_compare['num_differing_cases']} / {mixed_lambda_compare['num_total_cases']} cases differ"
        )
        if mixed_lambda_compare["selected_cases"]:
            print("Top differing mixed-prompt cases:")
            print(json.dumps(mixed_lambda_compare["selected_cases"], ensure_ascii=False, indent=2))
        else:
            print("No differing mixed-prompt cases found under the configured tolerance.")
    else:
        print("No mixed prompts found, skipped mixed-prompt lambda sweep.")

    print("Merged rollout data keys:")
    for key in sorted(merged.keys()):
        print(key)


if __name__ == "__main__":
    main()
