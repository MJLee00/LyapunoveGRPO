# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""GRPO + scalar Lagrangian multiplier for constrained CMDP-style training.

Shaped outcome: R_tilde = R - lambda * p
where p is the normalized penalty computed from the rollout/train mismatch signal.
The penalty can be applied either per sequence (scattered to the last valid token) or
per token (length-normalized across the response).

The dual variable is always a scalar updated by:
    lambda <- clip(lambda + eta * (g_bar - c), lambda_min, lambda_max)

When ``enable_lambda2`` is enabled, negative sequence-level mismatch trajectories use an
independent lambda2 / budget / buffer / learning rate. The original lambda continues to
handle only positive mismatch trajectories.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.rollout_corr_helper import (
    compute_raw_penalty_per_token,
    compute_raw_penalty_per_sequence,
    normalize_penalty_level,
    normalize_penalty_per_sequence,
    scale_token_penalty_by_response_length,
    scatter_sequence_values_to_last_valid_token,
)


_DEFAULT_BUFFER_MONITOR_MODES = (
    "mean",
    "p5", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p85", "p90", "p95",
    "min", "max",
    "gt0_mean", "lt0_mean", "abs_mean",
)


def apply_lagrangian_penalty_to_rewards(
    batch: DataProto,
    lagrange_lambda: float,
    penalty_values: Optional[torch.Tensor],
    penalty_mask: Optional[torch.Tensor] = None,
    metric_suffix: str = "",
) -> dict[str, float]:
    """Subtract ``lambda * penalty`` from the reward tensor."""
    weighted_penalty, metrics = build_lagrangian_weighted_penalty(
        reference_dtype=batch.batch["token_level_rewards"].dtype,
        lagrange_lambda=lagrange_lambda,
        penalty_values=penalty_values,
        penalty_mask=penalty_mask,
        metric_suffix=metric_suffix,
    )
    penalty_2d = materialize_lagrangian_penalty_to_token_tensor(
        weighted_penalty=weighted_penalty,
        response_mask=batch.batch["response_mask"],
        dtype=batch.batch["token_level_rewards"].dtype,
    )
    batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] - penalty_2d
    if "rollout_penalty" in batch.batch:
        batch.batch["rollout_penalty"] = batch.batch["rollout_penalty"] + penalty_2d
    else:
        batch.batch["rollout_penalty"] = penalty_2d.clone()
    return metrics


def materialize_lagrangian_penalty_to_token_tensor(
    weighted_penalty: torch.Tensor,
    response_mask: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Materialize a weighted penalty tensor onto response tokens."""
    if weighted_penalty.ndim == 1:
        return scatter_sequence_values_to_last_valid_token(
            values_per_seq=weighted_penalty,
            response_mask=response_mask,
            dtype=dtype,
        )
    if weighted_penalty.ndim == 2:
        if weighted_penalty.shape != response_mask.shape:
            raise ValueError(f"weighted_penalty shape {tuple(weighted_penalty.shape)} must match response_mask shape {tuple(response_mask.shape)}")
        out = weighted_penalty.to(dtype=dtype or weighted_penalty.dtype)
        return out * response_mask.to(dtype=out.dtype, device=out.device)
    raise ValueError(f"weighted_penalty must be 1D or 2D, got shape {tuple(weighted_penalty.shape)}")


def build_lagrangian_penalty_tensor(
    reference_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    lagrange_lambda: float,
    penalty_values: Optional[torch.Tensor],
    penalty_mask: Optional[torch.Tensor] = None,
    metric_suffix: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build a token-level penalty tensor and its summary metrics."""
    if penalty_values is None:
        return torch.zeros_like(reference_tensor), {}

    weighted_penalty, metrics = build_lagrangian_weighted_penalty(
        reference_dtype=reference_tensor.dtype,
        lagrange_lambda=lagrange_lambda,
        penalty_values=penalty_values,
        penalty_mask=penalty_mask,
        metric_suffix=metric_suffix,
    )
    penalty_2d = materialize_lagrangian_penalty_to_token_tensor(
        weighted_penalty=weighted_penalty,
        response_mask=response_mask,
        dtype=reference_tensor.dtype,
    )
    return penalty_2d, metrics


def build_lagrangian_weighted_penalty(
    reference_dtype: torch.dtype,
    lagrange_lambda: float,
    penalty_values: Optional[torch.Tensor],
    penalty_mask: Optional[torch.Tensor] = None,
    metric_suffix: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build a weighted penalty tensor with the same shape as ``penalty_values``."""
    if penalty_values is None:
        return torch.zeros(0, device="cpu", dtype=reference_dtype), {}

    if penalty_mask is None:
        active_mask = torch.ones_like(penalty_values, dtype=torch.bool)
    else:
        active_mask = penalty_mask.to(dtype=torch.bool, device=penalty_values.device)
        if active_mask.shape != penalty_values.shape:
            raise ValueError(f"penalty_mask shape {active_mask.shape} must match penalty shape {penalty_values.shape}")

    weighted_penalty = torch.zeros_like(penalty_values, dtype=reference_dtype)
    weighted_penalty[active_mask] = (lagrange_lambda * penalty_values[active_mask]).to(weighted_penalty.dtype)

    selected_penalty = penalty_values[active_mask]
    selected_weighted = weighted_penalty[active_mask]
    if selected_penalty.numel() == 0:
        return weighted_penalty, {f"lagrangian/lambda_applied{metric_suffix}": float(lagrange_lambda)}

    metrics = {
        f"lagrangian/penalty_mean{metric_suffix}": selected_penalty.mean().item(),
        f"lagrangian/penalty_abs_mean{metric_suffix}": selected_penalty.abs().mean().item(),
        f"lagrangian/penalty_pos_mean{metric_suffix}": selected_penalty[selected_penalty > 0].mean().item() if selected_penalty[selected_penalty > 0].numel() else 0,
        f"lagrangian/penalty_neg_mean{metric_suffix}": selected_penalty[selected_penalty < 0].mean().item() if selected_penalty[selected_penalty < 0].numel() else 0,
        f"lagrangian/penalty_max{metric_suffix}": selected_penalty.max().item(),
        f"lagrangian/penalty_min{metric_suffix}": selected_penalty.min().item(),
        f"lagrangian/penalty_after_lagrangian_mean{metric_suffix}": selected_weighted.mean().item(),
        f"lagrangian/penalty_after_lagrangian_abs_mean{metric_suffix}": selected_weighted.abs().mean().item(),
        f"lagrangian/penalty_after_lagrangian_pos_mean{metric_suffix}": selected_weighted[selected_weighted > 0].mean().item() if selected_weighted[selected_weighted > 0].numel() else 0,
        f"lagrangian/penalty_after_lagrangian_neg_mean{metric_suffix}": selected_weighted[selected_weighted < 0].mean().item() if selected_weighted[selected_weighted < 0].numel() else 0,
        f"lagrangian/penalty_after_lagrangian_max{metric_suffix}": selected_weighted.max().item(),
        f"lagrangian/penalty_after_lagrangian_min{metric_suffix}": selected_weighted.min().item(),
        f"lagrangian/lambda_applied{metric_suffix}": float(lagrange_lambda),
    }
    return weighted_penalty, metrics


def build_lagrangian_weighted_penalty_per_sequence(
    reference_dtype: torch.dtype,
    lagrange_lambda: float,
    penalty_per_seq: Optional[torch.Tensor],
    sequence_mask: Optional[torch.Tensor] = None,
    metric_suffix: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Backward-compatible wrapper for sequence-shaped penalties."""
    return build_lagrangian_weighted_penalty(
        reference_dtype=reference_dtype,
        lagrange_lambda=lagrange_lambda,
        penalty_values=penalty_per_seq,
        penalty_mask=sequence_mask,
        metric_suffix=metric_suffix,
    )


def _cfg_get(cfg: dict[str, Any], key: str, default: Any) -> Any:
    value = cfg.get(key)
    return default if value is None else value


def _cfg_get_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_buffer_mode(mode: Any) -> str:
    mode_str = str(mode).strip().lower()
    if "mean" in mode_str or mode_str == "min" or mode_str == "max":
        return mode_str
    match = re.fullmatch(r"p(\d{1,3})", mode_str)
    if match is None:
        raise ValueError(f"buffer mode must be 'mean' or 'pXX', got {mode!r}")
    quantile = int(match.group(1))
    if not 0 <= quantile <= 100:
        raise ValueError(f"buffer quantile must be between 0 and 100, got {mode!r}")
    return f"p{quantile}"


@dataclass(frozen=True)
class _PenaltyTerm:
    lagrange_lambda: float
    penalty_values: torch.Tensor
    penalty_mask: torch.Tensor
    metric_suffix: str


@dataclass
class _DualVariableState:
    constraint_budget_c: float
    lambda_init: float
    lambda_lr: float
    lambda_min: float
    lambda_max: float
    update_every_z: int
    buffer_size: int
    buffer_mode: str
    metric_suffix: str = ""
    buffer_monitor_modes: tuple[str, ...] = field(init=False)
    lambda_value: float = field(init=False)
    buffers: dict[str, deque[float]] = field(init=False)
    buffer: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.constraint_budget_c = float(self.constraint_budget_c)
        self.lambda_init = float(self.lambda_init)
        self.lambda_lr = float(self.lambda_lr)
        self.lambda_min = float(self.lambda_min)
        self.lambda_max = float(self.lambda_max)
        if self.lambda_min > self.lambda_max:
            raise ValueError(f"lambda_min ({self.lambda_min}) must be <= lambda_max ({self.lambda_max})")

        self.update_every_z = int(self.update_every_z)
        self.buffer_size = int(self.buffer_size)
        self.buffer_mode = _normalize_buffer_mode(self.buffer_mode)

        monitor_modes = list(_DEFAULT_BUFFER_MONITOR_MODES)
        if self.buffer_mode not in monitor_modes:
            monitor_modes.append(self.buffer_mode)
        self.buffer_monitor_modes = tuple(monitor_modes)

        self.lambda_value = float(np.clip(self.lambda_init, self.lambda_min, self.lambda_max))
        self.buffers = {mode: deque(maxlen=self.buffer_size) for mode in self.buffer_monitor_modes}
        self.buffer = self.buffers[self.buffer_mode]

    def current_lambda(self) -> float:
        return float(np.clip(self.lambda_value, self.lambda_min, self.lambda_max))

    def _reduce_constraint(self, penalties: torch.Tensor, mode: Optional[str] = None) -> float:
        if penalties.numel() == 0:
            return 0.0

        resolved_mode = self.buffer_mode if mode is None else _normalize_buffer_mode(mode)
        if resolved_mode.startswith("p"):
            return float(torch.quantile(penalties.float(), int(resolved_mode[1:]) / 100.0).item())
        if resolved_mode == "min":
            return float(penalties.min().item())
        if resolved_mode == "max":
            return float(penalties.max().item())
        if resolved_mode == "mean":
            return float(penalties.mean().item())
        if resolved_mode == "gt0_mean":
            selected = penalties[penalties > 0]
            return float(selected.mean().item()) if selected.numel() > 0 else 0.0
        if resolved_mode == "lt0_mean":
            selected = penalties[penalties < 0]
            return float(selected.mean().item()) if selected.numel() > 0 else 0.0
        if resolved_mode == "abs_mean":
            return float(penalties.abs().mean().item())
        raise ValueError(f"Unsupported buffer mode: {resolved_mode}")

    def record_constraint_stats(self, penalties: torch.Tensor) -> None:
        if penalties.numel() == 0:
            return
        for mode in self.buffer_monitor_modes:
            self.buffers[mode].append(self._reduce_constraint(penalties, mode=mode))

    def _buffer_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for mode, values in self.buffers.items():
            if len(values) == 0:
                continue
            metrics[f"lagrangian/g_bar_buffer{self.metric_suffix}_{mode}"] = float(np.mean(values))

        active_g_bar = metrics.get(f"lagrangian/g_bar_buffer{self.metric_suffix}_{self.buffer_mode}")
        if active_g_bar is not None:
            metrics[f"lagrangian/g_bar_buffer{self.metric_suffix}"] = active_g_bar
            metrics[f"lagrangian/constraint_c{self.metric_suffix}"] = self.constraint_budget_c
            metrics[f"lagrangian/violation{self.metric_suffix}"] = active_g_bar - self.constraint_budget_c
        return metrics

    def maybe_update(self, global_step: int) -> dict[str, float]:
        metrics = {f"lagrangian/lambda{self.metric_suffix}": self.current_lambda()}
        metrics.update(self._buffer_metrics())

        if self.update_every_z <= 0 or len(self.buffer) == 0 or global_step % self.update_every_z != 0:
            return metrics

        g_bar = float(np.mean(self.buffer))
        violation = g_bar - self.constraint_budget_c
        self.lambda_value = float(np.clip(self.lambda_value + self.lambda_lr * violation, self.lambda_min, self.lambda_max))

        metrics[f"lagrangian/lambda{self.metric_suffix}"] = self.current_lambda()
        metrics[f"lagrangian/g_bar_buffer{self.metric_suffix}"] = g_bar
        metrics[f"lagrangian/constraint_c{self.metric_suffix}"] = self.constraint_budget_c
        metrics[f"lagrangian/violation{self.metric_suffix}"] = violation
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "lambda_value": self.lambda_value,
            "buffer_mode": self.buffer_mode,
            "buffer_monitor_modes": list(self.buffer_monitor_modes),
            "buffers": {mode: list(buf) for mode, buf in self.buffers.items()},
            "constraint_budget_c": self.constraint_budget_c,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        lambda_value = payload.get("lambda_value", payload.get("lambda_dual", self.lambda_init))
        self.lambda_value = float(np.clip(float(lambda_value), self.lambda_min, self.lambda_max))

        for buf in self.buffers.values():
            buf.clear()

        saved_buffers = payload.get("buffers")
        if isinstance(saved_buffers, dict):
            for mode, values in saved_buffers.items():
                normalized_mode = _normalize_buffer_mode(mode)
                if normalized_mode not in self.buffers:
                    continue
                for value in values[-self.buffer_size :]:
                    self.buffers[normalized_mode].append(float(value))
            return

        legacy_buffer = payload.get("buffer", [])
        for value in legacy_buffer[-self.buffer_size :]:
            self.buffer.append(float(value))

    def summary(self) -> dict[str, Any]:
        suffix = self.metric_suffix or "1"
        return {
            f"lambda_value{self.metric_suffix}": self.lambda_value,
            f"buffer_mode{self.metric_suffix}": self.buffer_mode,
            f"buffer_lens{self.metric_suffix}": {mode: len(buf) for mode, buf in self.buffers.items()},
            f"constraint_budget_c{self.metric_suffix}": self.constraint_budget_c,
            "branch_name": f"lambda{suffix}",
        }


class GRPOLagrangianOperator:
    """Owns one or two scalar dual variables and applies them to GRPO rewards or advantages."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        shared_kwargs = {
            "lambda_min": float(_cfg_get(cfg, "lambda_min", 0.0)),
            "lambda_max": float(_cfg_get(cfg, "lambda_max", 1.0)),
            "update_every_z": int(_cfg_get(cfg, "operator_update_every_z", 1)),
            "buffer_size": int(_cfg_get(cfg, "buffer_size", 1)),
            "buffer_mode": _normalize_buffer_mode(_cfg_get(cfg, "buffer_mode", "mean")),
        }
        self.primary_state = _DualVariableState(
            constraint_budget_c=float(_cfg_get(cfg, "constraint_budget_c", 0.0)),
            lambda_init=float(_cfg_get(cfg, "lambda_init", 0.0)),
            lambda_lr=float(_cfg_get(cfg, "lambda_lr", 1e1)),
            metric_suffix="",
            **shared_kwargs,
        )

        self.secondary_state: Optional[_DualVariableState] = None
        if _cfg_get_bool(cfg, "enable_lambda2", False):
            self.secondary_state = _DualVariableState(
                constraint_budget_c=float(_cfg_get(cfg, "constraint_budget_c2", _cfg_get(cfg, "constraint_budget_c", 0.0))),
                lambda_init=float(_cfg_get(cfg, "lambda_init2", _cfg_get(cfg, "lambda_init", 0.0))),
                lambda_lr=float(_cfg_get(cfg, "lambda_lr2", _cfg_get(cfg, "lambda_lr", 1e1))),
                metric_suffix="2",
                **shared_kwargs,
            )

    def use_lambda2(self) -> bool:
        return self.secondary_state is not None

    def current_lambda(self) -> float:
        return self.primary_state.current_lambda()

    def current_lambda2(self) -> float:
        return 0.0 if self.secondary_state is None else self.secondary_state.current_lambda()

    def _build_penalty_terms(
        self,
        raw_penalty: torch.Tensor,
        max_ratio: Optional[float],
        normalize_mode: Optional[str],
        normalize_tanh_k: Optional[float],
        penalty_truncated: Optional[float | bool],
        penalty_mask: Optional[torch.Tensor] = None,
        normalize_by_response_length: bool = False,
        log_probs: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
    ) -> list[_PenaltyTerm]:
        if raw_penalty.numel() == 0:
            return []

        if penalty_mask is None:
            active_mask = torch.ones_like(raw_penalty, dtype=torch.bool)
        else:
            active_mask = penalty_mask.to(dtype=torch.bool, device=raw_penalty.device)
            if active_mask.shape != raw_penalty.shape:
                raise ValueError(f"penalty_mask shape {active_mask.shape} must match raw_penalty shape {raw_penalty.shape}")

        needs_token_scaling = raw_penalty.ndim == 2

        if not self.use_lambda2():
            penalty_values = normalize_penalty_per_sequence(
                raw_per_seq=torch.where(active_mask, raw_penalty, torch.zeros_like(raw_penalty)),
                max_ratio=max_ratio,
                normalize_mode=normalize_mode,
                normalize_tanh_k=normalize_tanh_k,
                penalty_truncated=penalty_truncated,
                log_probs=log_probs,
                response_mask=response_mask,
            )
            if penalty_values is None:
                return []
            if needs_token_scaling and normalize_by_response_length:
                penalty_values = scale_token_penalty_by_response_length(penalty_per_token=penalty_values, response_mask=active_mask)
            return [
                _PenaltyTerm(
                    lagrange_lambda=self.current_lambda(),
                    penalty_values=penalty_values,
                    penalty_mask=active_mask,
                    metric_suffix="",
                )
            ]

        terms: list[_PenaltyTerm] = []
        positive_mask = active_mask & (raw_penalty > 0)
        if positive_mask.any():
            positive_penalty = normalize_penalty_per_sequence(
                raw_per_seq=torch.where(positive_mask, raw_penalty, torch.zeros_like(raw_penalty)),
                max_ratio=max_ratio,
                normalize_mode=normalize_mode,
                normalize_tanh_k=normalize_tanh_k,
                penalty_truncated=penalty_truncated,
                log_probs=log_probs,
                response_mask=response_mask,
            )
            if positive_penalty is not None:
                if needs_token_scaling and normalize_by_response_length:
                    positive_penalty = scale_token_penalty_by_response_length(penalty_per_token=positive_penalty, response_mask=active_mask)
                terms.append(
                    _PenaltyTerm(
                        lagrange_lambda=self.current_lambda(),
                        penalty_values=positive_penalty,
                        penalty_mask=positive_mask,
                        metric_suffix="",
                    )
                )

        negative_mask = active_mask & (raw_penalty < 0)
        if negative_mask.any() and self.secondary_state is not None:
            negative_penalty = normalize_penalty_per_sequence(
                raw_per_seq=torch.where(negative_mask, -raw_penalty, torch.zeros_like(raw_penalty)),
                max_ratio=max_ratio,
                normalize_mode=normalize_mode,
                normalize_tanh_k=normalize_tanh_k,
                penalty_truncated=penalty_truncated,
                log_probs=log_probs,
                response_mask=response_mask,
            )
            if negative_penalty is not None:
                if needs_token_scaling and normalize_by_response_length:
                    negative_penalty = scale_token_penalty_by_response_length(penalty_per_token=negative_penalty, response_mask=active_mask)
                terms.append(
                    _PenaltyTerm(
                        lagrange_lambda=self.current_lambda2(),
                        penalty_values=negative_penalty,
                        penalty_mask=negative_mask,
                        metric_suffix="2",
                    )
                )

        return terms

    def record_batch_constraint_stats(self, raw_penalty: torch.Tensor, penalty_mask: Optional[torch.Tensor] = None) -> None:
        if raw_penalty is None or raw_penalty.numel() == 0:
            return

        if penalty_mask is None:
            selected_penalties = raw_penalty.reshape(-1)
        else:
            penalty_mask = penalty_mask.to(dtype=torch.bool, device=raw_penalty.device)
            if penalty_mask.shape != raw_penalty.shape:
                raise ValueError(f"penalty_mask shape {penalty_mask.shape} must match raw_penalty shape {raw_penalty.shape}")
            selected_penalties = raw_penalty[penalty_mask]
        if selected_penalties.numel() == 0:
            return

        if not self.use_lambda2():
            self.primary_state.record_constraint_stats(selected_penalties)
            return

        positive_penalties = selected_penalties[selected_penalties > 0]
        if positive_penalties.numel() > 0:
            self.primary_state.record_constraint_stats(positive_penalties)

        negative_penalties = (-selected_penalties[selected_penalties < 0]).abs()
        if negative_penalties.numel() > 0 and self.secondary_state is not None:
            self.secondary_state.record_constraint_stats(negative_penalties)

    def apply_to_batch(
        self,
        batch: DataProto,
        penalty_cost_start_step: Optional[int],
        global_step: Optional[int],
        clip_penalty_low: Optional[float],
        clip_penalty_high: Optional[float],
        normalize_mode: Optional[str] = None,
        normalize_tanh_k: Optional[float] = None,
        penalty_truncated: Optional[float | bool] = None,
    ) -> dict[str, float]:
        penalty_level = normalize_penalty_level(self.cfg.get("penalty_level", "sequence"))
        normalize_by_response_length = _cfg_get_bool(self.cfg, "normalize_by_response_length", False)
        if penalty_level == "token":
            raw_penalty = compute_raw_penalty_per_token(
                batch=batch,
                ratio_type=self.cfg.get("ratio_type", "seq_geo_no_exp"),
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
            )
            penalty_mask = batch.batch["response_mask"].to(dtype=torch.bool) if raw_penalty is not None else None
        else:
            raw_penalty = compute_raw_penalty_per_sequence(
                batch=batch,
                ratio_type=self.cfg.get("ratio_type", "seq_geo_no_exp"),
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
            )
            penalty_mask = None
        if raw_penalty is None:
            return {}

        self.record_batch_constraint_stats(raw_penalty, penalty_mask=penalty_mask)
        selected_penalty = raw_penalty if penalty_mask is None else raw_penalty[penalty_mask]
        metrics: dict[str, float] = {
            "lagrangian/num_gt0": len(selected_penalty[selected_penalty > 0]) if selected_penalty[selected_penalty > 0].numel() > 0 else 0,
            "lagrangian/num_lt0": len(selected_penalty[selected_penalty < 0]) if selected_penalty[selected_penalty < 0].numel() > 0 else 0,
        }
        for term in self._build_penalty_terms(
            raw_penalty=raw_penalty,
            max_ratio=self.cfg.get("max_ratio"),
            normalize_mode=normalize_mode,
            normalize_tanh_k=normalize_tanh_k,
            penalty_truncated=penalty_truncated,
            penalty_mask=penalty_mask,
            normalize_by_response_length=normalize_by_response_length,
            log_probs=batch.batch["old_log_probs"],
            response_mask=batch.batch["response_mask"],
        ):
            metrics.update(
                apply_lagrangian_penalty_to_rewards(
                    batch=batch,
                    lagrange_lambda=term.lagrange_lambda,
                    penalty_values=term.penalty_values,
                    penalty_mask=term.penalty_mask,
                    metric_suffix=term.metric_suffix,
                )
            )
        return metrics

    def apply_to_advantages(
        self,
        batch: DataProto,
        advantages: torch.Tensor,
        norm_adv_by_std_in_grpo: bool,
        penalty_cost_start_step: Optional[int],
        global_step: Optional[int],
        clip_penalty_low: Optional[float],
        clip_penalty_high: Optional[float],
        normalize_mode: Optional[str] = None,
        normalize_tanh_k: Optional[float] = None,
        penalty_truncated: Optional[float | bool] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        penalty_level = normalize_penalty_level(self.cfg.get("penalty_level", "sequence"))
        normalize_by_response_length = _cfg_get_bool(self.cfg, "normalize_by_response_length", False)
        if penalty_level == "token":
            raw_penalty = compute_raw_penalty_per_token(
                batch=batch,
                ratio_type=self.cfg.get("ratio_type", "seq_geo_no_exp"),
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
            )
            penalty_mask = batch.batch["response_mask"].to(dtype=torch.bool) if raw_penalty is not None else None
        else:
            raw_penalty = compute_raw_penalty_per_sequence(
                batch=batch,
                ratio_type=self.cfg.get("ratio_type", "seq_geo_no_exp"),
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
            )
            penalty_mask = None
        if raw_penalty is None:
            return advantages, {}

        if normalize_mode is not None and "samesign" in str(normalize_mode).strip().lower():
            response_mask = batch.batch["response_mask"].to(device=raw_penalty.device)
            prob_diff = torch.exp(batch.batch["old_log_probs"].float()) - torch.exp(batch.batch["rollout_log_probs"].float())
            if raw_penalty.ndim == 1:
                mask = response_mask.to(dtype=prob_diff.dtype)
                denom = mask.sum(dim=-1).clamp(min=1.0)
                adv_dir = (advantages.float() * mask).sum(dim=-1) / denom
                prob_dir = (prob_diff * mask).sum(dim=-1) / denom
            else:
                adv_dir = advantages.float()
                prob_dir = prob_diff
            same_sign = ((adv_dir * prob_dir) > 0)
            if str(normalize_mode).strip().lower() == "samesign0":
                same_sign = same_sign | (adv_dir == 0)
            if penalty_mask is not None:
                same_sign = same_sign & penalty_mask.to(device=same_sign.device)
            raw_penalty = torch.where(same_sign, raw_penalty, torch.zeros_like(raw_penalty))
            normalize_mode = "none"

        self.record_batch_constraint_stats(raw_penalty, penalty_mask=penalty_mask)
        penalty_advantages = torch.zeros_like(advantages)
        selected_penalty = raw_penalty if penalty_mask is None else raw_penalty[penalty_mask]
        metrics: dict[str, float] = {
            "lagrangian/num_gt0": len(selected_penalty[selected_penalty > 0]) if selected_penalty[selected_penalty > 0].numel() > 0 else 0,
            "lagrangian/num_lt0": len(selected_penalty[selected_penalty < 0]) if selected_penalty[selected_penalty < 0].numel() > 0 else 0,
        }
        for term in self._build_penalty_terms(
            raw_penalty=raw_penalty,
            max_ratio=self.cfg.get("max_ratio"),
            normalize_mode=normalize_mode,
            normalize_tanh_k=normalize_tanh_k,
            penalty_truncated=penalty_truncated,
            penalty_mask=penalty_mask,
            normalize_by_response_length=normalize_by_response_length,
            log_probs=batch.batch["old_log_probs"],
            response_mask=batch.batch["response_mask"],
        ):
            weighted_penalty, term_metrics = build_lagrangian_weighted_penalty(
                reference_dtype=advantages.dtype,
                lagrange_lambda=term.lagrange_lambda,
                penalty_values=term.penalty_values,
                penalty_mask=term.penalty_mask,
                metric_suffix=term.metric_suffix,
            )
            penalty_advantages = penalty_advantages + materialize_lagrangian_penalty_to_token_tensor(
                weighted_penalty=weighted_penalty,
                response_mask=batch.batch["response_mask"],
                dtype=advantages.dtype,
            )
            metrics.update(term_metrics)

        if "rollout_penalty" in batch.batch:
            batch.batch["rollout_penalty"] = batch.batch["rollout_penalty"] + penalty_advantages
        else:
            batch.batch["rollout_penalty"] = penalty_advantages.clone()
        return advantages - penalty_advantages, metrics

    def maybe_update(self, global_step: int) -> dict[str, float]:
        metrics = self.primary_state.maybe_update(global_step)
        if self.secondary_state is not None:
            metrics.update(self.secondary_state.maybe_update(global_step))
        return metrics

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 2,
            "enable_lambda2": self.use_lambda2(),
            "primary": self.primary_state.state_dict(),
        }
        if self.secondary_state is not None:
            payload["secondary"] = self.secondary_state.state_dict()
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        primary_payload = payload.get("primary", payload)
        self.primary_state.load_state_dict(primary_payload)

        if self.secondary_state is None:
            return

        secondary_payload = payload.get("secondary")
        if not isinstance(secondary_payload, dict):
            legacy_secondary_present = any(key in payload for key in ("lambda_dual2", "lambda_value2", "buffer2", "buffers2", "branch2"))
            secondary_payload = payload.get("branch2") if isinstance(payload.get("branch2"), dict) else None
            if secondary_payload is None and legacy_secondary_present:
                secondary_payload = {
                    "lambda_value": payload.get("lambda_value2", payload.get("lambda_dual2", self.current_lambda2())),
                    "buffer": payload.get("buffer2", []),
                    "buffers": payload.get("buffers2"),
                    "constraint_budget_c": payload.get("constraint_budget_c2", self.secondary_state.constraint_budget_c),
                }
        if isinstance(secondary_payload, dict):
            self.secondary_state.load_state_dict(secondary_payload)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

        summary = {"enable_lambda2": self.use_lambda2(), "primary": self.primary_state.summary()}
        if self.secondary_state is not None:
            summary["secondary"] = self.secondary_state.summary()
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str, cfg: dict[str, Any]) -> GRPOLagrangianOperator:
        operator = cls(cfg)
        if os.path.isfile(path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            operator.load_state_dict(payload)
        return operator


def grpo_lagrangian_config_from_algorithm(algorithm_cfg: Any) -> Optional[dict[str, Any]]:
    from omegaconf import DictConfig, OmegaConf

    gl = None
    if isinstance(algorithm_cfg, DictConfig) or OmegaConf.is_config(algorithm_cfg):
        gl = OmegaConf.select(algorithm_cfg, "grpo_lagrangian")
    elif isinstance(algorithm_cfg, dict):
        gl = algorithm_cfg.get("grpo_lagrangian")
    else:
        gl = getattr(algorithm_cfg, "grpo_lagrangian", None)
    if gl is None:
        return None

    config_dict = OmegaConf.to_container(gl, resolve=True) if OmegaConf.is_config(gl) else dict(gl)
    if not _cfg_get_bool(config_dict, "enable", False):
        return None
    return config_dict
