# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout Correction Helper Module

This module provides a complete pipeline to address **off-policy issues** in RL training,
including:
1. Policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)
2. Model update staleness (training on trajectories from older checkpoints)
3. General distribution shifts between data collection and training

Its core capabilities include computing importance sampling (IS) weights,
filtering outlier samples via rejection sampling (RS), and
tracking metrics to diagnose and correct off-policy issues.

## Core Capabilities
1. **Multi-Granularity Aggregation**:
   - Importance Sampling (IS):
        Token-level
        Sequence-level
   - Rejection Sampling (RS):
        Token-level
        Sequence/geometric (sequence-level geometric mean) — supports flexible outlier filtering.
2. **Catastrophic Outlier Veto**:
    Independent per-token veto mechanism — fully reject sequences containing tokens
    with extremely low IS weights (prevents catastrophic updates).
3. **Memory-Efficient Design**:
   - Log-space computations to avoid numerical overflow/underflow.
   - Fixed safety bounds (exp(±20)) for stable exponentiation.
   - Metrics calculated without large intermediate tensors (prevents CUDA OOM).
4. **Comprehensive Metrics Tracking**:
   - IS/RS statistics (mean/max/min, effective sample size ESS, rejection rate).
   - Off-policy diagnostics (KL divergence, perplexity PPL, log PPL difference, χ² divergence).
   - Sequence-level breakdowns (deviation from ideal weights, outlier fraction).


## Key Interfaces & Usage
- compute_rollout_correction_and_rejection_mask(): compute IS weights + rejection mask + veto.
- compute_rollout_correction_weights(): only compute truncated IS weights (for variance
  reduction, no outlier rejection).
- compute_rollout_rejection_mask(): only filter outliers (for sample cleaning, no IS weight
  computation).
- compute_offpolicy_metrics(): called by core functions to calculate off-policy diagnostics
  (KL/PPL/χ²) — no direct external calls needed.

### Integration Notes
- Used in `ray_trainer.py` via `compute_rollout_correction_and_add_to_batch()` (batch training pipeline).
- Used in `dp_actor.py` for distributed worker computations (distributed training scenarios).
- All functions support batch inputs and valid token masking (via `response_mask`).


## References
- "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch": https://richardli.xyz/rl-collapse
- Off-policy RL (theoretical basis for IS): https://fengyao.notion.site/off-policy-rl
"""

from typing import Any, Optional, Union

import torch

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.workers.config.actor import PolicyLossConfig

# Safety bound to prevent numerical overflow/underflow when exponentiating
# exp(20) ≈ 485 million (upper limit for stable weights), exp(-20) ≈ 2e-9 (lower limit)
SAFETY_BOUND = 20.0

# Clamp aggregate log-ratio before exp() when building wandb histograms.
# float32 exp overflows around 88.7; keep margin so Histogram / np.histogram stay finite.
_HISTOGRAM_LOG_EXP_CLAMP = 10.0

_OLD_PROB_MISMATCH_BUCKETS = (
    ("low", 0.0, 1.0 / 3, False),
    ("mid", 1.0 / 3, 2.0 / 3, False),
    ("high", 2.0 / 3, 1.0, True),
)

_BUCKET_STAT_QUANTILES = [5, 10, 15, 20, 30, 50, 70, 80, 85, 90, 95]

_LOCAL_RATIO_PENALTY_TYPES = (
    "seq",
    "seq_geo",
    "seq_geo_no_exp",
    "seq_geo_no_exp_sum",
    "seq_prob_diff",
    "seq_prob_diff_sum",
    "token_logp_diff",
    "token_prob_diff",
    "token_prob_ratio",
    "log_ppl_abs_diff",
    "seq_kl",
    "seq_jeffreys",
    "diff_seq_sum_add",
    "seq_forward_kl",
    "diff_seq_geo_abs",
    "seq_geo_inverse",
    "seq_k3_kl",
)


def _rollout_prob_weighted_metric_root(metric_root: str) -> str:
    return f"{metric_root}_mu_weighted"


def normalize_penalty_level(penalty_level: Optional[str]) -> str:
    """Normalize penalty level aliases to either ``sequence`` or ``token``."""
    if penalty_level is None:
        return "sequence"

    level = str(penalty_level).strip().lower()
    if level in {"sequence", "seq", "response", "response_level", "sequence_level"}:
        return "sequence"
    if level in {"token", "token_level"}:
        return "token"
    raise ValueError(f"penalty_level must be 'sequence' or 'token', got {penalty_level!r}")


def _apply_penalty_clip_band(
    raw_penalty: torch.Tensor,
    clip_penalty_low: Optional[float],
    clip_penalty_high: Optional[float],
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Zero penalty values inside the configured bilateral clip band."""
    if clip_penalty_low is None and clip_penalty_high is None:
        return raw_penalty

    effective_low = clip_penalty_low if clip_penalty_low is not None else float("-inf")
    effective_high = clip_penalty_high if clip_penalty_high is not None else float("inf")
    clip_mask = (raw_penalty > effective_low) & (raw_penalty < effective_high)
    if valid_mask is not None:
        clip_mask = clip_mask & valid_mask.to(dtype=torch.bool, device=raw_penalty.device)

    effective = raw_penalty.clone()
    effective[clip_mask] = 0
    return effective


def scale_token_penalty_by_response_length(
    penalty_per_token: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Scale token penalties so their sequence sum stays length-normalized.

    Without this scaling, subtracting a per-token penalty directly would make the total
    sequence penalty grow with response length. Dividing by the number of valid response
    tokens keeps the summed penalty comparable to a response-level mean penalty.
    """
    if penalty_per_token.ndim != 2:
        raise ValueError(f"penalty_per_token must be 2D, got shape {tuple(penalty_per_token.shape)}")
    if response_mask.shape != penalty_per_token.shape:
        raise ValueError(f"response_mask shape {tuple(response_mask.shape)} must match penalty_per_token shape {tuple(penalty_per_token.shape)}")

    mask = response_mask.to(dtype=penalty_per_token.dtype, device=penalty_per_token.device)
    valid_token_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return penalty_per_token * mask / valid_token_count


def scatter_sequence_values_to_last_valid_token(
    values_per_seq: torch.Tensor,
    response_mask: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Scatter one scalar per response onto the last valid response token."""
    if values_per_seq.ndim != 1:
        raise ValueError(f"values_per_seq must be 1D, got shape {tuple(values_per_seq.shape)}")
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be 2D, got shape {tuple(response_mask.shape)}")
    if response_mask.shape[0] != values_per_seq.shape[0]:
        raise ValueError(
            f"response_mask batch size {response_mask.shape[0]} must match values_per_seq size {values_per_seq.shape[0]}"
        )

    out = torch.zeros(
        response_mask.shape,
        device=response_mask.device,
        dtype=values_per_seq.dtype if dtype is None else dtype,
    )
    if values_per_seq.numel() == 0:
        return out

    valid_rows = response_mask.to(dtype=torch.bool).any(dim=-1)
    if not valid_rows.any():
        return out

    last_idx = response_mask.float().sum(dim=-1).long() - 1
    last_idx = last_idx.clamp(min=0)
    row_idx = torch.arange(response_mask.shape[0], device=response_mask.device)
    out[row_idx[valid_rows], last_idx[valid_rows]] = values_per_seq[valid_rows].to(out.dtype)
    return out


def compute_rollout_rejection_mask(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str = "token",
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rejection mask for outlier handling in off-policy RL training.

    This function identifies and masks outlier tokens/sequences using precomputed log ratios
    (log(π_train / π_rollout)). It supports multiple aggregation levels and uses log-space
    computations for numerical stability.

    Memory-efficient design:
    - Log-space calculations to avoid overflow
    - Fixed safety bounds on exponentiation
    - Metrics computed without large intermediate tensors

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Rejection sampling aggregation level, must be one of:
            - "token": Per-token outlier detection
            - "sequence": Aggregate across entire sequence (product of token ratios)
            - "geometric": Geometric mean across entire sequence
        rollout_rs_threshold: Upper threshold for valid IS weights (required for outlier detection).
        rollout_rs_threshold_lower: Lower threshold for valid IS weights. If None, defaults to 1/upper threshold.

    Returns:
        Tuple containing:
            modified_response_mask: Response mask with outliers masked (0=rejected),
                shape (batch_size, seq_length).
            metrics: Dictionary of rejection sampling metrics (all scalars), including:
                - rollout_rs_mean/max/min: Statistic of IS weights
                - rollout_rs_ratio_fraction_high/low: Fraction of weights exceeding thresholds
                - rollout_rs_masked_fraction: Fraction of tokens rejected (unified for all modes)
                - rollout_rs_seq_masked_fraction: Fraction of sequences rejected (mode-dependent)
    """
    # Validate input parameters
    valid_rs_levels = {"token", "sequence", "geometric"}
    if rollout_rs not in valid_rs_levels:
        raise ValueError(f"Invalid rollout_rs: {rollout_rs}. Must be one of {valid_rs_levels}.")
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be provided for rejection sampling.")

    # Set default lower threshold if not specified (reciprocal of upper threshold)
    upper_threshold = rollout_rs_threshold
    lower_threshold = rollout_rs_threshold_lower if rollout_rs_threshold_lower is not None else 1.0 / upper_threshold

    # Compute IS weights from log ratio (handles different aggregation levels)
    if rollout_rs == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_rs == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=0, max=20)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to (batch_size, seq_length)

    elif rollout_rs == "geometric":
        # Sequence-level geometric mean: exp(mean(log ratios))
        log_ratio_mean: torch.Tensor = verl_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_mean

        log_ratio_mean_safe: torch.Tensor = torch.clamp(log_ratio_mean, min=0.999, max=1.001)
        rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)

    else:
        raise ValueError(f"Unsupported rollout_rs: {rollout_rs}")

    # Generate outlier mask: 1=valid (within [lower, upper] threshold), 0=outlier
    mask: torch.Tensor = (rollout_is_weights >= lower_threshold) & (rollout_is_weights <= upper_threshold)
    mask = mask.float()

    # Compute rejection sampling metrics
    metrics: dict[str, float] = compute_rs_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=upper_threshold,
        rollout_rs_threshold_lower=lower_threshold,
    )

    # Track token-level and sequence-level rejection rates
    # rollout_rs_masked_fraction: fraction of tokens rejected (unified for all modes)
    metrics["rollout_rs_masked_fraction"] = verl_F.masked_mean(1 - mask, response_mask).item()

    # rollout_rs_seq_masked_fraction: fraction of sequences rejected (mode-dependent)
    if rollout_rs == "token":
        # Token-level aggregation: sequence is rejected if any token is rejected
        seq_has_masked: torch.Tensor = verl_F.masked_sum(1 - mask, response_mask, axis=-1) > 0
        metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
    else:
        # Sequence-level aggregation: check first token's mask (all tokens in sequence have same mask)
        metrics["rollout_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()

    # Apply rejection mask to original response mask
    modified_response_mask: torch.Tensor = response_mask * mask

    return modified_response_mask, metrics


def compute_rs_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str,
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
) -> dict[str, float]:
    """Compute comprehensive metrics for rejection sampling.

    This function calculates statistics for IS weights used in rejection sampling,
    balancing numerical stability (using clamped weights) and accuracy (using log-space
    for threshold checks).

    Args:
        rollout_is_weights: Clamped IS weights (π_train / π_rollout),
            shape (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio of training to rollout probabilities (unclamped),
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Rejection sampling aggregation level (matches compute_rollout_rejection_mask).
        rollout_rs_threshold: Upper threshold for valid IS weights.
        rollout_rs_threshold_lower: Lower threshold for valid IS weights.

    Returns:
        Dictionary of rejection sampling metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rollout_is_weights.device

    # Precompute log thresholds for accurate threshold checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    if rollout_rs in ["sequence", "geometric"]:
        # Sequence-level aggregation: use log-space for accurate max/min/threshold checks
        # True max/min (unclamped) converted with safety bounds
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_rs_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_rs_min"] = torch.exp(log_min).item()

        # Mean uses clamped weights to avoid overflow
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        # Both sequence and geometric modes operate at sequence level (batch_size, 1)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_rs_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from clamped weights
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_rs_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_rs_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_rs_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens first)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_rs_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_rs_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        # Clamp weights to threshold range to avoid squaring extreme values
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_rs_threshold_lower, max=rollout_rs_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        # Variance = E[X²] - (E[X])² (masked to valid tokens)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_rs_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_rs_std"] = 0.0

    # Compute Effective Sample Size (ESS) for IS weights
    # ESS = 1 / E[(w_i / E[w_i])²] (using clamped weights for stability)
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_rs_threshold_lower, max=rollout_rs_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_rs_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        # Mean weight per sequence (masked to valid tokens)
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_rs_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_rs_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_rs_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_rs_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_rs_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_rs_seq_fraction_high"] = (seq_mean_weights > rollout_rs_threshold).float().mean().item()
        metrics["rollout_rs_seq_fraction_low"] = (seq_mean_weights < rollout_rs_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute importance sampling weights to correct for off-policy distribution shifts.

    This function calculates IS weights (π_train / π_rollout) using log ratios for numerical stability.
    It supports multiple aggregation levels and truncates extreme weights to prevent training instability.

    Key design:
    - Log-space computations to avoid overflow
    - Truncation of extreme weights (TIS: Truncated Importance Sampling)
    - Optional batch normalization (normalize to mean=1.0)
    - Metrics tracking for weight distribution analysis

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level, must be one of:
            - "token": Per-token weights (biased, low variance)
            - "sequence": Per-sequence weight (product of tokens; unbiased, high variance)
        rollout_is_threshold: Upper threshold for truncating extreme weights (e.g., 2.0),
            default 2.0.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch,
            default False.

    Returns:
        Tuple containing:
            rollout_is_weights: Truncated IS weights (masked to zero for padding tokens),
                shape (batch_size, seq_length). If batch_normalize=True, normalized to mean=1.0.
            metrics: Dictionary of IS weight metrics (all scalars), including:
                - rollout_is_mean/max/min: Statistic of weights (before batch normalization)
                - rollout_is_eff_sample_size: Effective sample size (ESS)
                - rollout_is_seq_*: Sequence-level weight statistics
                - rollout_is_batch_norm_factor: Normalization factor (only if batch_normalize=True)
    """
    # Validate input parameters
    valid_is_levels = {"token", "sequence"}
    if rollout_is not in valid_is_levels:
        raise ValueError(f"Invalid rollout_is: {rollout_is}. Must be one of {valid_is_levels}.")
    if rollout_is_threshold <= 0:
        raise ValueError(f"rollout_is_threshold must be positive, got {rollout_is_threshold}.")

    # Compute IS weights from log ratio (handles different aggregation levels)
    if rollout_is == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_is == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to sequence length

    else:
        raise ValueError(f"Unsupported rollout_is: {rollout_is}")

    # Zero out weights for padding tokens using response mask
    rollout_is_weights = rollout_is_weights * response_mask

    # Compute IS weight metrics (BEFORE truncation to get accurate fraction_high/low)
    metrics: dict[str, float] = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
    )

    # Truncate extreme weights (TIS: Truncated Importance Sampling)
    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)

    # Detach weights to prevent gradient flow (mathematically required by IS theory)
    # IS weights change the measure, not the objective. See §3.2.2 in docs/algo/rollout_corr_math.md
    rollout_is_weights = rollout_is_weights.detach()

    # Apply batch normalization if requested
    if rollout_is_batch_normalize:
        # Compute mean based on aggregation level
        mask_float = response_mask.to(dtype=rollout_is_weights.dtype)
        if rollout_is == "token":
            # Token-level: normalize over all token weights
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(rollout_is_weights, mask_float)
            else:
                weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask)
        elif rollout_is == "sequence":
            # Sequence-level: normalize over sequence weights (one weight per sequence)
            # For each sequence, compute mean over valid tokens (they all have the same weight)
            # then average across sequences
            seq_weights = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)  # (batch_size,)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=rollout_is_weights.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(seq_weights, seq_mask)
            else:
                weights_mean = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported rollout_is: {rollout_is}")

        # Normalize to mean=1.0 (avoid division by zero)
        if weights_mean > 1e-8:
            rollout_is_weights = rollout_is_weights / weights_mean
            metrics["rollout_is_batch_norm_factor"] = weights_mean.item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return rollout_is_weights, metrics


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
    """Compute comprehensive metrics for truncated importance sampling weights.

    This function calculates statistics for truncated IS weights (TIS), using log-space
    for accurate threshold checks and clamped weights for stable mean/std calculations.

    Args:
        rollout_is_weights: Truncated IS weights (π_train / π_rollout),
            shape (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio of training to rollout probabilities (unclamped),
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level (matches compute_rollout_correction_weights).
        rollout_is_threshold: Upper threshold for truncated IS weights.

    Returns:
        Dictionary of IS weight metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rollout_is_weights.device
    # Default lower threshold (reciprocal of upper threshold)
    rollout_is_threshold_lower: float = 1.0 / rollout_is_threshold

    # Precompute log thresholds for accurate checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    if rollout_is == "sequence":
        # Sequence-level aggregation: use log-space for unclamped stats
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_is_min"] = torch.exp(log_min).item()

        # Mean uses truncated weights to avoid overflow
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_is_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from truncated weights
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_is_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_is_threshold_lower, max=rollout_is_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_is_std"] = 0.0

    # Compute Effective Sample Size (ESS) for truncated weights
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_is_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_is_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean().item()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: Optional[float] = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = 2.0,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Optional[DataProto], torch.Tensor, dict[str, float]]:
    """Unified interface for computing IS weights and rejection masks.

    This function combines IS weight calculation (truncated) and rejection sampling (masked)
    into a single pipeline. It also applies a per-token veto for catastrophic outliers
    (sequences with extremely low token ratios are fully rejected).

    Key design:
    - Separation of IS weights (for variance reduction) and rejection masks (for sample filtering)
    - Veto mechanism for catastrophic sequences (applied independently of other modes)
    - Comprehensive metrics tracking for mismatch diagnosis

    Args:
        old_log_prob: Log probabilities from the training policy (e.g., FSDP FP32),
            shape (batch_size, seq_length).
        rollout_log_prob: Log probabilities from the rollout policy (e.g., vLLM BF16),
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level (see compute_rollout_correction_weights for options).
            Set to None to disable IS weight computation.
        rollout_is_threshold: Upper threshold for truncated IS weights (used if rollout_is is set),
            default 2.0.
        rollout_rs: Rejection sampling aggregation level (see compute_rollout_rejection_mask for options).
            Set to None to disable rejection sampling.
        rollout_rs_threshold: Upper threshold for rejection sampling. Required if rollout_rs is enabled.
            Default 2.0.
        rollout_rs_threshold_lower: Lower threshold for rejection sampling (used if rollout_rs is set).
            Defaults to 1/rollout_rs_threshold if None.
        rollout_token_veto_threshold: Minimum allowed token-level IS weight. Sequences containing
            any token below this threshold are fully rejected. Set to None to disable veto.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch.
            Default: False.

    Returns:
        Tuple containing:
            rollout_is_weights_proto: DataProto with IS weights (None if rollout_is is None),
                key "rollout_is_weights", shape (batch_size, seq_length).
            modified_response_mask: Response mask with rejection sampling and veto applied,
                shape (batch_size, seq_length).
            metrics: Dictionary of all metrics (prefixed with "rollout_corr/"), including:
                - IS weight statistics
                - Rejection sampling rates
                - Veto statistics
                - Policy mismatch metrics (KL, PPL, etc.)
    """
    # Validate input masks
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")
    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            f"old_log_prob shape {old_log_prob.shape} does not match rollout_log_prob shape {rollout_log_prob.shape}."
        )
    if old_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"log_prob shape {old_log_prob.shape} does not match response_mask shape {response_mask.shape}."
        )

    # Step 1: Compute log ratio (log(π_train / π_rollout))
    log_ratio: torch.Tensor = old_log_prob - rollout_log_prob
    device: torch.device = log_ratio.device
    metrics: dict[str, float] = {}

    # Step 2: Compute IS weights (if enabled)
    rollout_is_weights: Optional[torch.Tensor] = None
    if rollout_is is not None and rollout_is_threshold is not None:
        rollout_is_weights, is_metrics = compute_rollout_correction_weights(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
        )
        metrics.update(is_metrics)

    # Step 3: Compute rejection mask (if enabled)
    modified_response_mask: torch.Tensor = response_mask.clone()
    if rollout_rs is not None:
        if rollout_rs_threshold is None:
            raise ValueError(
                "rollout_rs_threshold must be explicitly provided when rollout_rs is enabled. "
                "Set rollout_rs_threshold to the desired threshold value."
            )
        modified_response_mask, rs_metrics = compute_rollout_rejection_mask(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
            rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        )
        metrics.update(rs_metrics)

    # Step 4: Apply per-token veto (reject sequences with catastrophic tokens)
    if rollout_token_veto_threshold is not None:
        if rollout_token_veto_threshold <= 0:
            raise ValueError(f"rollout_token_veto_threshold must be positive, got {rollout_token_veto_threshold}.")

        # Compute log threshold for numerical stability
        log_veto_threshold: torch.Tensor = torch.log(torch.tensor(rollout_token_veto_threshold, device=device))
        # Identify catastrophic tokens (log ratio below threshold + valid mask)
        catastrophic_tokens: torch.Tensor = (log_ratio < log_veto_threshold) & response_mask.bool()
        # Check if sequence contains any catastrophic token
        has_catastrophic: torch.Tensor = catastrophic_tokens.any(dim=-1, keepdim=True)
        # Create veto mask (0=reject sequence, 1=keep)
        veto_mask: torch.Tensor = (~has_catastrophic).float()

        # Track veto metrics
        metrics["rollout_is_veto_fraction"] = has_catastrophic.float().mean().item()
        metrics["rollout_is_catastrophic_token_fraction"] = verl_F.masked_mean(
            catastrophic_tokens.float(), response_mask
        ).item()

        # Apply veto to response mask (overrides previous rejection)
        modified_response_mask = modified_response_mask * veto_mask
    else:
        # Add placeholder metrics if veto is disabled
        metrics["rollout_is_veto_fraction"] = 0.0
        metrics["rollout_is_catastrophic_token_fraction"] = 0.0

    # Step 5: Compute off-policy metrics (KL, PPL, χ², etc.) and ratio stats (token/seq/seq_geo)
    offpolicy_metrics: dict[str, float] = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    metrics.update(offpolicy_metrics)
    ratio_stats = compute_ratio_stats(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    metrics.update(ratio_stats)

    # Step 6: Add "rollout_corr/" prefix to all metrics for logging consistency
    metrics_scalar: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_scalar[f"rollout_corr/{key}"] = value

    # Step 7: Wrap IS weights in DataProto for consistency with API
    rollout_is_weights_proto: Optional[DataProto] = None
    if rollout_is_weights is not None:
        rollout_is_weights_proto = DataProto.from_dict(tensors={"rollout_is_weights": rollout_is_weights})

    return rollout_is_weights_proto, modified_response_mask, metrics_scalar


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute off-policy diagnostic metrics (helper function).

    This helper function operates on raw tensors and is used internally by:
    - compute_rollout_correction_and_rejection_mask() in this module (automatically included)
    - Tests (test_rollout_corr.py, test_rollout_corr_integration.py)

    These metrics help diagnose the off-policy gap between rollout and training policies,
    which can arise from:
    - Policy mismatch (e.g., vLLM BF16 vs FSDP FP32)
    - Model staleness (training on trajectories from older checkpoints)
    - General distribution shifts

    Key metrics:
    - kl: Direct KL divergence estimator KL(π_rollout || π_training)
    - k3_kl: K3 KL estimator for stability (more stable for small KL)
    - training_ppl: Perplexity of training policy
    - rollout_ppl: Perplexity of rollout policy
    - log_ppl_diff: Difference in log perplexities
    - ppl_ratio: Ratio of training PPL to rollout PPL
    - chi2_token: Token-level χ² divergence E[ρ²] - 1
    - chi2_seq: Sequence-level χ² divergence E[(∏ρ_t)²] - 1

    Args:
        old_log_prob: Log probabilities from training policy, shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy, shape (batch_size, seq_length)
        response_mask: Mask for valid tokens, shape (batch_size, seq_length)

    Returns:
        Dictionary of off-policy metrics (without prefix)
    """
    # Validate that we have at least one valid token
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics = {}

    # 1. Training policy perplexity (always available)
    # Formula: exp(-1/|T| * Σ log π_training(y_t|y_<t))
    # where |T| is the number of tokens generated by the model
    mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
    metrics["training_ppl"] = training_ppl.detach().item()

    # Also log log-ppl for easier analysis (avoids exponential scale)
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    # 2. Compute rollout off-policy metrics (only if rollout_log_probs available)
    if rollout_log_prob is not None:
        # 2a. kl: Direct estimator for KL(π_rollout || π_training)
        # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
        # Positive value means rollout policy is more confident than training policy
        log_prob_diff = rollout_log_prob - old_log_prob
        metrics["kl"] = verl_F.masked_mean(log_prob_diff, response_mask).detach().item()
        _flat_kl = log_prob_diff[response_mask.bool()]
        metrics["kl_p90"] = (
            torch.quantile(_flat_kl.float(), 0.90).detach().item()
            if _flat_kl.numel() >= 2
            else (_flat_kl.mean().detach().item() if _flat_kl.numel() > 0 else 0.0)
        )

        # 2b. k3_kl: K3 estimator for KL(π_rollout || π_training)
        # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
        # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
        log_ratio = old_log_prob - rollout_log_prob
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = verl_F.masked_mean(k3_kl_matrix, response_mask).detach().item()
        _flat_k3 = k3_kl_matrix[response_mask.bool()]
        metrics["k3_kl_p90"] = (
            torch.quantile(_flat_k3.float(), 0.90).detach().item()
            if _flat_k3.numel() >= 2
            else (_flat_k3.mean().detach().item() if _flat_k3.numel() > 0 else 0.0)
        )

        # jeffreys_kl: Jeffreys' divergence: J = KL(π_rollout||π_training) + KL(π_training||π_rollout)
        # rev_kl需要重要性采样，因为不存在pi_train的采样
        importance_weights = torch.exp(log_ratio)
        forward_log_prob_diff = importance_weights * log_ratio
        metrics["jeffreys_kl"] = verl_F.masked_mean(forward_log_prob_diff, response_mask).detach().item() + verl_F.masked_mean(log_prob_diff, response_mask).detach().item()
        metrics["forward_kl"] = verl_F.masked_mean(forward_log_prob_diff, response_mask).detach().item()
        _flat_forward_log_prob_diff = forward_log_prob_diff[response_mask.bool()]
        metrics["forward_kl_p90"] = (
            torch.quantile(_flat_forward_log_prob_diff.float(), 0.90).detach().item()
            if _flat_forward_log_prob_diff.numel() >= 2
            else (_flat_forward_log_prob_diff.mean().detach().item() if _flat_forward_log_prob_diff.numel() > 0 else 0.0)
        )
        jeffreys_token = forward_log_prob_diff + log_prob_diff
        _flat_jeffreys = jeffreys_token[response_mask.bool()]
        metrics["jeffreys_kl_p90"] = (
            torch.quantile(_flat_jeffreys.float(), 0.90).detach().item()
            if _flat_jeffreys.numel() >= 2
            else (_flat_jeffreys.mean().detach().item() if _flat_jeffreys.numel() > 0 else 0.0)
        )

        # Per-token matrices (masked so invalid positions are 0 for seq-level sum)
        seq_log_prob_diff = verl_F.masked_sum(log_prob_diff, response_mask, axis=-1)  # (B,) 对应 -r
        seq_log_ratio = verl_F.masked_sum(log_ratio, response_mask, axis=-1)  # (B,) r
        seq_kl = seq_log_prob_diff
        seq_k3 = torch.exp(seq_log_ratio) - seq_log_ratio - 1
        importance_weights_seq =torch.exp(seq_log_ratio)
        seq_jeffreys = importance_weights_seq * seq_log_ratio + seq_log_prob_diff
        seq_forward_kl = importance_weights_seq * seq_log_ratio 
        metrics["seq_forward_kl"] = seq_forward_kl.mean().detach().item()
        metrics["seq_forward_kl_p90"] = (
            torch.quantile(seq_forward_kl.float(), 0.90).detach().item()
            if seq_forward_kl.numel() >= 2
            else seq_forward_kl.mean().detach().item()
        )

        # 3. Standard Seq-level Metrics (按序列求和)
        # 物理意义：整句话的累计散度。可用于判断整条轨迹是否偏离过大
        metrics["seq_kl"] = seq_kl.mean().detach().item()
        metrics["seq_kl_p90"] = (
            torch.quantile(seq_kl.float(), 0.90).detach().item()
            if seq_kl.numel() >= 2
            else seq_kl.mean().detach().item()
        )
        metrics["seq_k3_kl"] = seq_k3.mean().detach().item()
        metrics["seq_k3_kl_p90"] = (
            torch.quantile(seq_k3.float(), 0.90).detach().item()
            if seq_k3.numel() >= 2
            else seq_k3.mean().detach().item()
        )
        metrics["seq_jeffreys_kl"] = seq_jeffreys.mean().detach().item()
        metrics["seq_jeffreys_kl_p90"] = (
            torch.quantile(seq_jeffreys.float(), 0.90).detach().item()
            if seq_jeffreys.numel() >= 2
            else seq_jeffreys.mean().detach().item()
        )

        # 2c. Rollout policy perplexity
        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL
        metrics["rollout_ppl"] = rollout_ppl.detach().item()
        metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

        # 2d. Log PPL difference (sequence-level perplexity difference)
        # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        # Since ppl = exp(-log_prob), we have:
        #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
        # Positive value means training assigns lower probability (higher PPL) than rollout
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_abs_diff_p90"] = (
            torch.quantile(log_ppl_diff.abs().float(), 0.90).detach().item()
            if log_ppl_diff.numel() >= 2
            else log_ppl_diff.abs().mean().detach().item()
        )
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()

        # 2e. PPL ratio (how much higher is training PPL vs rollout PPL)
        # IMPORTANT: Compute per-sequence ratio first, then average
        # For numerical stability, compute in log space using log_ppl_diff
        # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
        # This is the inverse of geometric IS: ppl_ratio_i = 1 / geometric_is_i for each sequence
        ppl_ratio = torch.exp(log_ppl_diff).mean()  # mean(exp(log_ppl_diff)) = mean(ppl_ratio_i)
        metrics["ppl_ratio"] = ppl_ratio.detach().item()

        # 2f. Chi-squared divergence: χ²(π_training || π_rollout) = E_μ[ρ²] - 1
        # where ρ = π_training / π_rollout and μ = π_rollout (rollout distribution)
        # This measures the variance of importance sampling weights
        # Token-level: E_token[ρ²] - 1 (averaged over all tokens)
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)  # ρ = π_training / π_rollout (token-level)
        rho_squared_token = rho_token.square()
        chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0
        metrics["chi2_token"] = chi2_token.detach().item()

        # Sequence-level: E_seq[(Π ρ_t)²] - 1 = E_seq[exp(2 * Σ log ρ_t)] - 1
        log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)  # Σ log ρ_t per sequence
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)  # (Π ρ_t)²
        chi2_seq = rho_squared_seq.mean() - 1.0
        metrics["chi2_seq"] = chi2_seq.detach().item()

    return metrics


def compute_ratio_stats(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute ratio (π_train / π_rollout) and seq-level train-vs-rollout diff statistics.

    Used for monitoring off-policy gap.

    Ratio (连乘 in prob space):
    - Token-level: ρ_t per token, then stats over all valid tokens (flatten).
    - Sequence-level (standard): ρ_seq = exp(sum_t log_ratio_t) per sequence, stats over (B,).
    - Sequence-level (geometric): ρ_geo = exp(mean_t log_ratio_t) per sequence, stats over (B,).

    Seq-level π_train, π_rollout and diff (in prob space, after exp):
    - pi_train_seq / pi_rollout_seq: exp(sum_t log π) (连乘).
    - diff_seq_sum: pi_train_seq - pi_rollout_seq (连乘后的差).
    - diff_seq_sum_add: sum_t (π_train_t - π_rollout_t) per sequence (先 token-level 概率差再对序列求和，连加).
    - diff_seq_geo: exp(mean_t log π_train) - exp(mean_t log π_rollout) (几何后的差).

    Args:
        old_log_prob: (batch_size, seq_length)
        rollout_log_prob: (batch_size, seq_length)
        response_mask: (batch_size, seq_length), 1=valid

    Returns:
        Dict with keys like ratio_token_mean, ratio_seq_mean, ratio_seq_geo_median,
        diff_seq_sum_mean, diff_seq_geo_mean, etc.
    """
    if rollout_log_prob is None or not response_mask.any():
        return {}
    # diff in log space: (log π_train - log π_rollout) per token
    log_ratio = old_log_prob - rollout_log_prob
    metrics: dict[str, float] = {}

    def _stats(tensor: torch.Tensor, prefix: str) -> None:
        is_inverse = "inverse" in prefix
        if tensor.numel() == 0:
            base_stats = ["mean", "max", "min", "std", "median", "p90", "p95", "p98"]
            if is_inverse:
                base_stats += ["p10", "p80", "p50"]
            for s in base_stats:
                metrics[f"{prefix}_{s}"] = 0.0
            return
        metrics[f"{prefix}_mean"] = tensor.mean().item()
        metrics[f"{prefix}_max"] = tensor.max().item()
        metrics[f"{prefix}_min"] = tensor.min().item()
        metrics[f"{prefix}_std"] = tensor.std().item() if tensor.numel() > 1 else 0.0
        metrics[f"{prefix}_median"] = tensor.median().item()
        if tensor.numel() >= 2:
            q = torch.tensor([0.90, 0.95, 0.98], device=tensor.device, dtype=tensor.dtype)
            p90, p95, p98 = torch.quantile(tensor.float(), q).tolist()
        else:
            p90 = p95 = p98 = tensor.item()
        metrics[f"{prefix}_p90"] = p90
        metrics[f"{prefix}_p95"] = p95
        metrics[f"{prefix}_p98"] = p98
        if is_inverse:
            if tensor.numel() >= 2:
                q_inv = torch.tensor([0.10, 0.80, 0.50], device=tensor.device, dtype=tensor.dtype)
                p10, p80, p50 = torch.quantile(tensor.float(), q_inv).tolist()
            else:
                p10 = p80 = p50 = tensor.item()
            metrics[f"{prefix}_p10"] = p10
            metrics[f"{prefix}_p80"] = p80
            # Keep consistent with naming request: 5分位 => p50
            metrics[f"{prefix}_p50"] = p50

    # Token-level: flatten valid tokens
    rho_token = torch.exp(log_ratio)
    flat = rho_token[response_mask.bool()]
    _stats(flat, "ratio_token")

    # Sequence-level standard: ρ_seq[b] = exp(sum_t log_ratio[b,t])
    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    rho_seq = torch.exp(log_ratio_sum)
    _stats(rho_seq, "ratio_seq")

    # Sequence-level geometric: ρ_geo[b] = exp(mean_t log_ratio[b,t])
    log_ratio_mean = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
    rho_seq_geo = torch.exp(log_ratio_mean)
    _stats(rho_seq_geo, "ratio_seq_geo")

    # Sequence-level geometric: ρ_geo[b] = exp(mean_t log_ratio[b,t])
    log_ratio_mean = verl_F.masked_mean(-log_ratio, response_mask, axis=-1)
    rho_seq_geo = torch.exp(log_ratio_mean)
    _stats(rho_seq_geo, "ratio_seq_geo_inverse")

    # Seq-level π_train, π_rollout (exp of sum/mean log prob) and diff in prob space
    log_prob_train_sum = verl_F.masked_sum(old_log_prob, response_mask, axis=-1)
    log_prob_rollout_sum = verl_F.masked_sum(rollout_log_prob, response_mask, axis=-1)
    pi_train_seq = torch.exp(log_prob_train_sum)   # seq-level π_train (连乘)
    pi_rollout_seq = torch.exp(log_prob_rollout_sum)  # seq-level π_rollout (连乘)
    diff_seq_sum = pi_train_seq - pi_rollout_seq
    _stats(diff_seq_sum, "diff_seq_sum")
    _stats(diff_seq_sum.abs(), "diff_seq_sum_abs")
    # 连加的 seq-level diff: 先 token-level 概率差 π_train_t - π_rollout_t，再对序列求和
    pi_train_token = torch.exp(old_log_prob)
    pi_rollout_token = torch.exp(rollout_log_prob)
    diff_token = pi_train_token - pi_rollout_token  # (B, T)
    diff_seq_sum_add = verl_F.masked_sum(diff_token, response_mask, axis=-1)
    diff_seq_sum_mean = verl_F.masked_mean(diff_token, response_mask, axis=-1)
    _stats(diff_seq_sum_add, "diff_seq_sum_add")
    _stats(diff_seq_sum_mean, "diff_seq_sum_mean")
    # abs 再 sum/mean，避免正负相抵
    diff_seq_sum_add_abs = verl_F.masked_sum(diff_token.abs(), response_mask, axis=-1)
    diff_seq_sum_mean_abs = verl_F.masked_mean(diff_token.abs(), response_mask, axis=-1)
    _stats(diff_seq_sum_add_abs, "diff_seq_sum_add_abs")
    _stats(diff_seq_sum_mean_abs, "diff_seq_sum_mean_abs")

    log_prob_train_mean = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
    log_prob_rollout_mean = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
    pi_train_seq_geo = torch.exp(log_prob_train_mean)   # exp(mean_t log π_train)
    pi_rollout_seq_geo = torch.exp(log_prob_rollout_mean)
    diff_seq_geo = pi_train_seq_geo - pi_rollout_seq_geo  # diff in prob space (几何)
    _stats(diff_seq_geo, "diff_seq_geo")
    # abs 再做统计（含 mean），避免正负相抵
    _stats(diff_seq_geo.abs(), "diff_seq_geo_abs")
    diff_seq_geo_mean_abs = diff_seq_geo.abs().mean().unsqueeze(0)  # batch 上先 abs 再 mean
    _stats(diff_seq_geo_mean_abs, "diff_seq_geo_mean_abs")
    return metrics


def compute_ratio_histograms(batch: DataProto) -> dict[str, Any]:
    """Compute wandb.Histogram objects for sequence-level IS ratio distributions.

    These are logged as wandb Histograms, which appear as interactive distribution
    plots in the wandb UI. The step slider lets you see how the distribution evolves
    over training.

    Log aggregates are clamped before ``exp`` and non-finite values are removed so
    ``numpy.histogram`` (inside wandb) never sees ``nan``/``inf`` (which can raise
    ``ValueError`` or ``IndexError``).

    Returns:
        Dict with wandb.Histogram values (compatible with wandb.log).
        Non-wandb loggers will silently ignore these non-scalar entries.
    """
    import wandb

    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return {}
    response_mask = batch.batch["response_mask"]
    if not response_mask.any():
        return {}
    log_ratio = batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"]

    def _finite_rho_list(log_agg: torch.Tensor) -> list[float]:
        # clamp does not clear nan; exp(nan)=nan — filter after exp
        safe = torch.clamp(
            log_agg,
            min=-_HISTOGRAM_LOG_EXP_CLAMP,
            max=_HISTOGRAM_LOG_EXP_CLAMP,
        )
        rho = torch.exp(safe).detach().float().cpu().reshape(-1)
        rho = rho[torch.isfinite(rho)]
        return rho.tolist()

    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    rho_seq = _finite_rho_list(log_ratio_sum)

    log_ratio_mean = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
    rho_seq_geo = _finite_rho_list(log_ratio_mean)

    out: dict[str, Any] = {}
    if rho_seq:
        out["rollout_corr/ratio_seq_dist"] = wandb.Histogram(rho_seq)
    if rho_seq_geo:
        out["rollout_corr/ratio_seq_geo_dist"] = wandb.Histogram(rho_seq_geo)
    return out
1

def compute_ratio_by_length_bucket_metrics(
    batch: DataProto, num_buckets: int = 4
) -> dict[str, float]:
    """按 response 长度分桶，统计各桶内 seq-level / seq-geo ratio 的 mean、max、min，用于观察 ratio 与长度的相关性。

    分桶方式：以当前 batch 的 max_response_length（response_mask 的 seq 维）为 L，等分为 num_buckets 档：
    bucket 0: [0, L/4), bucket 1: [L/4, L/2), bucket 2: [L/2, 3L/4), bucket 3: [3L/4, L]。
    每桶内对 ratio_seq / ratio_seq_geo 分别算 mean、max、min；空桶填 nan。
    返回的 metrics 键为 rollout_corr/ratio_seq_by_len_bucket{b}_mean/max/min 与 ratio_seq_geo_by_len_bucket{b}_mean/max/min。
    """
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return {}
    response_mask = batch.batch["response_mask"].bool()
    if not response_mask.any():
        return {}
    max_len = response_mask.shape[-1]
    response_length = response_mask.sum(-1).float()  # (B,)
    log_ratio = batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"]
    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    log_ratio_mean = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
    ratio_seq = torch.exp(log_ratio_sum)
    ratio_seq_geo = torch.exp(log_ratio_mean)

    metrics: dict[str, float] = {}
    bucket_size = max_len / num_buckets
    for b in range(num_buckets):
        low = b * bucket_size
        high = (b + 1) * bucket_size if b < num_buckets - 1 else max_len + 1
        mask = (response_length >= low) & (response_length < high)
        if mask.sum() == 0:
            metrics[f"rollout_corr/ratio_seq_by_len_bucket{b}_mean"] = float("nan")
            metrics[f"rollout_corr/ratio_seq_by_len_bucket{b}_max"] = float("nan")
            metrics[f"rollout_corr/ratio_seq_by_len_bucket{b}_min"] = float("nan")
            metrics[f"rollout_corr/ratio_seq_geo_by_len_bucket{b}_mean"] = float("nan")
            metrics[f"rollout_corr/ratio_seq_geo_by_len_bucket{b}_max"] = float("nan")
            metrics[f"rollout_corr/ratio_seq_geo_by_len_bucket{b}_min"] = float("nan")
        else:
            rs = ratio_seq[mask]
            rg = ratio_seq_geo[mask]
            metrics[f"rollout_corr/ratio_seq_by_len_bucket{b}_mean"] = rs.mean().item()
            metrics[f"rollout_corr/ratio_seq_by_len_bucket{b}_max"] = rs.max().item()
            metrics[f"rollout_corr/ratio_seq_by_len_bucket{b}_min"] = rs.min().item()
            metrics[f"rollout_corr/ratio_seq_geo_by_len_bucket{b}_mean"] = rg.mean().item()
            metrics[f"rollout_corr/ratio_seq_geo_by_len_bucket{b}_max"] = rg.max().item()
            metrics[f"rollout_corr/ratio_seq_geo_by_len_bucket{b}_min"] = rg.min().item()
    return metrics


def compute_mismatch_vs_logits_metrics(batch: DataProto) -> dict[str, float]:
    """高 mismatch token 与「该 token 的绝对 logit」的关联：只用 pre-softmax 的 chosen_logits（未除 temperature）。
    """
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return {}
    if "chosen_logits" not in batch.batch:
        return {}
    response_mask = batch.batch["response_mask"].bool()
    if not response_mask.any():
        return {}
    rollout_log_prob = batch.batch["rollout_log_probs"]
    log_ratio = batch.batch["old_log_probs"] - rollout_log_prob
    chosen_logits = batch.batch["chosen_logits"]

    flat_mask = response_mask.flatten()
    lr_flat = log_ratio.flatten()[flat_mask]
    cl_flat = chosen_logits.flatten()[flat_mask]
    if lr_flat.numel() < 2:
        return {}

    median_lr = lr_flat.median()
    high = lr_flat > median_lr
    low = ~high

    metrics: dict[str, float] = {}
    metrics["rollout_corr/mismatch_high_chosen_logit_mean"] = cl_flat[high].mean().item()
    metrics["rollout_corr/mismatch_high_chosen_logit_max"] = cl_flat[high].max().item()
    metrics["rollout_corr/mismatch_high_chosen_logit_min"] = cl_flat[high].min().item()
    metrics["rollout_corr/mismatch_low_chosen_logit_mean"] = cl_flat[low].mean().item()
    metrics["rollout_corr/mismatch_low_chosen_logit_max"] = cl_flat[low].max().item()
    metrics["rollout_corr/mismatch_low_chosen_logit_min"] = cl_flat[low].min().item()
    lr_c = lr_flat - lr_flat.mean()
    cl_c = cl_flat - cl_flat.mean()
    den_lr = (lr_c.pow(2).sum().clamp_min(1e-12)).sqrt()
    den_cl = (cl_c.pow(2).sum().clamp_min(1e-12)).sqrt()
    metrics["rollout_corr/corr_log_ratio_chosen_logit"] = (
        (lr_c * cl_c).sum() / (den_lr * den_cl + 1e-12)
    ).item()
    return metrics


def _add_bucket_distribution_stats(
    metrics: dict[str, float],
    prefix: str,
    bucket_values: torch.Tensor,
    total_tokens: int,
    stat_name_suffix: str = "",
    include_count_stats: bool = True,
    emit_quantiles: bool = False,
    emit_std: bool = True,
    emit_min_max: bool = True,
    filter_non_finite: bool = False,
    empty_stat_value: float = float("nan"),
) -> None:
    if filter_non_finite:
        bucket_values = bucket_values[torch.isfinite(bucket_values)]

    bucket_count = bucket_values.numel()
    if include_count_stats:
        metrics[f"{prefix}_token_frac{stat_name_suffix}"] = bucket_count / total_tokens
        metrics[f"{prefix}_token_cnt{stat_name_suffix}"] = bucket_count

    if bucket_count == 0:
        metrics[f"{prefix}_mean{stat_name_suffix}"] = empty_stat_value
        if emit_min_max:
            metrics[f"{prefix}_max{stat_name_suffix}"] = empty_stat_value
            metrics[f"{prefix}_min{stat_name_suffix}"] = empty_stat_value
        if emit_std:
            metrics[f"{prefix}_std{stat_name_suffix}"] = empty_stat_value
        if emit_quantiles:
            for q in _BUCKET_STAT_QUANTILES:
                metrics[f"{prefix}_p{q}{stat_name_suffix}"] = empty_stat_value
        return

    metrics[f"{prefix}_mean{stat_name_suffix}"] = bucket_values.mean().item()
    if emit_min_max:
        metrics[f"{prefix}_max{stat_name_suffix}"] = bucket_values.max().item()
        metrics[f"{prefix}_min{stat_name_suffix}"] = bucket_values.min().item()
    if emit_std:
        metrics[f"{prefix}_std{stat_name_suffix}"] = bucket_values.std().item() if bucket_count >= 2 else 0.0
    if not emit_quantiles:
        return

    if bucket_count >= 2:
        q_tensor = torch.tensor([q / 100.0 for q in _BUCKET_STAT_QUANTILES], device=bucket_values.device, dtype=torch.float32)
        q_values = torch.quantile(bucket_values.float(), q_tensor).tolist()
        for q, value in zip(_BUCKET_STAT_QUANTILES, q_values):
            metrics[f"{prefix}_p{q}{stat_name_suffix}"] = value
    else:
        scalar = bucket_values.item()
        for q in _BUCKET_STAT_QUANTILES:
            metrics[f"{prefix}_p{q}{stat_name_suffix}"] = scalar


def _compute_bucketed_signed_and_abs_metrics(
    *,
    metric_root: str,
    old_prob_flat: torch.Tensor,
    signed_value_flat: torch.Tensor,
    abs_value_flat: torch.Tensor,
    len_norm_signed_value_flat: Optional[torch.Tensor] = None,
    len_norm_abs_value_flat: Optional[torch.Tensor] = None,
    advantage_flat: Optional[torch.Tensor] = None,
    include_count_stats: bool = True,
    emit_quantiles: bool = False,
    filter_non_finite: bool = False,
    empty_stat_value: float = float("nan"),
) -> dict[str, float]:
    total_tokens = signed_value_flat.numel()
    if total_tokens == 0:
        return {}

    metrics: dict[str, float] = {}

    if advantage_flat is None:
        for bucket_name, low, high, include_high in _OLD_PROB_MISMATCH_BUCKETS:
            if include_high:
                bucket_mask = (old_prob_flat >= low) & (old_prob_flat <= high)
            else:
                bucket_mask = (old_prob_flat >= low) & (old_prob_flat < high)

            _add_bucket_distribution_stats(
                metrics=metrics,
                prefix=f"{metric_root}/{bucket_name}/raw",
                bucket_values=signed_value_flat[bucket_mask],
                total_tokens=total_tokens,
                include_count_stats=include_count_stats,
                emit_quantiles=emit_quantiles,
                emit_std=False,
                filter_non_finite=filter_non_finite,
                empty_stat_value=empty_stat_value,
            )
            _add_bucket_distribution_stats(
                metrics=metrics,
                prefix=f"{metric_root}/{bucket_name}/abs",
                bucket_values=abs_value_flat[bucket_mask],
                total_tokens=total_tokens,
                include_count_stats=False,
                emit_quantiles=emit_quantiles,
                emit_std=False,
                filter_non_finite=filter_non_finite,
                empty_stat_value=empty_stat_value,
            )
            if len_norm_signed_value_flat is not None and len_norm_abs_value_flat is not None:
                _add_bucket_distribution_stats(
                    metrics=metrics,
                    prefix=f"{metric_root}/{bucket_name}/raw",
                    bucket_values=len_norm_signed_value_flat[bucket_mask],
                    total_tokens=total_tokens,
                    stat_name_suffix="_len_norm",
                    include_count_stats=False,
                    emit_quantiles=emit_quantiles,
                    emit_std=False,
                    emit_min_max=False,
                    filter_non_finite=filter_non_finite,
                    empty_stat_value=empty_stat_value,
                )
                _add_bucket_distribution_stats(
                    metrics=metrics,
                    prefix=f"{metric_root}/{bucket_name}/abs",
                    bucket_values=len_norm_abs_value_flat[bucket_mask],
                    total_tokens=total_tokens,
                    stat_name_suffix="_len_norm",
                    include_count_stats=False,
                    emit_quantiles=emit_quantiles,
                    emit_std=False,
                    emit_min_max=False,
                    filter_non_finite=filter_non_finite,
                    empty_stat_value=empty_stat_value,
                )
        return metrics

    for sign_name, sign_mask in (
        ("pos", advantage_flat > 0),
        ("neg", advantage_flat < 0),
    ):
        _add_bucket_distribution_stats(
            metrics=metrics,
            prefix=f"{metric_root}/{sign_name}/raw",
            bucket_values=signed_value_flat[sign_mask],
            total_tokens=total_tokens,
            include_count_stats=include_count_stats,
            emit_quantiles=emit_quantiles,
            emit_std=False,
            filter_non_finite=filter_non_finite,
            empty_stat_value=empty_stat_value,
        )
        _add_bucket_distribution_stats(
            metrics=metrics,
            prefix=f"{metric_root}/{sign_name}/abs",
            bucket_values=abs_value_flat[sign_mask],
            total_tokens=total_tokens,
            include_count_stats=False,
            emit_quantiles=emit_quantiles,
            emit_std=False,
            filter_non_finite=filter_non_finite,
            empty_stat_value=empty_stat_value,
        )
        if len_norm_signed_value_flat is not None and len_norm_abs_value_flat is not None:
            _add_bucket_distribution_stats(
                metrics=metrics,
                prefix=f"{metric_root}/{sign_name}/raw",
                bucket_values=len_norm_signed_value_flat[sign_mask],
                total_tokens=total_tokens,
                stat_name_suffix="_len_norm",
                include_count_stats=False,
                emit_quantiles=emit_quantiles,
                emit_std=False,
                emit_min_max=False,
                filter_non_finite=filter_non_finite,
                empty_stat_value=empty_stat_value,
            )
            _add_bucket_distribution_stats(
                metrics=metrics,
                prefix=f"{metric_root}/{sign_name}/abs",
                bucket_values=len_norm_abs_value_flat[sign_mask],
                total_tokens=total_tokens,
                stat_name_suffix="_len_norm",
                include_count_stats=False,
                emit_quantiles=emit_quantiles,
                emit_std=False,
                emit_min_max=False,
                filter_non_finite=filter_non_finite,
                empty_stat_value=empty_stat_value,
            )

    return metrics


def _drop_abs_extrema_metrics(metrics: dict[str, float]) -> dict[str, float]:
    def _should_drop(key: str) -> bool:
        leaf = key.rsplit("/", 1)[-1]
        return leaf.startswith("abs_") and ("_max" in leaf or "_min" in leaf)

    return {key: value for key, value in metrics.items() if not _should_drop(key)}


def _compute_overall_signed_and_abs_metrics(
    *,
    metric_root: str,
    signed_value_flat: torch.Tensor,
    abs_value_flat: torch.Tensor,
    len_norm_signed_value_flat: Optional[torch.Tensor] = None,
    len_norm_abs_value_flat: Optional[torch.Tensor] = None,
    empty_stat_value: float = float("nan"),
) -> dict[str, float]:
    total_tokens = signed_value_flat.numel()
    if total_tokens == 0:
        return {}

    metrics: dict[str, float] = {}
    _add_bucket_distribution_stats(
        metrics=metrics,
        prefix=f"{metric_root}/all/raw",
        bucket_values=signed_value_flat,
        total_tokens=total_tokens,
        include_count_stats=False,
        emit_quantiles=False,
        empty_stat_value=empty_stat_value,
    )
    _add_bucket_distribution_stats(
        metrics=metrics,
        prefix=f"{metric_root}/all/abs",
        bucket_values=abs_value_flat,
        total_tokens=total_tokens,
        include_count_stats=False,
        emit_quantiles=False,
        empty_stat_value=empty_stat_value,
    )

    metrics[f"{metric_root}/all/raw_std"] = signed_value_flat.std().item() if total_tokens >= 2 else 0.0
    metrics[f"{metric_root}/all/abs_std"] = abs_value_flat.std().item() if total_tokens >= 2 else 0.0

    # `pos_mean` / `neg_mean` are conditional means over strictly positive / negative
    # mismatches only. They are useful for diagnosing sign-specific severity, but they
    # do not share the same denominator as `abs_mean`, which is averaged over all valid
    # tokens (including zero-mismatch tokens). Expose the signed-part means with the
    # global denominator as well so dashboards can decompose `abs_mean` directly.
    pos_values = signed_value_flat[signed_value_flat > 0]
    neg_values = signed_value_flat[signed_value_flat < 0]
    pos_count = pos_values.numel()
    neg_count = neg_values.numel()
    zero_count = total_tokens - pos_count - neg_count
    metrics[f"{metric_root}/all/pos_mean"] = pos_values.mean().item() if pos_values.numel() > 0 else 0.0
    metrics[f"{metric_root}/all/pos_token_mean"] = signed_value_flat.clamp(min=0).mean().item()
    metrics[f"{metric_root}/all/neg_mean"] = neg_values.mean().item() if neg_values.numel() > 0 else 0.0
    metrics[f"{metric_root}/all/neg_token_mean"] = signed_value_flat.clamp(max=0).mean().item()
    metrics[f"{metric_root}/all/pos_frac"] = pos_count / total_tokens
    metrics[f"{metric_root}/all/neg_frac"] = neg_count / total_tokens
    metrics[f"{metric_root}/all/zero_frac"] = zero_count / total_tokens

    if len_norm_signed_value_flat is not None and len_norm_abs_value_flat is not None:
        _add_bucket_distribution_stats(
            metrics=metrics,
            prefix=f"{metric_root}/all/raw",
            bucket_values=len_norm_signed_value_flat,
            total_tokens=total_tokens,
            stat_name_suffix="_len_norm",
            include_count_stats=False,
            emit_quantiles=False,
            emit_std=False,
            emit_min_max=False,
            empty_stat_value=empty_stat_value,
        )
        _add_bucket_distribution_stats(
            metrics=metrics,
            prefix=f"{metric_root}/all/abs",
            bucket_values=len_norm_abs_value_flat,
            total_tokens=total_tokens,
            stat_name_suffix="_len_norm",
            include_count_stats=False,
            emit_quantiles=False,
            emit_std=False,
            emit_min_max=False,
            empty_stat_value=empty_stat_value,
        )
        pos_values_len_norm = len_norm_signed_value_flat[len_norm_signed_value_flat > 0]
        neg_values_len_norm = len_norm_signed_value_flat[len_norm_signed_value_flat < 0]
        metrics[f"{metric_root}/all/pos_mean_len_norm"] = pos_values_len_norm.mean().item() if pos_values_len_norm.numel() > 0 else 0.0
        metrics[f"{metric_root}/all/neg_mean_len_norm"] = neg_values_len_norm.mean().item() if neg_values_len_norm.numel() > 0 else 0.0
    return metrics


def _compute_mismatch_metrics(
    *,
    metric_root: str,
    old_prob_flat: torch.Tensor,
    signed_value_flat: torch.Tensor,
    abs_value_flat: torch.Tensor,
    len_norm_signed_value_flat: Optional[torch.Tensor] = None,
    len_norm_abs_value_flat: Optional[torch.Tensor] = None,
    advantage_flat: Optional[torch.Tensor] = None,
    empty_stat_value: float = float("nan"),
) -> dict[str, float]:
    metrics = _compute_bucketed_signed_and_abs_metrics(
        metric_root=metric_root,
        old_prob_flat=old_prob_flat,
        signed_value_flat=signed_value_flat,
        abs_value_flat=abs_value_flat,
        len_norm_signed_value_flat=len_norm_signed_value_flat,
        len_norm_abs_value_flat=len_norm_abs_value_flat,
        advantage_flat=advantage_flat,
        include_count_stats=False,
        emit_quantiles=False,
        empty_stat_value=empty_stat_value,
    )
    metrics.update(
        _compute_overall_signed_and_abs_metrics(
            metric_root=metric_root,
            signed_value_flat=signed_value_flat,
            abs_value_flat=abs_value_flat,
            len_norm_signed_value_flat=len_norm_signed_value_flat,
            len_norm_abs_value_flat=len_norm_abs_value_flat,
            empty_stat_value=empty_stat_value,
        )
    )
    return _drop_abs_extrema_metrics(metrics)


def _compute_bucketed_value_metrics(
    *,
    metric_root: str,
    old_prob_flat: torch.Tensor,
    value_flat: torch.Tensor,
    len_norm_centered_value_flat: Optional[torch.Tensor] = None,
    len_norm_abs_value_flat: Optional[torch.Tensor] = None,
    advantage_flat: Optional[torch.Tensor] = None,
    center_value: float = 0.0,
    overall_sign_source_flat: Optional[torch.Tensor] = None,
    empty_stat_value: float = float("nan"),
) -> dict[str, float]:
    total_values = value_flat.numel()
    if total_values == 0:
        return {}

    centered_value_flat = value_flat - center_value
    abs_value_flat = centered_value_flat.abs()
    metrics: dict[str, float] = {}

    def _add_metrics(
        prefix: str,
        bucket_values: torch.Tensor,
        bucket_abs_values: torch.Tensor,
        bucket_len_norm_values: Optional[torch.Tensor] = None,
        bucket_len_norm_abs_values: Optional[torch.Tensor] = None,
    ) -> None:
        _add_bucket_distribution_stats(
            metrics=metrics,
            prefix=f"{prefix}/raw",
            bucket_values=bucket_values,
            total_tokens=total_values,
            include_count_stats=False,
            emit_quantiles=False,
            emit_std=False,
            filter_non_finite=True,
            empty_stat_value=empty_stat_value,
        )
        if bucket_len_norm_values is not None and bucket_len_norm_abs_values is not None:
            _add_bucket_distribution_stats(
                metrics=metrics,
                prefix=f"{prefix}/raw",
                bucket_values=bucket_len_norm_values,
                total_tokens=total_values,
                stat_name_suffix="_len_norm",
                include_count_stats=False,
                emit_quantiles=False,
                emit_std=False,
                emit_min_max=False,
                filter_non_finite=True,
                empty_stat_value=empty_stat_value,
            )
            _add_bucket_distribution_stats(
                metrics=metrics,
                prefix=f"{prefix}/abs",
                bucket_values=bucket_len_norm_abs_values,
                total_tokens=total_values,
                stat_name_suffix="_len_norm",
                include_count_stats=False,
                emit_quantiles=False,
                emit_std=False,
                emit_min_max=False,
                filter_non_finite=True,
                empty_stat_value=empty_stat_value,
            )
        _add_bucket_distribution_stats(
            metrics=metrics,
            prefix=f"{prefix}/abs",
            bucket_values=bucket_abs_values,
            total_tokens=total_values,
            include_count_stats=False,
            emit_quantiles=False,
            emit_std=False,
            filter_non_finite=True,
            empty_stat_value=empty_stat_value,
        )

    if advantage_flat is None:
        for bucket_name, low, high, include_high in _OLD_PROB_MISMATCH_BUCKETS:
            if include_high:
                bucket_mask = (old_prob_flat >= low) & (old_prob_flat <= high)
            else:
                bucket_mask = (old_prob_flat >= low) & (old_prob_flat < high)
            _add_metrics(
                f"{metric_root}/{bucket_name}",
                centered_value_flat[bucket_mask],
                abs_value_flat[bucket_mask],
                len_norm_centered_value_flat[bucket_mask] if len_norm_centered_value_flat is not None else None,
                len_norm_abs_value_flat[bucket_mask] if len_norm_abs_value_flat is not None else None,
            )
    else:
        for sign_name, sign_mask in (("pos", advantage_flat > 0), ("neg", advantage_flat < 0)):
            _add_metrics(
                f"{metric_root}/{sign_name}",
                centered_value_flat[sign_mask],
                abs_value_flat[sign_mask],
                len_norm_centered_value_flat[sign_mask] if len_norm_centered_value_flat is not None else None,
                len_norm_abs_value_flat[sign_mask] if len_norm_abs_value_flat is not None else None,
            )

    _add_metrics(
        f"{metric_root}/all",
        centered_value_flat,
        abs_value_flat,
        len_norm_centered_value_flat,
        len_norm_abs_value_flat,
    )

    metrics[f"{metric_root}/all/raw_std"] = centered_value_flat.std().item() if total_values >= 2 else 0.0
    metrics[f"{metric_root}/all/abs_std"] = abs_value_flat.std().item() if total_values >= 2 else 0.0

    sign_source_flat = overall_sign_source_flat if overall_sign_source_flat is not None else centered_value_flat
    pos_mask = sign_source_flat > 0
    neg_mask = sign_source_flat < 0
    pos_values = centered_value_flat[pos_mask]
    neg_values = centered_value_flat[neg_mask]
    pos_count = pos_values.numel()
    neg_count = neg_values.numel()
    zero_count = total_values - pos_count - neg_count
    metrics[f"{metric_root}/all/pos_mean"] = pos_values.mean().item() if pos_count > 0 else 0.0
    metrics[f"{metric_root}/all/pos_token_mean"] = pos_values.sum().item() / total_values if pos_count > 0 else 0.0
    metrics[f"{metric_root}/all/neg_mean"] = neg_values.mean().item() if neg_count > 0 else 0.0
    metrics[f"{metric_root}/all/neg_token_mean"] = neg_values.sum().item() / total_values if neg_count > 0 else 0.0
    metrics[f"{metric_root}/all/pos_frac"] = pos_count / total_values
    metrics[f"{metric_root}/all/neg_frac"] = neg_count / total_values
    metrics[f"{metric_root}/all/zero_frac"] = zero_count / total_values
    if len_norm_centered_value_flat is not None and len_norm_abs_value_flat is not None:
        pos_values_len_norm = len_norm_centered_value_flat[pos_mask]
        neg_values_len_norm = len_norm_centered_value_flat[neg_mask]
        metrics[f"{metric_root}/all/pos_mean_len_norm"] = (
            pos_values_len_norm.mean().item() if pos_values_len_norm.numel() > 0 else 0.0
        )
        metrics[f"{metric_root}/all/neg_mean_len_norm"] = (
            neg_values_len_norm.mean().item() if neg_values_len_norm.numel() > 0 else 0.0
        )
    return _drop_abs_extrema_metrics(metrics)


def _compute_token_frac_cnt_metrics(
    *,
    old_prob_flat: torch.Tensor,
    advantage_flat: Optional[torch.Tensor] = None,
    metric_root: str = "token_frac_cnt",
    frac_name: str = "token_frac",
    cnt_name: str = "token_cnt",
) -> dict[str, float]:
    total_tokens = old_prob_flat.numel()
    if total_tokens == 0:
        return {}

    metrics: dict[str, float] = {}

    def _add_count_metrics(prefix: str, mask: torch.Tensor) -> None:
        bucket_count = int(mask.sum().item())
        metrics[f"{prefix}/{frac_name}"] = bucket_count / total_tokens
        metrics[f"{prefix}/{cnt_name}"] = bucket_count

    if advantage_flat is None:
        for bucket_name, low, high, include_high in _OLD_PROB_MISMATCH_BUCKETS:
            if include_high:
                bucket_mask = (old_prob_flat >= low) & (old_prob_flat <= high)
            else:
                bucket_mask = (old_prob_flat >= low) & (old_prob_flat < high)
            _add_count_metrics(f"{metric_root}/{bucket_name}", bucket_mask)
    else:
        for sign_name, sign_mask in (("pos", advantage_flat > 0), ("neg", advantage_flat < 0)):
            _add_count_metrics(f"{metric_root}/{sign_name}", sign_mask)

    full_mask = torch.ones_like(old_prob_flat, dtype=torch.bool)
    _add_count_metrics(f"{metric_root}/all", full_mask)
    return metrics


def _extract_mismatch_flat_tensors(batch: DataProto) -> Optional[dict[str, torch.Tensor]]:
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return None

    response_mask = batch.batch["response_mask"].bool()
    if not response_mask.any():
        return None

    old_log_prob = batch.batch["old_log_probs"].float()
    rollout_log_prob = batch.batch["rollout_log_probs"].float()
    old_prob = torch.exp(old_log_prob).clamp(min=0.0, max=1.0)
    rollout_prob = torch.exp(rollout_log_prob).clamp(min=0.0, max=1.0)
    response_length = response_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(old_log_prob.dtype)

    mismatch_log_prob = old_log_prob - rollout_log_prob
    mismatch_prob = old_prob - rollout_prob
    prob_ratio = torch.exp(mismatch_log_prob)
    mismatch_log_prob_rollout_prob_weighted = mismatch_log_prob * rollout_prob
    mismatch_prob_rollout_prob_weighted = mismatch_prob * rollout_prob
    mismatch_log_prob_sq_rollout_prob_weighted = mismatch_log_prob.square() * rollout_prob
    mismatch_prob_sq_rollout_prob_weighted = mismatch_prob.square() * rollout_prob
    mismatch_prob_sq = mismatch_prob.square()
    prob_ratio_rollout_prob_weighted = prob_ratio * rollout_prob
    prob_ratio_centered_len_norm = (prob_ratio - 1.0) / response_length
    mismatch_log_prob_len_norm = mismatch_log_prob / response_length
    mismatch_prob_len_norm = mismatch_prob / response_length
    mismatch_log_prob_rollout_prob_weighted_len_norm = mismatch_log_prob_rollout_prob_weighted / response_length
    mismatch_prob_rollout_prob_weighted_len_norm = mismatch_prob_rollout_prob_weighted / response_length
    mismatch_log_prob_sq_rollout_prob_weighted_len_norm = mismatch_log_prob_sq_rollout_prob_weighted / response_length
    mismatch_prob_sq_rollout_prob_weighted_len_norm = mismatch_prob_sq_rollout_prob_weighted / response_length
    mismatch_prob_sq_len_norm = mismatch_prob_sq / response_length
    prob_ratio_rollout_prob_weighted_len_norm = prob_ratio_rollout_prob_weighted / response_length

    flat_mask = response_mask.flatten()
    tensors = {
        "old_prob_flat": old_prob.flatten()[flat_mask],
        "mismatch_log_prob_flat": mismatch_log_prob.flatten()[flat_mask],
        "mismatch_log_prob_abs_flat": mismatch_log_prob.abs().flatten()[flat_mask],
        "mismatch_log_prob_len_norm_flat": mismatch_log_prob_len_norm.flatten()[flat_mask],
        "mismatch_log_prob_len_norm_abs_flat": mismatch_log_prob_len_norm.abs().flatten()[flat_mask],
        "mismatch_prob_flat": mismatch_prob.flatten()[flat_mask],
        "mismatch_prob_abs_flat": mismatch_prob.abs().flatten()[flat_mask],
        "mismatch_prob_len_norm_flat": mismatch_prob_len_norm.flatten()[flat_mask],
        "mismatch_prob_len_norm_abs_flat": mismatch_prob_len_norm.abs().flatten()[flat_mask],
        "mismatch_log_prob_rollout_prob_weighted_flat": mismatch_log_prob_rollout_prob_weighted.flatten()[flat_mask],
        "mismatch_log_prob_rollout_prob_weighted_abs_flat": mismatch_log_prob_rollout_prob_weighted.abs().flatten()[flat_mask],
        "mismatch_log_prob_rollout_prob_weighted_len_norm_flat": mismatch_log_prob_rollout_prob_weighted_len_norm.flatten()[flat_mask],
        "mismatch_log_prob_rollout_prob_weighted_len_norm_abs_flat": mismatch_log_prob_rollout_prob_weighted_len_norm.abs().flatten()[flat_mask],
        "mismatch_prob_rollout_prob_weighted_flat": mismatch_prob_rollout_prob_weighted.flatten()[flat_mask],
        "mismatch_prob_rollout_prob_weighted_abs_flat": mismatch_prob_rollout_prob_weighted.abs().flatten()[flat_mask],
        "mismatch_prob_rollout_prob_weighted_len_norm_flat": mismatch_prob_rollout_prob_weighted_len_norm.flatten()[flat_mask],
        "mismatch_prob_rollout_prob_weighted_len_norm_abs_flat": mismatch_prob_rollout_prob_weighted_len_norm.abs().flatten()[flat_mask],
        "mismatch_log_prob_sq_rollout_prob_weighted_flat": mismatch_log_prob_sq_rollout_prob_weighted.flatten()[flat_mask],
        "mismatch_log_prob_sq_rollout_prob_weighted_len_norm_flat": mismatch_log_prob_sq_rollout_prob_weighted_len_norm.flatten()[flat_mask],
        "mismatch_log_prob_sq_rollout_prob_weighted_len_norm_abs_flat": mismatch_log_prob_sq_rollout_prob_weighted_len_norm.abs().flatten()[flat_mask],
        "mismatch_prob_sq_rollout_prob_weighted_flat": mismatch_prob_sq_rollout_prob_weighted.flatten()[flat_mask],
        "mismatch_prob_sq_rollout_prob_weighted_len_norm_flat": mismatch_prob_sq_rollout_prob_weighted_len_norm.flatten()[flat_mask],
        "mismatch_prob_sq_rollout_prob_weighted_len_norm_abs_flat": mismatch_prob_sq_rollout_prob_weighted_len_norm.abs().flatten()[flat_mask],
        "mismatch_prob_sq_flat": mismatch_prob_sq.flatten()[flat_mask],
        "mismatch_prob_sq_len_norm_flat": mismatch_prob_sq_len_norm.flatten()[flat_mask],
        "mismatch_prob_sq_len_norm_abs_flat": mismatch_prob_sq_len_norm.abs().flatten()[flat_mask],
        "prob_ratio_flat": prob_ratio.flatten()[flat_mask],
        "prob_ratio_len_norm_flat": prob_ratio_centered_len_norm.flatten()[flat_mask],
        "prob_ratio_len_norm_abs_flat": prob_ratio_centered_len_norm.abs().flatten()[flat_mask],
        "prob_ratio_rollout_prob_weighted_flat": prob_ratio_rollout_prob_weighted.flatten()[flat_mask],
        "prob_ratio_rollout_prob_weighted_len_norm_flat": prob_ratio_rollout_prob_weighted_len_norm.flatten()[flat_mask],
        "prob_ratio_rollout_prob_weighted_len_norm_abs_flat": prob_ratio_rollout_prob_weighted_len_norm.abs().flatten()[flat_mask],
    }
    response_correct_sign = torch.where(
        batch.batch["token_level_scores"].float().sum(dim=-1, keepdim=True) > 0,
        torch.ones_like(response_length),
        -torch.ones_like(response_length),
    )
    tensors["advantage_flat"] = response_correct_sign.expand_as(response_mask).flatten()[flat_mask]
    return tensors


def _extract_mismatch_sequence_tensors(batch: DataProto) -> Optional[dict[str, torch.Tensor]]:
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return None

    response_mask = batch.batch["response_mask"].bool()
    if not response_mask.any():
        return None

    old_log_prob = batch.batch["old_log_probs"].float()
    rollout_log_prob = batch.batch["rollout_log_probs"].float()
    response_length = response_mask.sum(dim=-1).clamp(min=1).to(old_log_prob.dtype)

    old_prob_token = torch.exp(old_log_prob).clamp(min=0.0, max=1.0)
    rollout_prob_token = torch.exp(rollout_log_prob).clamp(min=0.0, max=1.0)
    token_prob_ratio = torch.exp(old_log_prob - rollout_log_prob)
    token_logp_diff_rollout_prob_weighted = (old_log_prob - rollout_log_prob) * rollout_prob_token
    token_p_diff_rollout_prob_weighted = (old_prob_token - rollout_prob_token) * rollout_prob_token
    token_logp_diff_sq_rollout_prob_weighted = (old_log_prob - rollout_log_prob).square() * rollout_prob_token
    token_p_diff_sq_rollout_prob_weighted = (old_prob_token - rollout_prob_token).square() * rollout_prob_token
    token_p_diff_sq = (old_prob_token - rollout_prob_token).square()
    token_prob_ratio_rollout_prob_weighted = token_prob_ratio * rollout_prob_token
    # Bucket sequence samples by the geometric mean token probability so the value
    # stays within [0, 1] and matches the existing low/mid/high token buckets.
    old_prob_seq = torch.exp(verl_F.masked_mean(old_log_prob, response_mask, axis=-1))

    seq_logp_diff = verl_F.masked_mean(old_log_prob - rollout_log_prob, response_mask, axis=-1)
    seq_p_diff = verl_F.masked_mean(old_prob_token - rollout_prob_token, response_mask, axis=-1)
    seq_sum_logp_diff = verl_F.masked_sum(old_log_prob - rollout_log_prob, response_mask, axis=-1)
    seq_sum_p_diff = verl_F.masked_sum(old_prob_token - rollout_prob_token, response_mask, axis=-1)
    seq_logp_diff_rollout_prob_weighted = verl_F.masked_mean(token_logp_diff_rollout_prob_weighted, response_mask, axis=-1)
    seq_p_diff_rollout_prob_weighted = verl_F.masked_mean(token_p_diff_rollout_prob_weighted, response_mask, axis=-1)
    seq_logp_diff_sq_rollout_prob_weighted = verl_F.masked_mean(token_logp_diff_sq_rollout_prob_weighted, response_mask, axis=-1)
    seq_p_diff_sq_rollout_prob_weighted = verl_F.masked_mean(token_p_diff_sq_rollout_prob_weighted, response_mask, axis=-1)
    seq_p_diff_sq = verl_F.masked_mean(token_p_diff_sq, response_mask, axis=-1)
    seq_sum_logp_diff_rollout_prob_weighted = verl_F.masked_sum(token_logp_diff_rollout_prob_weighted, response_mask, axis=-1)
    seq_sum_p_diff_rollout_prob_weighted = verl_F.masked_sum(token_p_diff_rollout_prob_weighted, response_mask, axis=-1)
    seq_sum_logp_diff_sq_rollout_prob_weighted = verl_F.masked_sum(token_logp_diff_sq_rollout_prob_weighted, response_mask, axis=-1)
    seq_sum_p_diff_sq_rollout_prob_weighted = verl_F.masked_sum(token_p_diff_sq_rollout_prob_weighted, response_mask, axis=-1)
    seq_sum_p_diff_sq = verl_F.masked_sum(token_p_diff_sq, response_mask, axis=-1)
    seq_prob_ratio = verl_F.masked_mean(token_prob_ratio, response_mask, axis=-1)
    seq_sum_prob_ratio = verl_F.masked_sum(token_prob_ratio, response_mask, axis=-1)
    seq_prob_ratio_rollout_prob_weighted = verl_F.masked_mean(token_prob_ratio_rollout_prob_weighted, response_mask, axis=-1)
    seq_sum_prob_ratio_rollout_prob_weighted = verl_F.masked_sum(token_prob_ratio_rollout_prob_weighted, response_mask, axis=-1)
    seq_sum_prob_ratio_centered = seq_sum_prob_ratio - response_length
    seq_prob_ratio_len_norm = (seq_prob_ratio - 1.0) / response_length
    seq_sum_prob_ratio_len_norm = seq_sum_prob_ratio_centered / response_length
    seq_prob_ratio_rollout_prob_weighted_len_norm = seq_prob_ratio_rollout_prob_weighted / response_length
    seq_sum_prob_ratio_rollout_prob_weighted_len_norm = seq_sum_prob_ratio_rollout_prob_weighted / response_length

    tensors = {
        "old_prob_seq": old_prob_seq,
        "seq_logp_diff": seq_logp_diff,
        "seq_logp_diff_abs": seq_logp_diff.abs(),
        "seq_logp_diff_len_norm": seq_logp_diff / response_length,
        "seq_logp_diff_len_norm_abs": (seq_logp_diff / response_length).abs(),
        "seq_logp_diff_rollout_prob_weighted": seq_logp_diff_rollout_prob_weighted,
        "seq_logp_diff_rollout_prob_weighted_abs": seq_logp_diff_rollout_prob_weighted.abs(),
        "seq_logp_diff_rollout_prob_weighted_len_norm": seq_logp_diff_rollout_prob_weighted / response_length,
        "seq_logp_diff_rollout_prob_weighted_len_norm_abs": (seq_logp_diff_rollout_prob_weighted / response_length).abs(),
        "seq_logp_diff_sq_rollout_prob_weighted": seq_logp_diff_sq_rollout_prob_weighted,
        "seq_logp_diff_sq_rollout_prob_weighted_len_norm": seq_logp_diff_sq_rollout_prob_weighted / response_length,
        "seq_logp_diff_sq_rollout_prob_weighted_len_norm_abs": (seq_logp_diff_sq_rollout_prob_weighted / response_length).abs(),
        "seq_p_diff": seq_p_diff,
        "seq_p_diff_abs": seq_p_diff.abs(),
        "seq_p_diff_len_norm": seq_p_diff / response_length,
        "seq_p_diff_len_norm_abs": (seq_p_diff / response_length).abs(),
        "seq_p_diff_rollout_prob_weighted": seq_p_diff_rollout_prob_weighted,
        "seq_p_diff_rollout_prob_weighted_abs": seq_p_diff_rollout_prob_weighted.abs(),
        "seq_p_diff_rollout_prob_weighted_len_norm": seq_p_diff_rollout_prob_weighted / response_length,
        "seq_p_diff_rollout_prob_weighted_len_norm_abs": (seq_p_diff_rollout_prob_weighted / response_length).abs(),
        "seq_p_diff_sq_rollout_prob_weighted": seq_p_diff_sq_rollout_prob_weighted,
        "seq_p_diff_sq_rollout_prob_weighted_len_norm": seq_p_diff_sq_rollout_prob_weighted / response_length,
        "seq_p_diff_sq_rollout_prob_weighted_len_norm_abs": (seq_p_diff_sq_rollout_prob_weighted / response_length).abs(),
        "seq_p_diff_sq": seq_p_diff_sq,
        "seq_p_diff_sq_len_norm": seq_p_diff_sq / response_length,
        "seq_p_diff_sq_len_norm_abs": (seq_p_diff_sq / response_length).abs(),
        "seq_sum_logp_diff": seq_sum_logp_diff,
        "seq_sum_logp_diff_abs": seq_sum_logp_diff.abs(),
        "seq_sum_logp_diff_len_norm": seq_sum_logp_diff / response_length,
        "seq_sum_logp_diff_len_norm_abs": (seq_sum_logp_diff / response_length).abs(),
        "seq_sum_logp_diff_rollout_prob_weighted": seq_sum_logp_diff_rollout_prob_weighted,
        "seq_sum_logp_diff_rollout_prob_weighted_abs": seq_sum_logp_diff_rollout_prob_weighted.abs(),
        "seq_sum_logp_diff_rollout_prob_weighted_len_norm": seq_sum_logp_diff_rollout_prob_weighted / response_length,
        "seq_sum_logp_diff_rollout_prob_weighted_len_norm_abs": (seq_sum_logp_diff_rollout_prob_weighted / response_length).abs(),
        "seq_sum_logp_diff_sq_rollout_prob_weighted": seq_sum_logp_diff_sq_rollout_prob_weighted,
        "seq_sum_logp_diff_sq_rollout_prob_weighted_len_norm": seq_sum_logp_diff_sq_rollout_prob_weighted / response_length,
        "seq_sum_logp_diff_sq_rollout_prob_weighted_len_norm_abs": (seq_sum_logp_diff_sq_rollout_prob_weighted / response_length).abs(),
        "seq_sum_p_diff": seq_sum_p_diff,
        "seq_sum_p_diff_abs": seq_sum_p_diff.abs(),
        "seq_sum_p_diff_len_norm": seq_sum_p_diff / response_length,
        "seq_sum_p_diff_len_norm_abs": (seq_sum_p_diff / response_length).abs(),
        "seq_sum_p_diff_rollout_prob_weighted": seq_sum_p_diff_rollout_prob_weighted,
        "seq_sum_p_diff_rollout_prob_weighted_abs": seq_sum_p_diff_rollout_prob_weighted.abs(),
        "seq_sum_p_diff_rollout_prob_weighted_len_norm": seq_sum_p_diff_rollout_prob_weighted / response_length,
        "seq_sum_p_diff_rollout_prob_weighted_len_norm_abs": (seq_sum_p_diff_rollout_prob_weighted / response_length).abs(),
        "seq_sum_p_diff_sq_rollout_prob_weighted": seq_sum_p_diff_sq_rollout_prob_weighted,
        "seq_sum_p_diff_sq_rollout_prob_weighted_len_norm": seq_sum_p_diff_sq_rollout_prob_weighted / response_length,
        "seq_sum_p_diff_sq_rollout_prob_weighted_len_norm_abs": (seq_sum_p_diff_sq_rollout_prob_weighted / response_length).abs(),
        "seq_sum_p_diff_sq": seq_sum_p_diff_sq,
        "seq_sum_p_diff_sq_len_norm": seq_sum_p_diff_sq / response_length,
        "seq_sum_p_diff_sq_len_norm_abs": (seq_sum_p_diff_sq / response_length).abs(),
        "seq_prob_ratio": seq_prob_ratio,
        "seq_prob_ratio_len_norm": seq_prob_ratio_len_norm,
        "seq_prob_ratio_len_norm_abs": seq_prob_ratio_len_norm.abs(),
        "seq_prob_ratio_rollout_prob_weighted": seq_prob_ratio_rollout_prob_weighted,
        "seq_prob_ratio_rollout_prob_weighted_len_norm": seq_prob_ratio_rollout_prob_weighted_len_norm,
        "seq_prob_ratio_rollout_prob_weighted_len_norm_abs": seq_prob_ratio_rollout_prob_weighted_len_norm.abs(),
        "seq_sum_prob_ratio": seq_sum_prob_ratio_centered,
        "seq_sum_prob_ratio_len_norm": seq_sum_prob_ratio_len_norm,
        "seq_sum_prob_ratio_len_norm_abs": seq_sum_prob_ratio_len_norm.abs(),
        "seq_sum_prob_ratio_rollout_prob_weighted": seq_sum_prob_ratio_rollout_prob_weighted,
        "seq_sum_prob_ratio_rollout_prob_weighted_len_norm": seq_sum_prob_ratio_rollout_prob_weighted_len_norm,
        "seq_sum_prob_ratio_rollout_prob_weighted_len_norm_abs": seq_sum_prob_ratio_rollout_prob_weighted_len_norm.abs(),
    }
    tensors["advantage_seq"] = torch.where(
        batch.batch["token_level_scores"].float().sum(dim=-1) > 0,
        torch.ones_like(response_length),
        -torch.ones_like(response_length),
    )
    return tensors


def compute_mismatch_log_prob_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 old token probability 分桶统计 log-prob mismatch。

    这里的 old token probability 定义为 `exp(old_log_probs)`，bucket 为：
    - low:  [0, 1/3)
    - medium: [1/3, 2/3)
    - high: [2/3, 1]

    同时保留两套 token 级统计：
    - signed mismatch: `old_log_probs - rollout_log_probs`
    - abs mismatch: `|old_log_probs - rollout_log_probs|`
    """
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="token_logp_diff",
        old_prob_flat=tensors["old_prob_flat"],
        signed_value_flat=tensors["mismatch_log_prob_flat"],
        abs_value_flat=tensors["mismatch_log_prob_abs_flat"],
        len_norm_signed_value_flat=tensors["mismatch_log_prob_len_norm_flat"],
        len_norm_abs_value_flat=tensors["mismatch_log_prob_len_norm_abs_flat"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("token_logp_diff"),
            old_prob_flat=tensors["old_prob_flat"],
            signed_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_flat"],
            abs_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_abs_flat"],
            len_norm_signed_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_len_norm_abs_flat"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="token_logp_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_flat"],
            value_flat=tensors["mismatch_log_prob_sq_rollout_prob_weighted_flat"],
            len_norm_centered_value_flat=tensors["mismatch_log_prob_sq_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_log_prob_sq_rollout_prob_weighted_len_norm_abs_flat"],
            overall_sign_source_flat=tensors["mismatch_log_prob_flat"],
        )
    )
    return metrics


def compute_mismatch_prob_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 old token probability 分桶统计 prob mismatch。

    这里的 prob mismatch 定义为 `exp(old_log_probs) - exp(rollout_log_probs)`。
    """
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="token_p_diff",
        old_prob_flat=tensors["old_prob_flat"],
        signed_value_flat=tensors["mismatch_prob_flat"],
        abs_value_flat=tensors["mismatch_prob_abs_flat"],
        len_norm_signed_value_flat=tensors["mismatch_prob_len_norm_flat"],
        len_norm_abs_value_flat=tensors["mismatch_prob_len_norm_abs_flat"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("token_p_diff"),
            old_prob_flat=tensors["old_prob_flat"],
            signed_value_flat=tensors["mismatch_prob_rollout_prob_weighted_flat"],
            abs_value_flat=tensors["mismatch_prob_rollout_prob_weighted_abs_flat"],
            len_norm_signed_value_flat=tensors["mismatch_prob_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_prob_rollout_prob_weighted_len_norm_abs_flat"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="token_p_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_flat"],
            value_flat=tensors["mismatch_prob_sq_rollout_prob_weighted_flat"],
            len_norm_centered_value_flat=tensors["mismatch_prob_sq_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_prob_sq_rollout_prob_weighted_len_norm_abs_flat"],
            overall_sign_source_flat=tensors["mismatch_prob_flat"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="token_p_diff_sq",
            old_prob_flat=tensors["old_prob_flat"],
            value_flat=tensors["mismatch_prob_sq_flat"],
            len_norm_centered_value_flat=tensors["mismatch_prob_sq_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_prob_sq_len_norm_abs_flat"],
            overall_sign_source_flat=tensors["mismatch_prob_flat"],
        )
    )
    return metrics


def compute_prob_ratio_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 old token probability 分桶统计 token probability ratio。"""
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None:
        return {}
    return _compute_bucketed_value_metrics(
        metric_root="token_p_ratio",
        old_prob_flat=tensors["old_prob_flat"],
        value_flat=tensors["prob_ratio_flat"],
        len_norm_centered_value_flat=tensors["prob_ratio_len_norm_flat"],
        len_norm_abs_value_flat=tensors["prob_ratio_len_norm_abs_flat"],
        center_value=1.0,
    )


def compute_mismatch_log_prob_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 response 对错统计 log-prob mismatch。

    为兼容现有指标名，correct response 仍映射到 `pos`，incorrect response 映射到 `neg`。

    每个 response 对错子集下记录：
    - signed mismatch: `old_log_probs - rollout_log_probs`
    - abs mismatch: `|old_log_probs - rollout_log_probs|`
    """
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None or "advantage_flat" not in tensors:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="token_logp_diff",
        old_prob_flat=tensors["old_prob_flat"],
        signed_value_flat=tensors["mismatch_log_prob_flat"],
        abs_value_flat=tensors["mismatch_log_prob_abs_flat"],
        len_norm_signed_value_flat=tensors["mismatch_log_prob_len_norm_flat"],
        len_norm_abs_value_flat=tensors["mismatch_log_prob_len_norm_abs_flat"],
        advantage_flat=tensors["advantage_flat"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("token_logp_diff"),
            old_prob_flat=tensors["old_prob_flat"],
            signed_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_flat"],
            abs_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_abs_flat"],
            len_norm_signed_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_log_prob_rollout_prob_weighted_len_norm_abs_flat"],
            advantage_flat=tensors["advantage_flat"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="token_logp_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_flat"],
            value_flat=tensors["mismatch_log_prob_sq_rollout_prob_weighted_flat"],
            len_norm_centered_value_flat=tensors["mismatch_log_prob_sq_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_log_prob_sq_rollout_prob_weighted_len_norm_abs_flat"],
            advantage_flat=tensors["advantage_flat"],
            overall_sign_source_flat=tensors["mismatch_log_prob_flat"],
        )
    )
    return metrics


def compute_mismatch_prob_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 response 对错统计 prob mismatch。"""
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None or "advantage_flat" not in tensors:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="token_p_diff",
        old_prob_flat=tensors["old_prob_flat"],
        signed_value_flat=tensors["mismatch_prob_flat"],
        abs_value_flat=tensors["mismatch_prob_abs_flat"],
        len_norm_signed_value_flat=tensors["mismatch_prob_len_norm_flat"],
        len_norm_abs_value_flat=tensors["mismatch_prob_len_norm_abs_flat"],
        advantage_flat=tensors["advantage_flat"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("token_p_diff"),
            old_prob_flat=tensors["old_prob_flat"],
            signed_value_flat=tensors["mismatch_prob_rollout_prob_weighted_flat"],
            abs_value_flat=tensors["mismatch_prob_rollout_prob_weighted_abs_flat"],
            len_norm_signed_value_flat=tensors["mismatch_prob_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_prob_rollout_prob_weighted_len_norm_abs_flat"],
            advantage_flat=tensors["advantage_flat"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="token_p_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_flat"],
            value_flat=tensors["mismatch_prob_sq_rollout_prob_weighted_flat"],
            len_norm_centered_value_flat=tensors["mismatch_prob_sq_rollout_prob_weighted_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_prob_sq_rollout_prob_weighted_len_norm_abs_flat"],
            advantage_flat=tensors["advantage_flat"],
            overall_sign_source_flat=tensors["mismatch_prob_flat"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="token_p_diff_sq",
            old_prob_flat=tensors["old_prob_flat"],
            value_flat=tensors["mismatch_prob_sq_flat"],
            len_norm_centered_value_flat=tensors["mismatch_prob_sq_len_norm_flat"],
            len_norm_abs_value_flat=tensors["mismatch_prob_sq_len_norm_abs_flat"],
            advantage_flat=tensors["advantage_flat"],
            overall_sign_source_flat=tensors["mismatch_prob_flat"],
        )
    )
    return metrics


def compute_prob_ratio_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 response 对错统计 token probability ratio。"""
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None or "advantage_flat" not in tensors:
        return {}
    return _compute_bucketed_value_metrics(
        metric_root="token_p_ratio",
        old_prob_flat=tensors["old_prob_flat"],
        value_flat=tensors["prob_ratio_flat"],
        len_norm_centered_value_flat=tensors["prob_ratio_len_norm_flat"],
        len_norm_abs_value_flat=tensors["prob_ratio_len_norm_abs_flat"],
        advantage_flat=tensors["advantage_flat"],
        center_value=1.0,
    )


def compute_seq_mismatch_log_prob_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 sequence-level log-prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_logp_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_logp_diff"],
        abs_value_flat=tensors["seq_logp_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_logp_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_logp_diff_len_norm_abs"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_logp_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_logp_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_logp_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_logp_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_logp_diff_rollout_prob_weighted_len_norm_abs"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_logp_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_logp_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_logp_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_logp_diff_sq_rollout_prob_weighted_len_norm_abs"],
            overall_sign_source_flat=tensors["seq_logp_diff"],
        )
    )
    return metrics


def compute_seq_mismatch_prob_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 sequence-level prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_p_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_p_diff"],
        abs_value_flat=tensors["seq_p_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_p_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_p_diff_len_norm_abs"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_p_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_p_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_p_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_p_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_p_diff_rollout_prob_weighted_len_norm_abs"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_p_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_p_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_p_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_p_diff_sq_rollout_prob_weighted_len_norm_abs"],
            overall_sign_source_flat=tensors["seq_p_diff"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_p_diff_sq",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_p_diff_sq"],
            len_norm_centered_value_flat=tensors["seq_p_diff_sq_len_norm"],
            len_norm_abs_value_flat=tensors["seq_p_diff_sq_len_norm_abs"],
            overall_sign_source_flat=tensors["seq_p_diff"],
        )
    )
    return metrics


def compute_seq_prob_ratio_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 token ratio 的算术均值。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    return _compute_bucketed_value_metrics(
        metric_root="seq_p_ratio",
        old_prob_flat=tensors["old_prob_seq"],
        value_flat=tensors["seq_prob_ratio"],
        len_norm_centered_value_flat=tensors["seq_prob_ratio_len_norm"],
        len_norm_abs_value_flat=tensors["seq_prob_ratio_len_norm_abs"],
        center_value=1.0,
    )


def compute_seq_sum_prob_ratio_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 token ratio 的算术求和。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_bucketed_value_metrics(
        metric_root="seq_sum_p_ratio",
        old_prob_flat=tensors["old_prob_seq"],
        value_flat=tensors["seq_sum_prob_ratio"],
        len_norm_centered_value_flat=tensors["seq_sum_prob_ratio_len_norm"],
        len_norm_abs_value_flat=tensors["seq_sum_prob_ratio_len_norm_abs"],
        center_value=0.0,
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_sum_p_ratio"),
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_prob_ratio_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_sum_prob_ratio_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_prob_ratio_rollout_prob_weighted_len_norm_abs"],
        )
    )
    return metrics


def compute_seq_sum_mismatch_log_prob_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 sequence-sum log-prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_sum_logp_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_sum_logp_diff"],
        abs_value_flat=tensors["seq_sum_logp_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_sum_logp_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_sum_logp_diff_len_norm_abs"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_sum_logp_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted_len_norm_abs"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_sum_logp_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_logp_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_sum_logp_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_logp_diff_sq_rollout_prob_weighted_len_norm_abs"],
            overall_sign_source_flat=tensors["seq_sum_logp_diff"],
        )
    )
    return metrics


def compute_seq_sum_mismatch_prob_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 sequence-sum prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_sum_p_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_sum_p_diff"],
        abs_value_flat=tensors["seq_sum_p_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_sum_p_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_sum_p_diff_len_norm_abs"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_sum_p_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted_len_norm_abs"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_sum_p_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_p_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_sum_p_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_p_diff_sq_rollout_prob_weighted_len_norm_abs"],
            overall_sign_source_flat=tensors["seq_sum_p_diff"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_sum_p_diff_sq",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_p_diff_sq"],
            len_norm_centered_value_flat=tensors["seq_sum_p_diff_sq_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_p_diff_sq_len_norm_abs"],
            overall_sign_source_flat=tensors["seq_sum_p_diff"],
        )
    )
    return metrics


def compute_seq_mismatch_log_prob_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 sequence-level log-prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_logp_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_logp_diff"],
        abs_value_flat=tensors["seq_logp_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_logp_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_logp_diff_len_norm_abs"],
        advantage_flat=tensors["advantage_seq"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_logp_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_logp_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_logp_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_logp_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_logp_diff_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_logp_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_logp_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_logp_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_logp_diff_sq_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
            overall_sign_source_flat=tensors["seq_logp_diff"],
        )
    )
    return metrics


def compute_seq_mismatch_prob_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 sequence-level prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_p_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_p_diff"],
        abs_value_flat=tensors["seq_p_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_p_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_p_diff_len_norm_abs"],
        advantage_flat=tensors["advantage_seq"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_p_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_p_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_p_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_p_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_p_diff_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_p_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_p_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_p_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_p_diff_sq_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
            overall_sign_source_flat=tensors["seq_p_diff"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_p_diff_sq",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_p_diff_sq"],
            len_norm_centered_value_flat=tensors["seq_p_diff_sq_len_norm"],
            len_norm_abs_value_flat=tensors["seq_p_diff_sq_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
            overall_sign_source_flat=tensors["seq_p_diff"],
        )
    )
    return metrics


def compute_seq_prob_ratio_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 token ratio 的算术均值。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    return _compute_bucketed_value_metrics(
        metric_root="seq_p_ratio",
        old_prob_flat=tensors["old_prob_seq"],
        value_flat=tensors["seq_prob_ratio"],
        len_norm_centered_value_flat=tensors["seq_prob_ratio_len_norm"],
        len_norm_abs_value_flat=tensors["seq_prob_ratio_len_norm_abs"],
        advantage_flat=tensors["advantage_seq"],
        center_value=1.0,
    )


def compute_seq_sum_prob_ratio_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 token ratio 的算术求和。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    metrics = _compute_bucketed_value_metrics(
        metric_root="seq_sum_p_ratio",
        old_prob_flat=tensors["old_prob_seq"],
        value_flat=tensors["seq_sum_prob_ratio"],
        len_norm_centered_value_flat=tensors["seq_sum_prob_ratio_len_norm"],
        len_norm_abs_value_flat=tensors["seq_sum_prob_ratio_len_norm_abs"],
        advantage_flat=tensors["advantage_seq"],
        center_value=0.0,
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_sum_p_ratio"),
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_prob_ratio_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_sum_prob_ratio_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_prob_ratio_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
        )
    )
    return metrics


def compute_seq_sum_mismatch_log_prob_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 sequence-sum log-prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_sum_logp_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_sum_logp_diff"],
        abs_value_flat=tensors["seq_sum_logp_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_sum_logp_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_sum_logp_diff_len_norm_abs"],
        advantage_flat=tensors["advantage_seq"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_sum_logp_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_logp_diff_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_sum_logp_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_logp_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_sum_logp_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_logp_diff_sq_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
            overall_sign_source_flat=tensors["seq_sum_logp_diff"],
        )
    )
    return metrics


def compute_seq_sum_mismatch_prob_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 sequence-sum prob mismatch。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    metrics = _compute_mismatch_metrics(
        metric_root="seq_sum_p_diff",
        old_prob_flat=tensors["old_prob_seq"],
        signed_value_flat=tensors["seq_sum_p_diff"],
        abs_value_flat=tensors["seq_sum_p_diff_abs"],
        len_norm_signed_value_flat=tensors["seq_sum_p_diff_len_norm"],
        len_norm_abs_value_flat=tensors["seq_sum_p_diff_len_norm_abs"],
        advantage_flat=tensors["advantage_seq"],
    )
    metrics.update(
        _compute_mismatch_metrics(
            metric_root=_rollout_prob_weighted_metric_root("seq_sum_p_diff"),
            old_prob_flat=tensors["old_prob_seq"],
            signed_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted"],
            abs_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted_abs"],
            len_norm_signed_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_p_diff_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_sum_p_diff_sq_mu_weighted",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_p_diff_sq_rollout_prob_weighted"],
            len_norm_centered_value_flat=tensors["seq_sum_p_diff_sq_rollout_prob_weighted_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_p_diff_sq_rollout_prob_weighted_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
            overall_sign_source_flat=tensors["seq_sum_p_diff"],
        )
    )
    metrics.update(
        _compute_bucketed_value_metrics(
            metric_root="seq_sum_p_diff_sq",
            old_prob_flat=tensors["old_prob_seq"],
            value_flat=tensors["seq_sum_p_diff_sq"],
            len_norm_centered_value_flat=tensors["seq_sum_p_diff_sq_len_norm"],
            len_norm_abs_value_flat=tensors["seq_sum_p_diff_sq_len_norm_abs"],
            advantage_flat=tensors["advantage_seq"],
            overall_sign_source_flat=tensors["seq_sum_p_diff"],
        )
    )
    return metrics


def compute_mismatch_token_frac_cnt_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 old token probability 分桶统计 token fraction / count。"""
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None:
        return {}
    return _compute_token_frac_cnt_metrics(old_prob_flat=tensors["old_prob_flat"])


def compute_mismatch_token_frac_cnt_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 response 对错统计 token fraction / count。"""
    tensors = _extract_mismatch_flat_tensors(batch)
    if tensors is None or "advantage_flat" not in tensors:
        return {}
    return _compute_token_frac_cnt_metrics(
        old_prob_flat=tensors["old_prob_flat"],
        advantage_flat=tensors["advantage_flat"],
    )


def compute_mismatch_seq_frac_cnt_by_old_prob_bucket_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence old probability bucket 统计 sequence fraction / count。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None:
        return {}
    return _compute_token_frac_cnt_metrics(
        old_prob_flat=tensors["old_prob_seq"],
        metric_root="seq_frac_cnt",
        frac_name="seq_frac",
        cnt_name="seq_cnt",
    )


def compute_mismatch_seq_frac_cnt_by_old_prob_and_adv_sign_metrics(batch: DataProto) -> dict[str, float]:
    """按 sequence response 对错统计 sequence fraction / count。"""
    tensors = _extract_mismatch_sequence_tensors(batch)
    if tensors is None or "advantage_seq" not in tensors:
        return {}
    return _compute_token_frac_cnt_metrics(
        old_prob_flat=tensors["old_prob_seq"],
        advantage_flat=tensors["advantage_seq"],
        metric_root="seq_frac_cnt",
        frac_name="seq_frac",
        cnt_name="seq_cnt",
    )


def compute_prob_shift_by_old_prob_bucket_metrics(
    old_log_prob: torch.Tensor,
    new_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """按 old token probability 分桶统计训练前后 token probability shift。"""
    response_mask = response_mask.bool()
    if not response_mask.any():
        return {}

    old_log_prob = old_log_prob.float()
    new_log_prob = new_log_prob.float()
    old_prob = torch.exp(old_log_prob).clamp(min=0.0, max=1.0)
    prob_shift = torch.exp(new_log_prob) - old_prob
    prob_shift_abs = prob_shift.abs()

    flat_mask = response_mask.flatten()
    old_prob_flat = old_prob.flatten()[flat_mask]
    prob_shift_flat = prob_shift.flatten()[flat_mask]
    prob_shift_abs_flat = prob_shift_abs.flatten()[flat_mask]

    metrics = _compute_bucketed_signed_and_abs_metrics(
        metric_root="prob_shift",
        old_prob_flat=old_prob_flat,
        signed_value_flat=prob_shift_flat,
        abs_value_flat=prob_shift_abs_flat,
        include_count_stats=False,
        emit_quantiles=False,
        filter_non_finite=True,
        empty_stat_value=0.0,
    )
    return _drop_abs_extrema_metrics(metrics)


def compute_prob_shift_by_old_prob_and_adv_sign_metrics(
    old_log_prob: torch.Tensor,
    new_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """按 advantage 符号统计训练前后 token probability shift。"""
    response_mask = response_mask.bool()
    if not response_mask.any():
        return {}

    old_log_prob = old_log_prob.float()
    new_log_prob = new_log_prob.float()
    advantages = advantages.float()
    old_prob = torch.exp(old_log_prob).clamp(min=0.0, max=1.0)
    prob_shift = torch.exp(new_log_prob) - old_prob
    prob_shift_abs = prob_shift.abs()

    flat_mask = response_mask.flatten()
    old_prob_flat = old_prob.flatten()[flat_mask]
    advantages_flat = advantages.flatten()[flat_mask]
    prob_shift_flat = prob_shift.flatten()[flat_mask]
    prob_shift_abs_flat = prob_shift_abs.flatten()[flat_mask]

    metrics = _compute_bucketed_signed_and_abs_metrics(
        metric_root="prob_shift",
        old_prob_flat=old_prob_flat,
        signed_value_flat=prob_shift_flat,
        abs_value_flat=prob_shift_abs_flat,
        advantage_flat=advantages_flat,
        include_count_stats=False,
        emit_quantiles=False,
        filter_non_finite=True,
        empty_stat_value=0.0,
    )
    return _drop_abs_extrema_metrics(metrics)


def get_ratio_tensors_for_accumulation(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw ratio and seq-level π_train/π_rollout (1D) for one micro/mini batch, to be concatenated for full-batch stats.

    Returns:
        token_flat, rho_seq, rho_seq_geo, pi_train_seq, pi_rollout_seq, pi_train_seq_geo, pi_rollout_seq_geo,
        diff_seq_sum_add: sum_t (π_train_t - π_rollout_t) per sequence (连加).
    """
    if rollout_log_prob is None or not response_mask.any():
        empty = torch.empty(0, device=old_log_prob.device, dtype=old_log_prob.dtype)
        return (empty, empty, empty, empty, empty, empty, empty, empty)
    log_ratio = old_log_prob - rollout_log_prob
    rho_token = torch.exp(log_ratio)
    token_flat = rho_token[response_mask.bool()].flatten()
    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    rho_seq = torch.exp(log_ratio_sum)
    log_ratio_mean = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
    rho_seq_geo = torch.exp(log_ratio_mean)
    log_prob_train_sum = verl_F.masked_sum(old_log_prob, response_mask, axis=-1)
    log_prob_rollout_sum = verl_F.masked_sum(rollout_log_prob, response_mask, axis=-1)
    pi_train_seq = torch.exp(log_prob_train_sum)
    pi_rollout_seq = torch.exp(log_prob_rollout_sum)
    log_prob_train_mean = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
    log_prob_rollout_mean = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
    pi_train_seq_geo = torch.exp(log_prob_train_mean)
    pi_rollout_seq_geo = torch.exp(log_prob_rollout_mean)
    # 连加: 先 token-level 概率差，再对序列求和
    pi_train_token = torch.exp(old_log_prob)
    pi_rollout_token = torch.exp(rollout_log_prob)
    diff_seq_sum_add = verl_F.masked_sum(pi_train_token - pi_rollout_token, response_mask, axis=-1)
    return token_flat, rho_seq, rho_seq_geo, pi_train_seq, pi_rollout_seq, pi_train_seq_geo, pi_rollout_seq_geo, diff_seq_sum_add


def compute_ratio_stats_from_tensors(
    token_flat: torch.Tensor,
    rho_seq: torch.Tensor,
    rho_seq_geo: torch.Tensor,
    pi_train_seq: Optional[torch.Tensor] = None,
    pi_rollout_seq: Optional[torch.Tensor] = None,
    pi_train_seq_geo: Optional[torch.Tensor] = None,
    pi_rollout_seq_geo: Optional[torch.Tensor] = None,
    diff_seq_sum_add: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute ratio and seq-level π_train/π_rollout/diff stats from pre-concatenated full-batch tensors.

    Used in actor after accumulating tensors across all micro batches so std/median are true batch stats.
    If pi_* are provided, reports pi_train_seq, pi_rollout_seq, diff_seq_sum, diff_seq_geo, diff_seq_sum_add (连加).
    """
    metrics: dict[str, float] = {}

    def _stats(tensor: torch.Tensor, prefix: str) -> None:
        if tensor.numel() == 0:
            for s in ["mean", "max", "min", "std", "median", "p90", "p95", "p98"]:
                metrics[f"{prefix}_{s}"] = 0.0
            if prefix == "ratio_seq_geo":
                for s in ["p60", "p70", "p80", "p85"]:
                    metrics[f"{prefix}_{s}"] = 0.0
            return
        metrics[f"{prefix}_mean"] = tensor.mean().item()
        metrics[f"{prefix}_max"] = tensor.max().item()
        metrics[f"{prefix}_min"] = tensor.min().item()
        metrics[f"{prefix}_std"] = tensor.std().item() if tensor.numel() > 1 else 0.0
        metrics[f"{prefix}_median"] = tensor.median().item()
        if tensor.numel() >= 2:
            q = torch.tensor([0.90, 0.95, 0.98], device=tensor.device, dtype=tensor.dtype)
            p90, p95, p98 = torch.quantile(tensor.float(), q).tolist()
        else:
            p90 = p95 = p98 = tensor.item()
        metrics[f"{prefix}_p90"] = p90
        metrics[f"{prefix}_p95"] = p95
        metrics[f"{prefix}_p98"] = p98
        if prefix == "ratio_seq_geo":
            if tensor.numel() >= 2:
                q_mid = torch.tensor([0.60, 0.70, 0.80, 0.85], device=tensor.device, dtype=tensor.dtype)
                p60, p70, p80, p85 = torch.quantile(tensor.float(), q_mid).tolist()
            else:
                p60 = p70 = p80 = p85 = tensor.item()
            metrics[f"{prefix}_p60"] = p60
            metrics[f"{prefix}_p70"] = p70
            metrics[f"{prefix}_p80"] = p80
            metrics[f"{prefix}_p85"] = p85

    _stats(token_flat, "ratio_token")
    _stats(rho_seq, "ratio_seq")
    _stats(rho_seq_geo, "ratio_seq_geo")
    if (
        pi_train_seq is not None
        and pi_rollout_seq is not None
        and pi_train_seq.numel() > 0
    ):
        _stats(pi_train_seq - pi_rollout_seq, "diff_seq_sum")
    if (
        pi_train_seq_geo is not None
        and pi_rollout_seq_geo is not None
        and pi_train_seq_geo.numel() > 0
    ):
        _stats(pi_train_seq_geo - pi_rollout_seq_geo, "diff_seq_geo")
    if diff_seq_sum_add is not None and diff_seq_sum_add.numel() > 0:
        _stats(diff_seq_sum_add, "diff_seq_sum_add")
    return metrics


def compute_rollout_correction_and_add_to_batch(
    batch: DataProto, rollout_corr_config: RolloutCorrectionConfig
) -> tuple[DataProto, dict]:
    """Compute rollout correction weights and apply rejection sampling.

    Computes importance sampling weights to correct for off-policy issues between
    rollout and training policies. Applies rejection sampling by modifying response_mask.
    Always updates response_mask; conditionally adds IS weights.

    Key behavior:
    - response_mask: ALWAYS updated with rejection (veto + optional RS excluded from training)
    - rollout_is_weights: Added to batch ONLY if rollout_is parameter is set

    This separation ensures:
    - Rejection works independently of IS weight application
    - Metrics can be monitored before enabling IS weight correction

    Args:
        batch: DataProto with old_log_probs, rollout_log_probs, response_mask

    Returns:
        Tuple of (updated_batch, metrics):
            updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if enabled)
            metrics: Dict of IS and off-policy metrics, all with "rollout_corr/" prefix

    Note:
        The implementation is copied from szrlee <szrlee@gmail.com>.
    """
    # Get new API parameters directly from config
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # Compute IS weights and get modified response_mask
    rollout_is_weights, modified_response_mask, rollout_corr_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=batch.batch["response_mask"],
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
    )

    # ALWAYS update response_mask with rejection applied
    batch.batch["response_mask"] = modified_response_mask

    # Add IS weights to batch if computed
    if rollout_is_weights is not None:
        batch = batch.union(rollout_is_weights)

    return batch, rollout_corr_metrics


def compute_rollout_corr_metrics_only(batch: DataProto) -> dict[str, float]:
    """Compute rollout correction metrics only (no IS weights, no rejection, no batch modification).

    Use in the trainer when rollout_log_probs are available but correction is disabled or bypass,
    so that all rollout_corr/ metrics (off-policy + ratio stats) are still recorded for monitoring.

    Requires batch.batch to contain: old_log_probs, rollout_log_probs, response_mask.

    Returns:
        Dict of metrics with "rollout_corr/" prefix (off-policy + ratio_token/seq/seq_geo stats).
    """
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return {}
    old_log_prob = batch.batch["old_log_probs"]
    rollout_log_prob = batch.batch["rollout_log_probs"]
    response_mask = batch.batch["response_mask"]
    if not response_mask.any():
        return {}
    offpolicy = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    ratio_stats = compute_ratio_stats(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    out = {}
    for key, value in {**offpolicy, **ratio_stats}.items():
        if isinstance(value, torch.Tensor):
            out[f"rollout_corr/{key}"] = value.item()
        else:
            out[f"rollout_corr/{key}"] = value
    return out

def compute_raw_penalty_per_sequence(
    batch: DataProto,
    ratio_type: str,
    penalty_cost_start_step: Optional[int] = None,
    global_step: Optional[int] = None,
    clip_penalty_low: Optional[float] = None,
    clip_penalty_high: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Compute per-sequence raw penalty for local ratio modes before normalization.

    Returns:
        Tensor of shape (B,) on the same device as logprobs, or None if unsupported / skipped / missing inputs.
    """
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return None
    if ratio_type in ("seq_global", "seq_geo_global"):
        return None
    if (
        penalty_cost_start_step is not None
        and global_step is not None
        and global_step < penalty_cost_start_step
    ):
        return None

    old_log_prob = batch.batch["old_log_probs"]
    rollout_log_prob = batch.batch["rollout_log_probs"]
    response_mask = batch.batch["response_mask"]
    log_ratio = old_log_prob - rollout_log_prob

    if ratio_type == "seq":
        log_agg = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
        raw_per_seq = torch.exp(log_agg)
    elif ratio_type == "seq_geo":
        log_agg = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
        raw_per_seq = torch.exp(log_agg)
    elif ratio_type == "seq_geo_no_exp":
        raw_per_seq = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
    elif ratio_type == "seq_geo_no_exp_sum":
        raw_per_seq = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    elif ratio_type == "seq_prob_diff":
        raw_per_seq = verl_F.masked_mean(torch.exp(old_log_prob) - torch.exp(rollout_log_prob), response_mask, axis=-1)
    elif ratio_type == "seq_prob_diff_sum":
        raw_per_seq = verl_F.masked_sum(torch.exp(old_log_prob) - torch.exp(rollout_log_prob), response_mask, axis=-1)
    elif ratio_type == "seq_geo_inverse":
        log_agg = verl_F.masked_mean(-log_ratio, response_mask, axis=-1)
        raw_per_seq = torch.exp(log_agg)
    elif ratio_type == "log_ppl_abs_diff":
        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
        mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        raw_per_seq = log_ppl_diff.abs()
    elif ratio_type == "seq_kl":
        log_prob_diff = rollout_log_prob - old_log_prob
        seq_kl = verl_F.masked_sum(log_prob_diff, response_mask, axis=-1)
        raw_per_seq = seq_kl
    elif ratio_type == "seq_forward_kl":
        seq_log_ratio = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
        importance_weights_seq = torch.exp(seq_log_ratio)
        raw_per_seq = importance_weights_seq * seq_log_ratio
    elif ratio_type == "seq_jeffreys":
        log_prob_diff = rollout_log_prob - old_log_prob
        seq_log_prob_diff = verl_F.masked_sum(log_prob_diff, response_mask, axis=-1)
        seq_log_ratio = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
        importance_weights_seq = torch.exp(seq_log_ratio)
        raw_per_seq = importance_weights_seq * seq_log_ratio + seq_log_prob_diff
    elif ratio_type == "diff_seq_sum_add":
        pi_train_token = torch.exp(old_log_prob)
        pi_rollout_token = torch.exp(rollout_log_prob)
        diff_token = (pi_train_token - pi_rollout_token).abs()
        raw_per_seq = verl_F.masked_sum(diff_token, response_mask, axis=-1)
    elif ratio_type == "diff_seq_geo_abs":
        log_prob_train_mean = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
        log_prob_rollout_mean = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
        pi_train_seq_geo = torch.exp(log_prob_train_mean)
        pi_rollout_seq_geo = torch.exp(log_prob_rollout_mean)
        diff_seq_geo = pi_train_seq_geo - pi_rollout_seq_geo
        raw_per_seq = diff_seq_geo.abs()
    else:
        seq_log_ratio = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
        seq_k3 = torch.exp(seq_log_ratio) - seq_log_ratio - 1
        raw_per_seq = seq_k3

    return _apply_penalty_clip_band(
        raw_penalty=raw_per_seq,
        clip_penalty_low=clip_penalty_low,
        clip_penalty_high=clip_penalty_high,
    )


def compute_raw_penalty_per_token(
    batch: DataProto,
    ratio_type: str,
    penalty_cost_start_step: Optional[int] = None,
    global_step: Optional[int] = None,
    clip_penalty_low: Optional[float] = None,
    clip_penalty_high: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Compute per-token raw penalty for local ratio modes before normalization."""
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return None
    if ratio_type in ("seq_global", "seq_geo_global"):
        return None
    if (
        penalty_cost_start_step is not None
        and global_step is not None
        and global_step < penalty_cost_start_step
    ):
        return None

    old_log_prob = batch.batch["old_log_probs"]
    rollout_log_prob = batch.batch["rollout_log_probs"]
    response_mask = batch.batch["response_mask"]
    valid_mask = response_mask.to(dtype=torch.bool, device=old_log_prob.device)
    log_ratio = old_log_prob - rollout_log_prob

    if ratio_type in ("seq", "seq_geo"):
        raw_per_token = torch.exp(log_ratio)
    elif ratio_type == "token_logp_diff":
        raw_per_token = log_ratio
    elif ratio_type == "token_prob_diff":
        raw_per_token = torch.exp(old_log_prob) - torch.exp(rollout_log_prob)
    elif ratio_type == "token_prob_ratio":
        raw_per_token = torch.exp(log_ratio) - 1.0
    elif ratio_type == "seq_geo_inverse":
        raw_per_token = torch.exp(-log_ratio)
    elif ratio_type == "log_ppl_abs_diff":
        raw_per_token = (rollout_log_prob - old_log_prob).abs()
    elif ratio_type == "seq_kl":
        raw_per_token = rollout_log_prob - old_log_prob
    elif ratio_type == "seq_forward_kl":
        importance_weights_token = torch.exp(log_ratio)
        raw_per_token = importance_weights_token * log_ratio
    elif ratio_type == "seq_jeffreys":
        importance_weights_token = torch.exp(log_ratio)
        raw_per_token = importance_weights_token * log_ratio + (rollout_log_prob - old_log_prob)
    elif ratio_type in ("diff_seq_sum_add", "diff_seq_geo_abs"):
        pi_train_token = torch.exp(old_log_prob)
        pi_rollout_token = torch.exp(rollout_log_prob)
        raw_per_token = (pi_train_token - pi_rollout_token).abs()
    else:
        raw_per_token = torch.exp(log_ratio) - log_ratio - 1

    raw_per_token = _apply_penalty_clip_band(
        raw_penalty=raw_per_token,
        clip_penalty_low=clip_penalty_low,
        clip_penalty_high=clip_penalty_high,
        valid_mask=valid_mask,
    )
    return raw_per_token * response_mask.to(dtype=raw_per_token.dtype, device=raw_per_token.device)


def compute_normalized_penalty_per_sequence(
    batch: DataProto,
    ratio_type: str,
    max_ratio: Optional[float] = None,
    penalty_cost_start_step: Optional[int] = None,
    global_step: Optional[int] = None,
    clip_penalty_low: Optional[float] = None,
    clip_penalty_high: Optional[float] = None,
    normalize_mode: Optional[str] = None,
    normalize_tanh_k: Optional[float] = None,
    penalty_truncated: Optional[Union[float, bool]] = None,
) -> Optional[torch.Tensor]:
    """Compute per-sequence normalized penalty for local ratio modes."""
    raw_per_seq = compute_raw_penalty_per_sequence(
        batch=batch,
        ratio_type=ratio_type,
        penalty_cost_start_step=penalty_cost_start_step,
        global_step=global_step,
        clip_penalty_low=clip_penalty_low,
        clip_penalty_high=clip_penalty_high,
    )
    return normalize_penalty_per_sequence(
        raw_per_seq=raw_per_seq,
        max_ratio=max_ratio,
        normalize_mode=normalize_mode,
        normalize_tanh_k=normalize_tanh_k,
        penalty_truncated=penalty_truncated,
        log_probs=batch.batch["old_log_probs"],
        response_mask=batch.batch["response_mask"],
    )


def _compute_one_minus_prob_weight(
    raw_penalty: torch.Tensor,
    log_probs: Optional[torch.Tensor],
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the ``1 - exp(log_prob)`` weight matching the penalty tensor shape."""
    if log_probs is None:
        raise ValueError("normalize_mode='one_minus_prob' requires log_probs.")

    safe_log_probs = torch.clamp(log_probs.to(device=raw_penalty.device), max=0.0)
    if raw_penalty.shape == safe_log_probs.shape:
        return (1.0 - torch.exp(safe_log_probs)).clamp(min=0.0, max=1.0).to(dtype=raw_penalty.dtype)

    if raw_penalty.ndim != 1:
        raise ValueError(
            f"one_minus_prob weight expects 1D or shape-matched penalties, got penalty shape {tuple(raw_penalty.shape)} "
            f"and log_probs shape {tuple(safe_log_probs.shape)}"
        )

    if safe_log_probs.ndim == 1:
        if safe_log_probs.shape != raw_penalty.shape:
            raise ValueError(
                f"1D log_probs shape {tuple(safe_log_probs.shape)} must match penalty shape {tuple(raw_penalty.shape)}"
            )
        return (1.0 - torch.exp(safe_log_probs)).clamp(min=0.0, max=1.0).to(dtype=raw_penalty.dtype)

    if response_mask is None:
        raise ValueError("normalize_mode='one_minus_prob' requires response_mask for sequence-level penalties.")
    if safe_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"log_probs shape {tuple(safe_log_probs.shape)} must match response_mask shape {tuple(response_mask.shape)}"
        )

    seq_log_prob = verl_F.masked_mean(safe_log_probs, response_mask, axis=-1)
    return (1.0 - torch.exp(seq_log_prob)).clamp(min=0.0, max=1.0).to(dtype=raw_penalty.dtype)


def normalize_penalty_per_sequence(
    raw_per_seq: Optional[torch.Tensor],
    max_ratio: Optional[float] = None,
    normalize_mode: Optional[str] = None,
    normalize_tanh_k: Optional[float] = None,
    penalty_truncated: Optional[Union[float, bool]] = None,
    log_probs: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Normalize a precomputed raw penalty tensor.

    The implementation is shape-agnostic and preserves the input shape, so it can be used
    for either response-level tensors with shape ``(B,)`` or token-level tensors with shape
    ``(B, T)``. ``normalize_mode='one_minus_prob'`` additionally requires the matching
    ``log_probs`` tensor; response-level penalties use the masked mean log-prob per sequence.
    """
    if raw_per_seq is None:
        return None
    if normalize_mode is None or normalize_mode == "none":
        penalty_per_seq = raw_per_seq
    elif "clip_negative" in normalize_mode:
        penalty_per_seq = raw_per_seq.clamp(min=0.0)
        if "sigmoid" in normalize_mode:
            penalty_per_seq = torch.sigmoid(penalty_per_seq) - 0.5
    elif "absneg" in normalize_mode:
        mode, neg_clip_ratio, submode = normalize_mode.split("_")
        pos_penalty_per_seq = raw_per_seq.clamp(min=0.0)
        neg_penalty_per_seq = raw_per_seq.clamp(max=float(neg_clip_ratio))
        neg_penalty_per_seq[neg_penalty_per_seq == float(neg_clip_ratio)] = 0
        neg_penalty_per_seq = neg_penalty_per_seq.abs()
        if submode == "none":
            penalty_per_seq = pos_penalty_per_seq + neg_penalty_per_seq
        elif submode == "sigmoid":
            penalty_per_seq = pos_penalty_per_seq + neg_penalty_per_seq
            penalty_per_seq = torch.sigmoid(penalty_per_seq) - 0.5
        elif submode == "maxnorm":
            if neg_penalty_per_seq.max() > 0 and pos_penalty_per_seq.max() > 0:
                neg_penalty_per_seq = neg_penalty_per_seq / neg_penalty_per_seq.max() * pos_penalty_per_seq.max()
                penalty_per_seq = pos_penalty_per_seq + neg_penalty_per_seq
            else:
                penalty_per_seq = pos_penalty_per_seq + neg_penalty_per_seq
        else:
            raise NotImplementedError(normalize_mode)
    elif normalize_mode == "rational":
        penalty_per_seq = raw_per_seq / (1.0 + raw_per_seq)
    elif normalize_mode == "sigmoid":
        penalty_per_seq = torch.sigmoid(raw_per_seq) - 0.5
    elif normalize_mode == "cubed":
        rational = raw_per_seq / (1.0 + raw_per_seq)
        penalty_per_seq = rational**3
    elif normalize_mode == "tanh":
        k = 16 if normalize_tanh_k is None else float(normalize_tanh_k)
        penalty_per_seq = 0.45 + 0.05 * torch.tanh(k * (raw_per_seq - 1.0))
    elif normalize_mode in ("one_minus_prob", "prob_weighted", "one_minus_exp_log_prob"):
        prob_weight = _compute_one_minus_prob_weight(
            raw_penalty=raw_per_seq,
            log_probs=log_probs,
            response_mask=response_mask,
        )
        penalty_per_seq = raw_per_seq * prob_weight
    else:
        raise ValueError(
            f"normalize_mode must be one of 'none'|'rational'|'sigmoid'|'cubed'|'tanh'|'one_minus_prob', got {normalize_mode!r}"
        )

    if max_ratio is not None:
        penalty_per_seq = max_ratio * penalty_per_seq

    if penalty_truncated is not None:
        if isinstance(penalty_truncated, bool):
            trunc_max = 0.5 if penalty_truncated else None
        else:
            trunc_max = float(penalty_truncated)
        if trunc_max is not None:
            penalty_per_seq = penalty_per_seq.clamp(max=trunc_max)

    return penalty_per_seq


def apply_ratio_penalty_to_reward(
    batch: DataProto,
    ratio_type: str,
    max_ratio: Optional[float] = None,
    penalty_level: str = "sequence",
    normalize_by_response_length: bool = False,
    penalty_cost_start_step: Optional[int] = None,
    global_step: Optional[int] = None,
    clip_penalty_low: Optional[float] = None,
    clip_penalty_high: Optional[float] = None,
    normalize_mode: Optional[str] = None,
    normalize_tanh_k: Optional[float] = None,
    penalty_truncated: Optional[Union[float, bool]] = None,
) -> dict[str, float]:
    """Apply a ratio-based penalty to token_level_rewards.

    Ratio = π_train / π_rollout (old_log_prob - rollout_log_prob in log space), range (0, +∞).

    The penalty can be applied either:
    - at ``sequence`` level: one scalar per response, scattered to the last valid token.
    - at ``token`` level: one value per valid token, divided by response length so the
      total sequence penalty stays length-normalized.

    Supports local/global ratio normalization modes:

    Local modes:
    compute a per-sequence `raw_per_seq` according to `ratio_type`, then map it into
    `penalty_per_seq` using `normalize_mode` (default "rational").
    - "seq": exp(sum_t log_ratio)
    - "seq_geo": exp(mean_t log_ratio)
    - "seq_geo_no_exp": mean_t log_ratio
    - "log_ppl_abs_diff": |mean_log_prob_rollout - mean_log_prob_training|
    - "seq_kl": sum_t(rollout_log_prob - old_log_prob)
    - "seq_forward_kl": exp(sum_t log_ratio) * sum_t log_ratio
    - "seq_jeffreys": exp(seq_log_ratio) * seq_log_ratio + seq_log_prob_diff
    - "diff_seq_sum_add": sum_t |pi_train_t - pi_rollout_t|
    - "diff_seq_geo_abs": |exp(mean_t log pi_train) - exp(mean_t log pi_rollout)|
    - "seq_k3_kl": exp(seq_log_ratio) - seq_log_ratio - 1

    Global normalization (penalty subtracted first, then the combined score is min-max normalized):
    - "seq_global":     exp(Σ log_ratio) per sequence; final_score = minmax(reward - ratio).
    - "seq_geo_global": exp(mean log_ratio) per sequence; final_score = minmax(reward - ratio).

    Modifies batch.batch["token_level_rewards"] in place.

    Args:
        batch: DataProto with old_log_probs, rollout_log_probs, response_mask, token_level_rewards.
        ratio_type: One of "seq", "seq_geo", "seq_geo_no_exp", "log_ppl_abs_diff", "seq_kl",
            "seq_jeffreys", "diff_seq_sum_add", "seq_forward_kl", "diff_seq_geo_abs",
            "seq_geo_inverse", "seq_k3_kl" (or "seq_global", "seq_geo_global").
        max_ratio: If set, normalize as ratio / (max_ratio + ratio) so penalty ∈ (0, 1) without clamp; else use ratio/(1+ratio).
        penalty_level: ``sequence`` or ``token``.
        normalize_by_response_length: When using ``penalty_level=token``, divide valid-token
            penalties by response length before applying them.
        penalty_cost_start_step: If set with global_step, penalty is applied only when global_step >= this value (shared with cost in compute_advantage).
        global_step: Current training step; used with penalty_cost_start_step for staged penalty.
        clip_penalty_low: Bilateral clip on `raw_per_seq` before mapping (not applied when normalize_mode is
            "tanh_inverse", since that path only uses the tanh formula on inverse_ratio).
        clip_penalty_high: See clip_penalty_low.
        normalize_mode: How to map `raw_per_seq` into `penalty_per_seq` (default "rational").
            Options: "rational" | "sigmoid" | "cubed" | "tanh_inverse" | "one_minus_prob".
            ``one_minus_prob`` means ``raw_penalty * (1 - exp(log_prob))`` where ``log_prob``
            comes from ``old_log_probs``. Sequence-level penalties use the masked mean log-prob.
        normalize_tanh_k: Used when normalize_mode == "tanh_inverse": coefficient k in
            penalty = 0.45 + 0.05 * tanh(k * (inverse_ratio - 1)), where
            inverse_ratio = exp(mean_t(-log_ratio)) = π_rollout/π_train (geo). Defaults to 1.0 if unset.
        penalty_truncated: Upper cap for the final penalty after `max_ratio` scaling.
            - None / missing: no clamp.
            - float: `penalty_per_seq = penalty_per_seq.clamp(max=penalty_truncated)`.
            - Legacy bool: True -> clamp max 0.5; False -> no clamp.
    """
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return {}
    if "token_level_rewards" not in batch.batch:
        return {}
    if (
        penalty_cost_start_step is not None
        and global_step is not None
        and global_step < penalty_cost_start_step
    ):
        return {}
    old_log_prob = batch.batch["old_log_probs"]
    rollout_log_prob = batch.batch["rollout_log_probs"]
    response_mask = batch.batch["response_mask"]
    log_ratio = old_log_prob - rollout_log_prob
    penalty_level = normalize_penalty_level(penalty_level)

    # Index of last valid token per sequence (shared by all modes)
    last_idx = response_mask.float().sum(dim=-1).long() - 1  # (B,)
    last_idx = last_idx.clamp(min=0)
    row_idx = torch.arange(last_idx.shape[0], device=last_idx.device)

    # --- Compute per-sequence raw penalty source only (same definitions as rollout_corr metrics) ---
    if ratio_type in _LOCAL_RATIO_PENALTY_TYPES:
        if penalty_level == "token":
            raw_per_token = compute_raw_penalty_per_token(
                batch=batch,
                ratio_type=ratio_type,
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
            )
            if raw_per_token is None:
                return {}
            penalty_per_token = normalize_penalty_per_sequence(
                raw_per_seq=raw_per_token,
                max_ratio=max_ratio,
                normalize_mode=normalize_mode,
                normalize_tanh_k=normalize_tanh_k,
                penalty_truncated=penalty_truncated,
                log_probs=old_log_prob,
                response_mask=response_mask,
            )
            if penalty_per_token is None:
                return {}
            if normalize_by_response_length:
                penalty_2d = scale_token_penalty_by_response_length(
                    penalty_per_token=penalty_per_token,
                    response_mask=response_mask,
                )
            else:
                penalty_2d = penalty_per_token * response_mask.to(
                    dtype=penalty_per_token.dtype,
                    device=penalty_per_token.device,
                )
            penalty_2d = penalty_2d.to(batch.batch["token_level_rewards"].dtype)
            batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] - penalty_2d
            batch.batch["rollout_penalty"] = penalty_2d.clone()
            metrics_vals = penalty_2d[response_mask.bool()]
        else:
            penalty_per_seq = compute_normalized_penalty_per_sequence(
                batch=batch,
                ratio_type=ratio_type,
                max_ratio=max_ratio,
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
                normalize_mode=normalize_mode,
                normalize_tanh_k=normalize_tanh_k,
                penalty_truncated=penalty_truncated,
            )
            if penalty_per_seq is None:
                return {}
            penalty_2d = torch.zeros_like(batch.batch["token_level_rewards"])
            penalty_2d[row_idx, last_idx] = penalty_per_seq.to(penalty_2d.dtype)
            batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] - penalty_2d
            batch.batch["rollout_penalty"] = penalty_2d.clone()
            metrics_vals = penalty_per_seq

    elif ratio_type in ("seq_global", "seq_geo_global"):
        if penalty_level == "token":
            raise ValueError(f"Global ratio penalty type {ratio_type!r} does not support penalty_level='token'")
        # --- Global normalization: subtract raw ratio from reward, then min-max normalize ---
        if ratio_type == "seq_global":
            log_agg = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
        else:
            log_agg = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
        rho = torch.exp(log_agg)  # (B,) in (0, +∞)

        # Bilateral clip: zero out rho when low < rho < high. None for low/high means no bound (-inf / +inf).
        if clip_penalty_low is not None or clip_penalty_high is not None:
            effective_low = clip_penalty_low if clip_penalty_low is not None else float("-inf")
            effective_high = clip_penalty_high if clip_penalty_high is not None else float("inf")
            effective_rho = rho.clone()
            effective_rho[(rho > effective_low) & (rho < effective_high)] = 0
        else:
            effective_rho = rho

        # Original per-sequence reward (scalar at last token, zeros elsewhere)
        original_reward = batch.batch["token_level_rewards"].sum(dim=-1)  # (B,)

        # Subtract raw ratio, then min-max normalize across the batch to [0, 1]
        combined = original_reward - effective_rho  # (B,), unbounded
        normalized = (combined - combined.mean()) / (combined.std() + 1e-8)

        new_rewards = torch.zeros_like(batch.batch["token_level_rewards"])
        new_rewards[row_idx, last_idx] = normalized.to(new_rewards.dtype)
        batch.batch["token_level_rewards"] = new_rewards
        penalty_2d = torch.zeros_like(batch.batch["token_level_rewards"])
        penalty_2d[row_idx, last_idx] = effective_rho.to(penalty_2d.dtype)
        batch.batch["rollout_penalty"] = penalty_2d

        # Report the effective penalty (what was subtracted per sequence)
        metrics_vals = effective_rho

    else:
        raise ValueError(
            f"ratio_type must be one of seq, seq_geo, seq_geo_no_exp, log_ppl_abs_diff, "
            f"seq_kl, seq_jeffreys, diff_seq_sum_add, seq_forward_kl, diff_seq_geo_abs, "
            f"seq_geo_inverse, seq_k3_kl, seq_global, seq_geo_global, got {ratio_type!r}"
        )

    if metrics_vals.numel() == 0:
        return {}

    return {
        "critic/ratio_penalty/mean": metrics_vals.mean().item(),
        "critic/ratio_penalty/max": metrics_vals.max().item(),
        "critic/ratio_penalty/min": metrics_vals.min().item(),
    }


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout correction metrics from log probabilities during training.

    This function is used in the **actor** to compute metrics using the CURRENT policy
    π_θ vs π_rollout (per micro-batch). Use prefix "rollout_corr_actor/" so they do not
    collide with trainer's "rollout_corr/" (full-batch π_old vs π_rollout).

    It computes off-policy diagnostic metrics (KL, PPL, χ²) and ratio stats
    (token/seq/seq_geo mean,max,min,std,median) from log probabilities.

    Args:
        log_prob: Current policy log probabilities, shape (batch_size, seq_length)
        rollout_log_prob: Rollout policy log probabilities, shape (batch_size, seq_length)
        response_mask: Valid token mask, shape (batch_size, seq_length)

    Returns:
        Dictionary of metrics with "rollout_corr_actor/" prefix (actor / π_θ vs π_rollout)
    """
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    ratio_stats = compute_ratio_stats(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    # Actor prefix: distinct from trainer's rollout_corr/ (step-level π_old vs π_rollout)
    metrics_with_prefix = {}
    for key, value in {**offpolicy_metrics, **ratio_stats}.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"rollout_corr_actor/{key}"] = value.item()
        else:
            metrics_with_prefix[f"rollout_corr_actor/{key}"] = value

    return metrics_with_prefix


def apply_rollout_correction(
    batch: DataProto,
    rollout_corr_config: Optional[RolloutCorrectionConfig] = None,
    policy_loss_config: PolicyLossConfig = None,
) -> None:
    """
    BYPASS MODE: Use rollout_log_probs as old_log_probs
    Skips expensive actor forward pass for old_log_prob computation

    Two sub-modes (controlled by use_policy_gradient):
    1. Bypass + PPO loss (use_policy_gradient=False, default):
       - Uses standard PPO loss function with old_log_prob=rollout_log_prob
       - PPO clips ratio π_θ/π_rollout instead of π_θ/π_old

    2. Bypass + Policy Gradient loss (use_policy_gradient=True):
       - Uses compute_policy_loss_with_rollout_correction()
       - Policy gradient (REINFORCE-style) with IS/RS correction applied
       - No PPO clipping

    Note:
        The implementation is copied from szrlee <szrlee@gmail.com>.
    """
    from omegaconf import open_dict

    if "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "bypass_mode=True requires rollout_log_probs in batch. "
            "Ensure rollout worker is configured to calculate_log_probs=true."
        )

    # Use rollout log probs as old log probs (zero-cost substitution)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    with open_dict(policy_loss_config):
        # Always pass rollout_correction config to actor for metrics computation
        policy_loss_config["rollout_correction"] = rollout_corr_config

    # Check if policy gradient loss mode is enabled
    use_policy_gradient = rollout_corr_config.get("use_policy_gradient", False)

    if use_policy_gradient:
        # Policy gradient mode: Configure actor to use rollout_correction loss function
        # This will use compute_policy_loss_with_rollout_correction (no PPO clipping)
        policy_loss_config["loss_mode"] = "rollout_correction"
