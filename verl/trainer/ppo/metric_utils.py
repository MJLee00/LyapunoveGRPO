# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.torch_functional import masked_mean, masked_sum
import verl.utils.torch_functional as verl_F


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _safe_mean(tensor: torch.Tensor, mask: torch.Tensor = None) -> float:
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        tensor = tensor[mask]
    if tensor.numel() == 0:
        return 0.0
    return tensor.mean().item()


def _safe_max(tensor: torch.Tensor, mask: torch.Tensor = None) -> float:
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        tensor = tensor[mask]
    if tensor.numel() == 0:
        return 0.0
    return tensor.max().item()


def _safe_min(tensor: torch.Tensor, mask: torch.Tensor = None, min_value=None) -> float:
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        tensor = tensor[mask]
    if tensor.numel() == 0:
        return 0.0
    if min_value is not None:
        return max(tensor.min().item(), min_value)
    else:
        return tensor.min().item()


def _safe_std(tensor: torch.Tensor, mask: torch.Tensor = None) -> float:
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        tensor = tensor[mask]
    if tensor.numel() == 0:
        return 0.0
    return tensor.std().item()


def _safe_topk_mean(tensor: torch.Tensor, mask: torch.Tensor = None, k: int = 0, percent: float = 0.20) -> float:
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        tensor = tensor[mask]
    if tensor.numel() == 0:
        return 0.0
    if k > 0:
        num = min(k, tensor.numel())
    else:
        num = min(int(percent * tensor.numel()), tensor.numel())

    return torch.topk(tensor, k=num).values.mean().item()


@torch.no_grad()
def _compute_rankme(X: torch.Tensor, eps: float = 1e-7):
    """
    Compute RankMe = exp( -sum p_i log p_i ) where p_i are normalized eigenvalues of covariance.

    Args:
        X: (n, d) float tensor. n samples, d dim repr.
        eps: numerical epsilon.
        center: subtract mean over samples.
        l2_normalize: normalize each sample vector to unit norm before center/cov.

    Returns:
        rankme: scalar tensor
        p: (k,) normalized eigenvalues (k = min(n, d), but using Gram gives k <= n)
    """
    n, d = X.shape

    # # (1) Optional L2 normalization across feature dim
    # if l2_normalize:
    #     X = X / (X.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))

    # # (2) Optional centering across samples
    # if center:
    #     X = X - X.mean(dim=0, keepdim=True)

    # (3) One SVD: X = U diag(s) Vh
    # full_matrices=False gives U:(n,k), s:(k,), Vh:(k,d), k=min(n,d)
    # singular_values = torch.linalg.svdvals(X)  # s.shape = (min(N, K),)
    U, s, _Vh = torch.linalg.svd(X, full_matrices=False)  # U:(n,k), s:(k,)
    k = s.numel()

    # normalize singular values to sum to 1
    sv_norm = torch.linalg.norm(s, ord=1)
    probs = s / sv_norm.clamp_min(eps)  # (k,)
    entropy = -(probs * (probs.clamp_min(eps)).log()).sum()
    rankme = torch.exp(entropy)

    effective_rank = (s / (s.max() + eps)).sum()

    # ---- Added: per-sample "contribution probabilities" from ONE decomposition ----
    # Covariance eigvals for Sigma=(1/n)X^T X are lambda_k = s_k^2 / n
    lam = (s * s) / float(n)  # (k,)

    # "Energy contributed by sample i on direction k":
    # c_{i,k} = lambda_k * U_{i,k}^2
    # U is (n,k), so U^2 is (n,k); broadcast multiply by lam (k,)
    U2 = U * U  # (n,k)
    C = U2 * lam.unsqueeze(0)  # (n,k)

    # per-sample total energy
    spectral_energy = C.sum(dim=-1)  # (n,), 样本 i 在每个谱方向 k 的供能
    spectral_energy_probs = spectral_energy / spectral_energy.sum().clamp_min(eps)  # (n,)

    # per-sample distribution over directions, 方向分布，response 的“总贡献概率”
    # q_{i,k} = c_{i,k} / sum_t c_{i,t}
    q = C / spectral_energy.unsqueeze(-1).clamp_min(eps)  # (n,k)

    # 样本谱熵（coverage entropy），越大表示该样本覆盖更多谱方向（更“补充新方向”）。
    coverage_entropy = -(q * (q.clamp_min(eps)).log()).sum(dim=-1)  # (n,)
    # coverage_rank = coverage_entropy.exp()  # (n,)

    metrics = {
        "rankme": rankme,
        "effective_rank": effective_rank,
        "singular_values": s,
        # "singular_value_probs": probs,
        "spectral_energy": spectral_energy,
        # "spectral_energy_probs": spectral_energy_probs,
        "coverage_entropy": coverage_entropy,
    }

    return metrics


def compute_length_metrics(batch: DataProto, config, mode: str = ""):
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    if mode == "":
        avg_accs = sequence_score = batch.batch["token_level_scores"].sum(-1)
        sequence_acc = torch.where(sequence_score > 0, 1.0, 0.0)
        avg_accs = sequence_acc.reshape(-1, config.actor_rollout_ref.rollout.n).mean(dim=-1)

        hard_mask = avg_accs <= 0.25
        easy_mask = avg_accs >= 0.75
        medium_mask = (~hard_mask) & (~easy_mask)
        expanded_hard_mask = hard_mask.repeat_interleave(config.actor_rollout_ref.rollout.n)
        expanded_medium_mask = medium_mask.repeat_interleave(config.actor_rollout_ref.rollout.n)
        expanded_easy_mask = easy_mask.repeat_interleave(config.actor_rollout_ref.rollout.n)

    suffix = "/pos" if mode == "pos" else "/neg" if mode == "neg" else ""
    suffix = "_selected" if mode == "selected" else suffix
    metrics = {
        # response length
        f"response_length{suffix}/mean": _safe_mean(response_length),
        f"response_length{suffix}/max": _safe_max(response_length),
        f"response_length{suffix}/min": _safe_min(response_length, min_value=0.0),
        f"response_length{suffix}/clip_ratio": _safe_mean(torch.eq(response_length, max_response_length).float()),
        # prompt length
        f"prompt_length{suffix}/mean": _safe_mean(prompt_length),
        f"prompt_length{suffix}/max": _safe_max(prompt_length),
        f"prompt_length{suffix}/min": _safe_min(prompt_length, min_value=0.0),
        f"prompt_length{suffix}/clip_ratio": _safe_mean(torch.eq(prompt_length, max_prompt_length).float()),
    }
    if mode == "":
        metrics.update({
            "response_length/hard/mean": _safe_mean(response_length, expanded_hard_mask.bool()),
            "response_length/medium/mean": _safe_mean(response_length, expanded_medium_mask.bool()),
            "response_length/easy/mean": _safe_mean(response_length, expanded_easy_mask.bool()),
            "response_length/hard/clip_ratio": torch.mean(torch.eq(response_length[expanded_hard_mask.bool()], max_response_length).float()).detach().item(),
            "response_length/medium/clip_ratio": torch.mean(torch.eq(response_length[expanded_medium_mask.bool()], max_response_length).float()).detach().item(),
            "response_length/easy/clip_ratio": torch.mean(torch.eq(response_length[expanded_easy_mask.bool()], max_response_length).float()).detach().item(),
            "prompt_length/hard/mean": _safe_mean(prompt_length, expanded_hard_mask.bool()),
            "prompt_length/medium/mean": _safe_mean(prompt_length, expanded_medium_mask.bool()),
            "prompt_length/easy/mean": _safe_mean(prompt_length, expanded_easy_mask.bool()),
        })
    proxies = {
        "response_length": response_length,
    }
    return metrics, proxies


def compute_critic_metrics(batch: DataProto, use_critic: bool = True, mode: str = ""):
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
    """
    metrics, proxies = {}, {}

    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    response_mask = batch.batch["response_mask"].bool()

    response_info = _compute_response_info(batch)
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    suffix = "/pos" if mode == "pos" else "/neg" if mode == "neg" else ""
    suffix = "_selected" if mode == "selected" else suffix
    metrics.update({
        # score
        f"critic{suffix}/score/mean": score_mean,
        f"critic{suffix}/score/max": score_max,
        f"critic{suffix}/score/min": score_min,
        # reward
        f"critic{suffix}/rewards/mean": reward_mean,
        f"critic{suffix}/rewards/max": reward_max,
        f"critic{suffix}/rewards/min": reward_min,
        # adv
        f"critic{suffix}/advantages/mean": torch.mean(valid_adv).detach().item(),
        f"critic{suffix}/advantages/max": torch.max(valid_adv).detach().item(),
        f"critic{suffix}/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        f"critic{suffix}/returns/mean": torch.mean(valid_returns).detach().item(),
        f"critic{suffix}/returns/max": torch.max(valid_returns).detach().item(),
        f"critic{suffix}/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                f"critic{suffix}/values/mean": torch.mean(valid_values).detach().item(),
                f"critic{suffix}/values/max": torch.max(valid_values).detach().item(),
                f"critic{suffix}/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                f"critic{suffix}/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
    })
    return metrics, proxies


def compute_diversity_metrics(batch: DataProto, eps, mode: str = ""):
    metrics, proxies = {}, {}

    response_mask = batch.batch["response_mask"].bool()
    response_info = _compute_response_info(batch)
    response_length = response_info["response_length"]

    entropys = batch.batch["entropys"]
    valid_entropys = torch.masked_select(entropys, response_mask)
    resp_entropy = torch.sum(entropys * response_mask, dim=-1) / (response_length + eps)  # token-mean
    resp_length_mask = (response_length > 0).float()  # exclude fully masked sequences
    mean_resp_entropy = verl_F.masked_sum(resp_entropy, resp_length_mask) / resp_length_mask.sum()  # seq-mean

    old_log_probs = batch.batch["old_log_probs"]
    valid_old_log_probs = torch.masked_select(old_log_probs, response_mask)
    token_ppl = torch.exp(-valid_old_log_probs.mean())
    resp_ppl = torch.exp(torch.sum(-old_log_probs * response_mask, dim=-1) / (response_length + eps))  # token-mean
    mean_resp_ppl = verl_F.masked_sum(resp_ppl, resp_length_mask) / resp_length_mask.sum()  # seq-mean

    suffix = "/pos" if mode == "pos" else "/neg" if mode == "neg" else ""
    suffix = "_selected" if mode == "selected" else suffix
    metrics.update({
        f"diversity{suffix}/unique_n_gram_ratio": _safe_mean(batch.batch["diversity/unique_n_gram_ratio"]),
        f"diversity{suffix}/repetition_ratio": _safe_mean(batch.batch["diversity/repetition_ratio"]),
        f"diversity{suffix}/max_n_gram_cnt": _safe_mean(batch.batch["diversity/max_n_gram_cnt"]),
        f"diversity{suffix}/token_entropy/mean": _safe_mean(valid_entropys),
        f"diversity{suffix}/token_entropy/max": _safe_max(valid_entropys),
        f"diversity{suffix}/token_entropy/std": _safe_std(valid_entropys),
        f"diversity{suffix}/resp_entropy/mean": mean_resp_entropy.item(),
        f"diversity{suffix}/resp_entropy/max": _safe_max(resp_entropy, mask=resp_length_mask.bool()),
        f"diversity{suffix}/resp_entropy/std": _safe_std(resp_entropy, mask=resp_length_mask.bool()),
        f"diversity{suffix}/token_ppl/mean": token_ppl.item(),
        f"diversity{suffix}/resp_ppl/mean": mean_resp_ppl.item(),
        f"diversity{suffix}/resp_ppl/max": _safe_max(resp_ppl, mask=resp_length_mask.bool()),
        f"diversity{suffix}/resp_ppl/std": _safe_std(resp_ppl, mask=resp_length_mask.bool()),
    })
    if "collision_probs" in batch.batch.keys():
        collision_probs = batch.batch["collision_probs"]
        valid_collision_probs = torch.masked_select(collision_probs, response_mask)
        resp_collision_prob = torch.sum(collision_probs * response_mask, dim=-1) / (response_length + eps)  # token-mean
        mean_collision_prob = verl_F.masked_sum(resp_collision_prob, resp_length_mask) / resp_length_mask.sum()  # seq-mean

        collision_entropys = - torch.log(collision_probs + eps)
        valid_collision_entropys = torch.masked_select(collision_entropys, response_mask)
        resp_collision_entropy = torch.sum(collision_entropys * response_mask, dim=-1) / (response_length + eps)  # token-mean
        mean_collision_entropy = verl_F.masked_sum(resp_collision_entropy, resp_length_mask) / resp_length_mask.sum()  # seq-mean

        metrics.update({
            f"diversity{suffix}/token_coll_prob/mean": _safe_mean(valid_collision_probs),
            f"diversity{suffix}/token_coll_prob/max": _safe_max(valid_collision_probs),
            f"diversity{suffix}/token_coll_prob/std": _safe_std(valid_collision_probs),
            f"diversity{suffix}/resp_coll_prob/mean": mean_collision_prob.item(),
            f"diversity{suffix}/resp_coll_prob/max": _safe_max(resp_collision_prob, mask=resp_length_mask.bool()),
            f"diversity{suffix}/resp_coll_prob/std": _safe_std(resp_collision_prob, mask=resp_length_mask.bool()),
            f"diversity{suffix}/token_coll_entropy/mean": _safe_mean(valid_collision_entropys),
            f"diversity{suffix}/token_coll_entropy/max": _safe_max(valid_collision_entropys),
            f"diversity{suffix}/token_coll_entropy/std": _safe_std(valid_collision_entropys),
            f"diversity{suffix}/resp_coll_entropy/mean": mean_collision_entropy.item(),
            f"diversity{suffix}/resp_coll_entropy/max": _safe_max(resp_collision_entropy, mask=resp_length_mask.bool()),
            f"diversity{suffix}/resp_coll_entropy/std": _safe_std(resp_collision_entropy, mask=resp_length_mask.bool()),
        })

    if mode == "":
        proxies.update({
            "entropy": resp_entropy,
            "ppl": resp_ppl,
        })
    return metrics, proxies


def compute_difficulty_metrics(batch: DataProto, trainer, epoch, config, batch_len=0, mode: str = ""):
    metrics, proxies = {}, {}

    response_mask = batch.batch["response_mask"].bool()

    valid_resp = torch.where(masked_sum(response_mask, batch.batch["advantages"] != 0, axis=1) > 0, 1, 0)
    valid_token_cnt = masked_sum(response_mask.sum(-1), valid_resp).item()

    if mode == "" or mode == "selected":
        sequence_score = batch.batch["token_level_scores"].sum(-1)
        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
        sequence_acc = torch.where(sequence_score > 0, 1.0, 0.0)

        data_indices = torch.tensor(batch.non_tensor_batch["index"].astype(int))
        # check if data_indices is increasing and unique within consecutive n rollouts
        unique_data_indices = data_indices.reshape(-1, config.actor_rollout_ref.rollout.n)[:, 0]
        # assert torch.all(unique_data_indices[1:] > unique_data_indices[:-1]), f"unique_data_indices: {unique_data_indices} is not increasing"
        assert torch.all(unique_data_indices == data_indices.reshape(-1, config.actor_rollout_ref.rollout.n).float().mean(dim=-1).int()), f"unique_data_indices: {unique_data_indices} is not unique within consecutive n rollouts: {data_indices.reshape(-1, config.actor_rollout_ref.rollout.n).float().mean(dim=-1).int()}"

        # avg_accs = torch.zeros(len(unique_data_indices))
        # for i, data_index in enumerate(unique_data_indices):
        #     # assert data_index == int(batch.non_tensor_batch["index"][i * config.actor_rollout_ref.rollout.n])
        #     data_index_mask = data_indices == data_index
        #     avg_accs[i] = sequence_acc[data_index_mask].mean()  # Sum rewards for each sequence
        avg_accs = sequence_acc.reshape(-1, config.actor_rollout_ref.rollout.n).mean(dim=-1)

        if epoch >= 1:
            last_avg_accs = trainer.data_filter.get_prompt_data_at_key("acc", epoch - 1, unique_data_indices)
            print(f"last_epoch_accs: {avg_accs.shape}")
            delta_accs = avg_accs - last_avg_accs
            last_epoch_sampled = last_avg_accs != trainer.data_filter.c
            increased_accs_cnt = (delta_accs > 0 & last_epoch_sampled).sum().item()
            decreased_accs_cnt = (delta_accs < 0 & last_epoch_sampled).sum().item()
            unchanged_accs_cnt = (delta_accs == 0 & last_epoch_sampled).sum().item()
            last_epoch_not_sampled_cnt = (last_avg_accs == trainer.data_filter.c).sum().item()
            print(f"increased_accs_cnt: {increased_accs_cnt}, decreased_accs_cnt: {decreased_accs_cnt}, unchanged_accs_cnt: {unchanged_accs_cnt}, last_epoch_not_sampled_cnt: {last_epoch_not_sampled_cnt}")

        solve_none = (avg_accs == 0).sum().item()
        solve_all = (avg_accs == 1).sum().item()
        valid_prom_cnt = len(unique_data_indices) - solve_none - solve_all

        hard_mask = avg_accs <= 0.25
        easy_mask = avg_accs >= 0.75
        medium_mask = (~hard_mask) & (~easy_mask)
        expanded_hard_mask = hard_mask.repeat_interleave(config.actor_rollout_ref.rollout.n)
        expanded_medium_mask = medium_mask.repeat_interleave(config.actor_rollout_ref.rollout.n)
        expanded_easy_mask = easy_mask.repeat_interleave(config.actor_rollout_ref.rollout.n)

        hard_cnt = hard_mask.sum().item()
        medium_cnt = medium_mask.sum().item()
        easy_cnt = easy_mask.sum().item()

        print(f"valid prompt: {len(unique_data_indices) - solve_all - solve_none}, solve_none: {solve_none}, solve_all: {solve_all}")
        print(f"hard_cnt: {hard_cnt}, medium_cnt: {medium_cnt}, easy_cnt: {easy_cnt}")

        suffix = "/pos" if mode == "pos" else "/neg" if mode == "neg" else ""
        suffix = "_selected" if mode == "selected" else suffix
        metrics.update({
            # difficulty
            f"difficulty{suffix}/valid_resp_cnt": valid_resp.sum().item(),
            f"difficulty{suffix}/valid_resp_ratio": valid_resp.sum().item() / len(batch),
            f"difficulty{suffix}/valid_token_cnt": valid_token_cnt,
            f"difficulty{suffix}/valid_token_ratio": valid_token_cnt / response_mask.sum().item(),
            f"difficulty{suffix}/valid_prom_cnt": valid_prom_cnt,
            f"difficulty{suffix}/valid_prom_ratio": valid_prom_cnt / len(unique_data_indices),
            f"difficulty{suffix}/solve_none_prom_cnt": solve_none,
            f"difficulty{suffix}/solve_none_prom_ratio": solve_none / len(unique_data_indices),
            f"difficulty{suffix}/solve_all_prom_cnt": solve_all,
            f"difficulty{suffix}/solve_all_prom_ratio": solve_all / len(unique_data_indices),
            f"difficulty{suffix}/hard_cnt": hard_cnt,
            f"difficulty{suffix}/hard_ratio": hard_cnt / len(unique_data_indices),
            f"difficulty{suffix}/medium_cnt": medium_cnt,
            f"difficulty{suffix}/medium_ratio": medium_cnt / len(unique_data_indices),
            f"difficulty{suffix}/easy_cnt": easy_cnt,
            f"difficulty{suffix}/easy_ratio": easy_cnt / len(unique_data_indices),
            f"difficulty{suffix}/avg_acc": avg_accs.mean().item(),
        })
        if epoch >= 1:
            metrics.update({
                f"prompt_tracking{suffix}/increased_ratio": increased_accs_cnt / len(unique_data_indices),
                f"prompt_tracking{suffix}/decreased_ratio": decreased_accs_cnt / len(unique_data_indices),
                f"prompt_tracking{suffix}/unchanged_ratio": unchanged_accs_cnt / len(unique_data_indices),
                f"prompt_tracking{suffix}/last_epoch_not_sampled_ratio": last_epoch_not_sampled_cnt / len(unique_data_indices),
                f"prompt_tracking{suffix}/hard/increased_ratio": ((delta_accs > 0) & hard_mask).sum().item() / hard_mask.sum().item() if hard_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/hard/decreased_ratio": ((delta_accs < 0) & hard_mask).sum().item() / hard_mask.sum().item() if hard_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/hard/unchanged_ratio": ((delta_accs == 0) & hard_mask).sum().item() / hard_mask.sum().item() if hard_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/medium/increased_ratio": ((delta_accs > 0) & medium_mask).sum().item() / medium_mask.sum().item() if medium_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/medium/decreased_ratio": ((delta_accs < 0) & medium_mask).sum().item() / medium_mask.sum().item() if medium_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/medium/unchanged_ratio": ((delta_accs == 0) & medium_mask).sum().item() / medium_mask.sum().item() if medium_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/easy/increased_ratio": ((delta_accs > 0) & easy_mask).sum().item() / easy_mask.sum().item() if easy_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/easy/decreased_ratio": ((delta_accs < 0) & easy_mask).sum().item() / easy_mask.sum().item() if easy_mask.sum().item() > 0 else 0.0,
                f"prompt_tracking{suffix}/easy/unchanged_ratio": ((delta_accs == 0) & easy_mask).sum().item() / easy_mask.sum().item() if easy_mask.sum().item() > 0 else 0.0,
            })
        if mode == "":
            proxies.update({
                "acc": avg_accs.repeat_interleave(config.actor_rollout_ref.rollout.n),
                "reward": sequence_reward,
            })
    elif mode == "pos" or mode == "neg":
        suffix = "/pos" if mode == "pos" else "/neg"
        metrics.update({
            f"difficulty{suffix}/resp_ratio": len(batch) / batch_len,
            f"difficulty{suffix}/valid_resp_cnt": valid_resp.sum().item(),
            f"difficulty{suffix}/valid_resp_ratio": valid_resp.sum().item() / len(batch),
            f"difficulty{suffix}/valid_token_cnt": valid_token_cnt,
            f"difficulty{suffix}/valid_token_ratio": valid_token_cnt / response_mask.sum().item(),
        })
    return metrics, proxies


def compute_other_metrics(batch: DataProto, mode: str = ""):
    metrics, proxies = {}, {}

    suffix = "/pos" if mode == "pos" else "/neg" if mode == "neg" else ""
    suffix = "_selected" if mode == "selected" else suffix
    for key in batch.batch.keys():
        if key.startswith("token/"):
            if mode == "":
                metrics[key] = _safe_mean(batch.batch[key])
            else:
                key_left, key_right = key.split("/")
                metrics[f"{key_left}{suffix}/{key_right}"] = _safe_mean(batch.batch[key])
    return metrics, proxies


def _compute_ratio_seq_and_seq_geo(batch: DataProto, rollout_n: int):
    """从 batch 的 old_log_probs / rollout_log_probs 计算 train/infer ratio（sequence-level 与 seq-geo），与 rollout_corr 一致。"""
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return None, None
    response_mask = batch.batch["response_mask"].bool()
    if not response_mask.any():
        return None, None
    log_ratio = batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"]
    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    log_ratio_mean = verl_F.masked_mean(log_ratio, response_mask, axis=-1)
    rho_seq = torch.exp(log_ratio_sum)
    rho_seq_geo = torch.exp(log_ratio_mean)
    return rho_seq, rho_seq_geo


def process_data_filter(batch: DataProto, trainer, epoch, config, proxies, mode="", ratio_seq=None, ratio_seq_geo=None):
    data_indices = torch.tensor(batch.non_tensor_batch["index"].astype(int))
    rollout_n = config.actor_rollout_ref.rollout.n
    unique_data_indices = data_indices.reshape(-1, rollout_n)[:, 0]
    if mode == "selected":
        for i, data_index in enumerate(unique_data_indices):
            trainer.data_filter.add_prompt_data_at_key("selected", epoch, data_index, 1)
    else:
        for i, data_index in enumerate(unique_data_indices):
            data_index_mask = data_indices == data_index
            trainer.data_filter.add_prompt_data_at_key("step", epoch, data_index, trainer.global_steps)
            trainer.data_filter.add_prompt_data_at_key("acc", epoch, data_index, proxies.batch["acc"][i])
            trainer.data_filter.add_prompt_data_at_key("reward", epoch, data_index, proxies.batch["reward"][data_index_mask])
            trainer.data_filter.add_prompt_data_at_key("entropy", epoch, data_index, proxies.batch["entropy"][data_index_mask])
            trainer.data_filter.add_prompt_data_at_key("ppl", epoch, data_index, proxies.batch["ppl"][data_index_mask])
            trainer.data_filter.add_prompt_data_at_key("response_length", epoch, data_index, proxies.batch["response_length"][data_index_mask])
            if ratio_seq is not None and ratio_seq_geo is not None:
                trainer.data_filter.add_prompt_data_at_key("ratio_seq", epoch, data_index, ratio_seq[i * rollout_n:(i + 1) * rollout_n])
                trainer.data_filter.add_prompt_data_at_key("ratio_seq_geo", epoch, data_index, ratio_seq_geo[i * rollout_n:(i + 1) * rollout_n])


def compute_data_metrics(
    batch: DataProto, trainer, proxies_dict, curr_weights, epoch, use_critic: bool = True, config=None, batch_grad_direction=None,
    hidden_states_svd_metrics=None, selected_mask=None
):
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    diversity / uncertainty: entropy, ppl, n-gram, accuracy std
    gradient: grad norm, grad angle
    difficulty: difficulty
    reward: score, reward, advantage, return, value

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    eps = 1e-8

    mode, proxies = "", None

    diversity_metrics, diversity_proxies = compute_diversity_metrics(batch, eps, mode=mode)
    difficulty_metrics, difficulty_proxies = compute_difficulty_metrics(batch, trainer, epoch, config, mode=mode)

    critic_metrics, critic_proxies = compute_critic_metrics(batch, use_critic, mode=mode)
    length_metrics, length_proxies = compute_length_metrics(batch, config, mode=mode)
    other_metrics, other_proxies = compute_other_metrics(batch, mode=mode)

    metrics = {
        **diversity_metrics,
        **difficulty_metrics,
        **critic_metrics,
        **length_metrics,
        **other_metrics,
    }
    if selected_mask is None:
        proxies = DataProto.from_dict(tensors={
            **diversity_proxies,
            **difficulty_proxies,
            **critic_proxies,
            **length_proxies,
            **other_proxies,
        })

    # train/infer ratio（sequence-level / seq-geo）写入 data_filter，用于轨迹层面 mismatch 统计
    ratio_seq, ratio_seq_geo = None, None
    if mode == "" and config is not None:
        ratio_seq, ratio_seq_geo = _compute_ratio_seq_and_seq_geo(batch, config.actor_rollout_ref.rollout.n)
    process_data_filter(batch, trainer, epoch, config, proxies, mode=mode, ratio_seq=ratio_seq, ratio_seq_geo=ratio_seq_geo)

    # GRPO 同一 prompt 的 n 条轨迹：各 prompt 间 train/infer ratio 的 trajectory-level mismatch 均值与方差（ratio_seq / ratio_seq_geo）
    if (
        mode == ""
        and config is not None
        and trainer is not None
        and getattr(trainer, "data_filter", None) is not None
        and ratio_seq is not None
        and ratio_seq_geo is not None
    ):
        rollout_n = config.actor_rollout_ref.rollout.n
        if rollout_n > 1:
            data_indices = torch.tensor(batch.non_tensor_batch["index"].astype(int))
            unique_data_indices = data_indices.reshape(-1, rollout_n)[:, 0]
            mismatch_stats = trainer.data_filter.compute_rollout_mismatch_stats(epoch, unique_data_indices)
            metrics.update(mismatch_stats)

    if selected_mask is not None:
        return metrics, proxies

    sequence_score = batch.batch["token_level_scores"].sum(-1)

    for mode in ["pos", "neg"]:
        mask = sequence_score > 0 if mode == "pos" else sequence_score <= 0
        mode_batch = batch[mask]
        if mode_batch.batch['responses'].shape[0] != 0:
            mode_metrics = {}
            mode_diversity_metrics, _ = compute_diversity_metrics(mode_batch, eps, mode=mode)
            mode_difficulty_metrics, _ = compute_difficulty_metrics(mode_batch, trainer, epoch, config, batch_len=len(batch), mode=mode)
            mode_critic_metrics, _ = compute_critic_metrics(mode_batch, use_critic, mode=mode)
            mode_length_metrics, _ = compute_length_metrics(mode_batch, config, mode=mode)
            mode_other_metrics, _ = compute_other_metrics(mode_batch, mode=mode)
            mode_metrics.update(
                **mode_diversity_metrics,
                **mode_difficulty_metrics,
                **mode_critic_metrics,
                **mode_length_metrics,
                **mode_other_metrics,
            )
            metrics.update(mode_metrics)

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics, proxies, hidden_states_svd_metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    data_std = {}
    data_response_length = {}
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        all_accs = []
        all_response_lengths = []
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                # if var_name != "acc":
                #     continue
                if var_name == "response_length":
                    all_response_lengths.extend(var_vals)
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"avg@{n_resps}"] = np.mean(var_vals)
                metric[f"pass@{n_resps}"] = np.max(var_vals)
                if var_name == "acc":
                    all_accs.append(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    # n = 2
                    # while n < n_resps:
                    #     ns.append(n)
                    #     n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric
        all_accs = np.stack(all_accs)
        all_accs_per_eval = np.mean(all_accs, axis=0)
        data_std[data_source] = np.std(all_accs_per_eval)
        data_response_length[data_source] = np.mean(all_response_lengths)

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                if var_name == "acc" and metric_name.startswith("std@"):
                    data_src2var2metric2val[data_source][var_name][metric_name] = data_std[data_source]
                else:
                    data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)
        data_src2var2metric2val[data_source]["response_length"]["mean"] = data_response_length[data_source]

    return data_src2var2metric2val
