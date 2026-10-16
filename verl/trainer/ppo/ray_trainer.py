# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.metrics.data_filter import DataFilter


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    global_step: Optional[int] = None,
    grpo_lagrangian_operator=None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.
        global_step (int, optional): Current training step; used with config.penalty_cost_start_step for
            staged cost (cost applied only when global_step >= penalty_cost_start_step, shared with penalty).

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

        # CMDP: compute adv_cost from IS ratio and subtract from adv_reward
        adv_cost_cfg = config.get("adv_cost", None) if config is not None else None
        penalty_cost_start_step = config.get("penalty_cost_start_step", None) if config is not None else None
        cost_enabled_by_step = (
            penalty_cost_start_step is None
            or global_step is None
            or global_step >= penalty_cost_start_step
        )
        if (
            adv_cost_cfg
            and adv_cost_cfg.get("enable", False)
            and cost_enabled_by_step
            and "rollout_log_probs" in data.batch
            and "old_log_probs" in data.batch
        ):
            import verl.utils.torch_functional as verl_F

            cost_type = adv_cost_cfg.get("type", "seq_geo")
            cost_norm_by_std = adv_cost_cfg.get("norm_adv_by_std", True)
            cost_coef = adv_cost_cfg.get("coef", 1.0)
            clip_penalty_low = config.get("clip_penalty_low", None) if config is not None else None
            clip_penalty_high = config.get("clip_penalty_high", None) if config is not None else None
            old_lp = data.batch["old_log_probs"]
            rollout_lp = data.batch["rollout_log_probs"]
            log_ratio = old_lp - rollout_lp

            # Per-sequence raw cost source only (same definitions as rollout_corr metrics / reward penalty)
            if cost_type == "seq":
                log_agg = verl_F.masked_sum(log_ratio, grpo_calculation_mask, axis=-1)
                raw_per_seq = torch.exp(log_agg)
            elif cost_type == "seq_geo":
                log_agg = verl_F.masked_mean(log_ratio, grpo_calculation_mask, axis=-1)
                raw_per_seq = torch.exp(log_agg)
            elif cost_type == "log_ppl_abs_diff":
                mean_log_prob_rollout = verl_F.masked_mean(rollout_lp, grpo_calculation_mask, axis=-1)
                mean_log_prob_training = verl_F.masked_mean(old_lp, grpo_calculation_mask, axis=-1)
                log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
                raw_per_seq = log_ppl_diff.abs()
            elif cost_type == "seq_kl":
                log_prob_diff = rollout_lp - old_lp
                seq_kl = verl_F.masked_sum(log_prob_diff, grpo_calculation_mask, axis=-1)
                raw_per_seq = seq_kl.abs()
            elif cost_type == "seq_forward_kl":
                seq_log_ratio = verl_F.masked_sum(log_ratio, grpo_calculation_mask, axis=-1)
                importance_weights_seq = torch.exp(seq_log_ratio)
                raw_per_seq = importance_weights_seq * seq_log_ratio
            elif cost_type == "seq_jeffreys":
                # Identical to metrics: seq_jeffreys = exp(seq_log_ratio)*seq_log_ratio + seq_log_prob_diff
                log_prob_diff = rollout_lp - old_lp
                seq_log_prob_diff = verl_F.masked_sum(log_prob_diff, grpo_calculation_mask, axis=-1)
                seq_log_ratio = verl_F.masked_sum(log_ratio, grpo_calculation_mask, axis=-1)
                importance_weights_seq = torch.exp(seq_log_ratio)
                seq_jeffreys = importance_weights_seq * seq_log_ratio + seq_log_prob_diff
                raw_per_seq = seq_jeffreys 
            elif cost_type == "diff_seq_sum_add":
                # Identical to metrics: diff_token = π_train - π_rollout, diff_seq_sum_add = masked_sum(diff_token)
                pi_train_token = torch.exp(old_lp)
                pi_rollout_token = torch.exp(rollout_lp)
                diff_token = pi_train_token - pi_rollout_token
                diff_seq_sum_add = verl_F.masked_sum(diff_token, grpo_calculation_mask, axis=-1)
                raw_per_seq = diff_seq_sum_add.abs()
            else:
                raise ValueError(
                    f"adv_cost type must be seq, seq_geo, log_ppl_abs_diff, seq_kl, seq_forward_kl, seq_jeffreys or diff_seq_sum_add, got {cost_type!r}"
                )
            # Bilateral clip: zero out when low < raw < high. None for low/high means no bound (-inf / +inf).
            if clip_penalty_low is not None or clip_penalty_high is not None:
                effective_low = clip_penalty_low if clip_penalty_low is not None else float("-inf")
                effective_high = clip_penalty_high if clip_penalty_high is not None else float("inf")
                effective = raw_per_seq.clone()
                effective[(raw_per_seq > effective_low) & (raw_per_seq < effective_high)] = 0
                raw_per_seq = effective
            # cost_per_seq = raw_per_seq / (1.0 + raw_per_seq)

            # Pack cost into token_level shape: place scalar at last valid token (like reward)
            token_level_costs = torch.zeros_like(data.batch["token_level_rewards"])
            last_idx = grpo_calculation_mask.float().sum(dim=-1).long() - 1
            last_idx = last_idx.clamp(min=0)
            row_idx = torch.arange(raw_per_seq.shape[0], device=raw_per_seq.device)
            token_level_costs[row_idx, last_idx] = raw_per_seq.to(token_level_costs.dtype)

            adv_cost, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_costs,
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=cost_norm_by_std,
            )

            # adv_cost = adv_cost / (1.0 + adv_cost)
            adv_reward = advantages  # save before subtraction
            adv_cost = torch.sigmoid(adv_cost)
            advantages = adv_reward - cost_coef * adv_cost

            # Metrics: masked_select on response_mask (same aggregation as critic/advantages/)
            mask_bool = grpo_calculation_mask.bool()
            valid_adv_reward = torch.masked_select(adv_reward, mask_bool)
            valid_adv_cost = torch.masked_select(adv_cost, mask_bool)

            data.meta_info["adv_cost_metrics"] = {
                "critic/adv_reward/mean": valid_adv_reward.mean().item(),
                "critic/adv_reward/max": valid_adv_reward.max().item(),
                "critic/adv_reward/min": valid_adv_reward.min().item(),
                "critic/adv_cost/mean": valid_adv_cost.mean().item(),
                "critic/adv_cost/max": valid_adv_cost.max().item(),
                "critic/adv_cost/min": valid_adv_cost.min().item(),
            }

        lagrangian_cfg = config.get("grpo_lagrangian", None) if config is not None else None
        lagrangian_apply_on = lagrangian_cfg.get("apply_on", "reward") if lagrangian_cfg is not None else "reward"
        if (
            grpo_lagrangian_operator is not None
            and lagrangian_cfg is not None
            and lagrangian_cfg.get("enable", False)
            and lagrangian_apply_on == "adv"
            and "rollout_log_probs" in data.batch
        ):
            penalty_cost_start_step = config.get("penalty_cost_start_step", None) if config is not None else None
            clip_penalty_low = config.get("clip_penalty_low", None) if config is not None else None
            clip_penalty_high = config.get("clip_penalty_high", None) if config is not None else None
            normalize_mode = config.get("normalize_mode", None) if config is not None else None
            normalize_tanh_k = config.get("normalize_tanh_k", None) if config is not None else None
            penalty_truncated = config.get("penalty_truncated", None) if config is not None else None
            advantages, lagrangian_metrics = grpo_lagrangian_operator.apply_to_advantages(
                batch=data,
                advantages=advantages,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                penalty_cost_start_step=penalty_cost_start_step,
                global_step=global_step,
                clip_penalty_low=clip_penalty_low,
                clip_penalty_high=clip_penalty_high,
                normalize_mode=normalize_mode,
                normalize_tanh_k=normalize_tanh_k,
                penalty_truncated=penalty_truncated,
            )
            data.meta_info["grpo_lagrangian_metrics"] = lagrangian_metrics

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.post_init()

        from verl.trainer.ppo.grpo_lagrangian import (
            GRPOLagrangianOperator,
            grpo_lagrangian_config_from_algorithm,
        )

        self._grpo_lagrangian = None
        _gl_cfg = grpo_lagrangian_config_from_algorithm(self.config.algorithm)
        if _gl_cfg:
            self._grpo_lagrangian = GRPOLagrangianOperator(_gl_cfg)

    def post_init(self):
        self.data_filter = DataFilter(self.train_dataset, self.config)
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_steps = [[0, 0], [0, 0], [0, 0]]

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, **kwargs):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(scores)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        for key in list(base_data.keys()):
            if base_data[key] is None:
                base_data.pop(key)
        for key in list(kwargs.keys()):
            if kwargs[key] is not None:
                base_data[key] = kwargs[key]

        if reward_extra_infos_dict is not None:
            for k, v in reward_extra_infos_dict.items():
                if k in ["pred"] and len(v) == n:
                    base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items() if not isinstance(v, torch.Tensor)}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            # inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            # outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            valid_response_length = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:].sum(-1).tolist()
            valid_prompt_length = batch.batch["attention_mask"][:, :-batch.batch["responses"].shape[-1]].sum(-1).tolist()
            pred = batch.non_tensor_batch["pred"]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=None,
                dump_path=rollout_data_dir,
                pred=pred,
                prompt_length=valid_prompt_length,
                response_length=valid_response_length,
            )

    def _save_all_rollout_batch_tensors(
        self, batch: DataProto, timing_raw: dict, rollout_data_dir: str, actor_output: Optional[DataProto] = None, extra_tensors=None
    ):
        """Save per-batch rollout tensors for offline analysis when explicitly enabled."""
        with marked_timer("dump_rollout_batch_tensors", timing_raw, color="green"):
            tensor_dump_path = os.path.join(rollout_data_dir, f"{self.global_steps}.pt")

            def _cpu_tensor_or_none(key: str):
                value = batch.batch.get(key, None)
                if value is None:
                    return None
                if torch.is_tensor(value):
                    return value.detach().cpu()
                return value

            payload = {
                "step": self.global_steps,
                "advantages": _cpu_tensor_or_none("advantages"),
                "old_log_probs": _cpu_tensor_or_none("old_log_probs"),
                "rollout_log_probs": _cpu_tensor_or_none("rollout_log_probs"),
                "entropys": _cpu_tensor_or_none("entropys"),
                "penalty": _cpu_tensor_or_none("rollout_penalty"),
                "prompt_token_ids": _cpu_tensor_or_none("prompts"),
                "response_token_ids": _cpu_tensor_or_none("responses"),
                "response_mask": _cpu_tensor_or_none("response_mask"),
            }
            if actor_output is not None and actor_output.batch is not None:
                for key, value in actor_output.batch.items():
                    payload[key] = value.detach().cpu() if torch.is_tensor(value) else value
            if extra_tensors:
                for key, value in extra_tensors.items():
                    payload[key] = value.detach().cpu() if torch.is_tensor(value) else value
            torch.save(payload, tensor_dump_path)
            print(f"Dumped rollout batch tensors to {tensor_dump_path}")

    @staticmethod
    def _prefix_prob_shift_metrics(metric_dict: dict[str, float], minibatch_idx: int) -> dict[str, float]:
        prefixed_metrics: dict[str, float] = {}
        minibatch_prefix = f"prob_shift_mb{minibatch_idx + 1}/"
        for key, value in metric_dict.items():
            assert key.startswith("prob_shift/"), key
            prefixed_metrics[f"{minibatch_prefix}{key.removeprefix('prob_shift/')}"] = value
        return prefixed_metrics

    def _get_actor_minibatch_row_ranges(self, batch_size: int) -> list[tuple[int, int]]:
        ppo_mini_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        if ppo_mini_batch_size <= 0:
            raise ValueError(f"actor.ppo_mini_batch_size must be positive, got {ppo_mini_batch_size}")
        return [(start, min(start + ppo_mini_batch_size, batch_size)) for start in range(0, batch_size, ppo_mini_batch_size)]

    def _compute_post_update_actor_outputs(
        self, batch: DataProto, timing_raw: dict, save_all_rollout_data: bool, compute_prob_shift: bool, actor_output: Optional[DataProto] = None,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        # save_all_rollout_data dumps the tensors already produced during rollout/actor update.
        # Recomputing post-update log probs is only needed for prob-shift diagnostics.
        if not compute_prob_shift:
            return {}, {}

        batch_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in batch.non_tensor_batch.keys() else []

        with marked_timer("compute_post_update_log_prob", timing_raw, color="blue"):
            eval_batch = batch.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys, deepcopy=True)
            post_update_output = self.actor_rollout_wg.compute_log_prob(eval_batch)

        new_log_probs = post_update_output.batch["old_log_probs"].detach().cpu()
        extra_tensors: dict[str, torch.Tensor] = {}
        metrics: dict[str, float] = {}

        response_minibatch_idx = None
        if actor_output is not None and actor_output.batch is not None:
            response_minibatch_idx = actor_output.batch.get("response_minibatch_idx", None)
        if response_minibatch_idx is not None:
            response_minibatch_idx = response_minibatch_idx.detach().cpu().long()
        else:
            row_ranges = self._get_actor_minibatch_row_ranges(batch.batch["old_log_probs"].shape[0])
            response_minibatch_idx = torch.empty((batch.batch["old_log_probs"].shape[0],), dtype=torch.long)
            for batch_idx, (start_idx, end_idx) in enumerate(row_ranges):
                response_minibatch_idx[start_idx:end_idx] = batch_idx + 1

        if response_minibatch_idx.shape != (batch.batch["old_log_probs"].shape[0],):
            raise ValueError("response_minibatch_idx shape mismatch: expected {(batch.batch['old_log_probs'].shape[0],)}, got {tuple(response_minibatch_idx.shape)}")

        if save_all_rollout_data:
            extra_tensors["new_log_probs"] = new_log_probs
            extra_tensors["response_minibatch_idx"] = response_minibatch_idx

        if compute_prob_shift:
            from verl.trainer.ppo.rollout_corr_helper import (
                compute_prob_shift_by_old_prob_and_adv_sign_metrics,
                compute_prob_shift_by_old_prob_bucket_metrics,
            )

            old_log_probs = batch.batch["old_log_probs"].detach().cpu()
            response_mask = batch.batch["response_mask"].detach().cpu()
            advantages = batch.batch["advantages"].detach().cpu()

            num_minibatches = int(response_minibatch_idx.max().item()) if response_minibatch_idx.numel() > 0 else 0
            for minibatch_idx in range(1, num_minibatches + 1):
                row_mask = response_minibatch_idx == minibatch_idx
                per_minibatch_metrics = compute_prob_shift_by_old_prob_bucket_metrics(
                    old_log_prob=old_log_probs[row_mask],
                    new_log_prob=new_log_probs[row_mask],
                    response_mask=response_mask[row_mask],
                )
                per_minibatch_metrics.update(
                    compute_prob_shift_by_old_prob_and_adv_sign_metrics(
                        old_log_prob=old_log_probs[row_mask],
                        new_log_prob=new_log_probs[row_mask],
                        advantages=advantages[row_mask],
                        response_mask=response_mask[row_mask],
                    )
                )
                metrics.update(self._prefix_prob_shift_metrics(per_minibatch_metrics, minibatch_idx - 1))

            metrics.update(
                compute_prob_shift_by_old_prob_bucket_metrics(
                    old_log_prob=old_log_probs,
                    new_log_prob=new_log_probs,
                    response_mask=response_mask,
                )
            )
            metrics.update(
                compute_prob_shift_by_old_prob_and_adv_sign_metrics(
                    old_log_prob=old_log_probs,
                    new_log_prob=new_log_probs,
                    advantages=advantages,
                    response_mask=response_mask,
                )
            )

        return metrics, extra_tensors

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _maybe_log_extra_reward_metrics(self, reward_result, acc, metrics: dict):
        tensors = {}
        if "thinking_tokens_info" in reward_result:
            thinking_tokens_infos_dict = reward_result["thinking_tokens_info"]
            for key_info in list(thinking_tokens_infos_dict.keys()):
                lst = thinking_tokens_infos_dict[key_info]
                tensors[key_info] = torch.tensor(lst, dtype=torch.float32)
                # key_left, key_right = key_info.split("/")
                # assert len(lst) == 0 or len(lst) == len(acc), f"{key_info}: {len(lst)=}, {len(acc)=}"
                # for value, score in zip(lst, acc):
                #     if score > 0:
                #         thinking_tokens_infos_dict[f"{key_left}/pos/{key_right}"].append(value)
                #     else:
                #         thinking_tokens_infos_dict[f"{key_left}/neg/{key_right}"].append(value)

            # for key_info, lst in thinking_tokens_infos_dict.items():
            #     metrics[key_info] = sum(lst) / len(lst)

        if "repetition_info" in reward_result:
            repetition_infos_dict = reward_result["repetition_info"]
            for key_info in list(repetition_infos_dict.keys()):
                lst = repetition_infos_dict[key_info]
                tensors[key_info] = torch.tensor(lst, dtype=torch.float32)
                # key_left, key_right = key_info.split("/")
                # assert len(lst) == 0 or len(lst) == len(acc), f"{key_info}: {len(lst)=}, {len(acc)=}"
                # for value, score in zip(lst, acc):
                #     if score > 0:
                #         repetition_infos_dict[f"{key_left}/pos/{key_right}"].append(value)
                #     else:
                #         repetition_infos_dict[f"{key_left}/neg/{key_right}"].append(value)

            # for key_info, lst in repetition_infos_dict.items():
            #     metrics[key_info] = sum(lst) / len(lst)

        batch = DataProto.from_dict(tensors=tensors)

        return batch

    def _get_gen_batch(self, batch: DataProto, keys_to_keep=[]) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        if keys_to_keep and "data_source" in keys_to_keep:
            reward_model_keys.remove("data_source")

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        thinking_tokens_infos_dict: dict[str, list] = defaultdict(list)
        repetition_infos_dict: dict[str, list] = defaultdict(list)
        response_length_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            index = [item.non_tensor_batch["index"] for item in test_batch]

            val_only = self.config.trainer.get("val_only", None)
            test_gen_batch = self._get_gen_batch(test_batch, keys_to_keep=["data_source"])
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "use_tqdm": True if val_only else False,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            if self.config.actor_rollout_ref.rollout.engine_kwargs.get("balance_val_batch", False):
                print(f"Using balance batch for validation")
                test_global_idx, test_reverse_global_idx = self._balance_gen_batch(test_gen_batch, mode="val")
            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            if self.config.actor_rollout_ref.rollout.engine_kwargs.get("balance_val_batch", False):
                test_output_gen_batch.reorder(test_reverse_global_idx)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            valid_response_length = test_output_gen_batch.batch["attention_mask"][:, -output_ids.shape[-1]:].sum(-1).tolist()
            output_ids_rmpad = [ids.tolist()[:valid_response_length[i]] for i, ids in enumerate(output_ids)]
            sample_outputs.extend(output_texts)
            valid_prompt_length = test_output_gen_batch.batch["attention_mask"][:, :-output_ids.shape[-1]].sum(-1).tolist()
            input_ids_rmpad = [ids.tolist()[-valid_prompt_length[i]:] for i, ids in enumerate(input_ids)]

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            if "thinking_tokens_info" in result:
                for key, lst in result["thinking_tokens_info"].items():
                    thinking_tokens_infos_dict[key].extend(lst)

            if "repetition_info" in result:
                for key, lst in result["repetition_info"].items():
                    repetition_infos_dict[key].extend(lst)

            if "response_length_info" in result:
                for key, lst in result["response_length_info"].items():
                    response_length_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            if "filter" in self.config.trainer.get("validation_data_dir", None):
                val_data_dir = self.config.trainer.get("validation_data_dir", None).replace("val", "val_results")
                self._dump_generations(
                    inputs=None,
                    outputs=None,
                    gts=ground_truths,
                    scores=scores,
                    reward_extra_infos_dict={"pred": reward_extra_infos_dict["pred"][-len(scores):]},
                    dump_path=val_data_dir,
                    prompt_length=valid_prompt_length,
                    response_length=valid_response_length,
                    index=index,
                )
                self.global_steps += 1

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_only:
            import datetime
            curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            val_ckpt = str(self.config.trainer.get("val_ckpt", "0"))
            val_data_dir = val_data_dir.replace("val", "val_results")
            balance_val_batch = self.config.actor_rollout_ref.rollout.engine_kwargs.get("balance_val_batch", False)
            val_data_dir = os.path.join(val_data_dir, f"{val_ckpt}_{balance_val_batch}_{self.actor_rollout_wg.world_size}_{curr_time}")

        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                input_ids=input_ids_rmpad,
                prompt_length=valid_prompt_length,
                output_ids=output_ids_rmpad,
                response_length=valid_response_length,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        for key_info in list(thinking_tokens_infos_dict.keys()):
            lst = thinking_tokens_infos_dict[key_info]
            key_left, key_right = key_info.split("/")
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            for value, score in zip(lst, sample_scores):
                if score > 0:
                    thinking_tokens_infos_dict[f"{key_left}/pos/{key_right}"].append(value)
                else:
                    thinking_tokens_infos_dict[f"{key_left}/neg/{key_right}"].append(value)

        for key_info in list(repetition_infos_dict.keys()):
            lst = repetition_infos_dict[key_info]
            key_left, key_right = key_info.split("/")
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            for value, score in zip(lst, sample_scores):
                if score > 0:
                    repetition_infos_dict[f"{key_left}/pos/{key_right}"].append(value)
                else:
                    repetition_infos_dict[f"{key_left}/neg/{key_right}"].append(value)

        for key_info in list(response_length_infos_dict.keys()):
            lst = response_length_infos_dict[key_info]
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            for value, score in zip(lst, sample_scores):
                if score > 0:
                    response_length_infos_dict[key_info + "/pos"].append(value)
                else:
                    response_length_infos_dict[key_info + "/neg"].append(value)

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        all_avg_lst = []
        all_response_length_mean_lst = []
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if var_name == "response_length":
                    continue
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["avg", "mean", "pass", "maj", "std", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                        if metric_name.startswith("avg") and f"@{n_max}" in metric_name:
                            all_avg_lst.append(metric_val)
                        if "maj@" in metric_name and "std" in metric_name:
                            continue
                        if "best" in metric_name and "std" in metric_name:
                            continue
                    else:
                        metric_sec = "val-aux"
                        continue

                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
            all_response_length_mean_lst.append(data_src2var2metric2val[data_source]["response_length"]["mean"])
            metric_dict[f"val_response_length/{data_source}/mean"] = data_src2var2metric2val[data_source]["response_length"]["mean"]
        metric_dict[f"val_response_length/all/mean"] = sum(all_response_length_mean_lst) / len(all_response_length_mean_lst)

        metric_dict["val-core/all/acc/avg"] = sum(all_avg_lst) / len(all_avg_lst)
        self.best_steps = sorted(self.best_steps, key=lambda x: x[1], reverse=True)
        if metric_dict["val-core/all/acc/avg"] > self.best_steps[-1][1]:
            self.best_steps[-1][0] = self.global_steps
            self.best_steps[-1][1] = metric_dict["val-core/all/acc/avg"]
            print(f"New best steps: {self.best_steps}")
            self.best_steps = sorted(self.best_steps, key=lambda x: x[1], reverse=True)

        for key_info, lst in thinking_tokens_infos_dict.items():
            metric_dict[f"val_{key_info}"] = sum(lst) / len(lst)

        for key_info, lst in repetition_infos_dict.items():
            metric_dict[f"val_{key_info}"] = sum(lst) / len(lst)

        for key_info, lst in response_length_infos_dict.items():
            metric_dict[f"val_{key_info}/max"] = max(lst)
            metric_dict[f"val_{key_info}/mean"] = sum(lst) / len(lst)
            metric_dict[f"val_{key_info}/min"] = min(lst)
            metric_dict[f"val_{key_info}/clip_ratio"] = sum([_ == output_ids.shape[1] for _ in lst]) / len(lst)

        if val_data_dir:
            with open(os.path.join(val_data_dir, f"metrics_{self.global_steps}.json"), "w", encoding="utf-8") as f:
                json.dump(metric_dict, f, ensure_ascii=False, indent=4)

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        should_save_hf_model = False
        # if self.global_steps % 100 == 0:
        #     should_save_hf_model = True
        if self.config.trainer.get("save_ckpt", False):
            if self.global_steps >= self.total_training_steps:
                should_save_hf_model = True
            if self.best_steps is not None and len(self.best_steps) > 0 and (self.global_steps == self.best_steps[0][0] or self.global_steps == self.best_steps[1][0] or self.global_steps == self.best_steps[2][0]):
                should_save_hf_model = True
            if self.config.trainer.save_freq == 1:
                should_save_hf_model = True
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep, should_save_hf_model=should_save_hf_model, best_steps=self.best_steps)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        try:
            # save data filter self.data_filter
            local_data_filter = os.path.join(local_global_step_folder, "data_filter.pt")
            if self.data_filter is not None:
                self.data_filter.save(local_data_filter)
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save the current epoch to the checkpoint
            local_current_epoch = os.path.join(self.config.trainer.default_local_dir, "current_epoch.txt")
            with open(local_current_epoch, "w") as f:
                f.write(str(self.current_epoch))
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.best_steps
            local_best_steps = os.path.join(self.config.trainer.default_local_dir, "best_steps.pt")
            self.best_steps = sorted(self.best_steps, key=lambda x: x[1], reverse=True)
            torch.save(self.best_steps, local_best_steps)
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            if getattr(self, "_grpo_lagrangian", None) is not None:
                lag_path = os.path.join(local_global_step_folder, "grpo_lagrangian.pt")
                self._grpo_lagrangian.save(lag_path)
        except Exception as e:
            print(f"Saving grpo_lagrangian state error: {e}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        try:
            # load data filter
            local_data_filter = os.path.join(global_step_folder, "data_filter.pt")
            if os.path.exists(local_data_filter):  # TODO (lrz): check
                self.data_filter.load(local_data_filter)
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load the current epoch to the checkpoint
            local_current_epoch = os.path.join(self.config.trainer.default_local_dir, "current_epoch.txt")
            if os.path.exists(local_current_epoch):
                with open(local_current_epoch, "r") as f:
                    self.start_epoch = int(f.read())
                print(f"Loaded current epoch: {self.start_epoch}")
            else:
                print(f"Warning: No current epoch found at {local_current_epoch}, set epoch = 0")
                self.start_epoch = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load best_steps
            local_best_steps = os.path.join(self.config.trainer.default_local_dir, "best_steps.pt")
            if os.path.exists(local_best_steps):  # TODO (lrz): check
                obj = torch.load(local_best_steps, weights_only=False)
                self.best_steps = obj
                print(f"Loaded best_steps: {self.best_steps}")
            else:
                print(f"Warning: No best_steps found at {local_best_steps}, set best_steps = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            lag_path = os.path.join(global_step_folder, "grpo_lagrangian.pt")
            if getattr(self, "_grpo_lagrangian", None) is not None and os.path.isfile(lag_path):
                payload = torch.load(lag_path, map_location="cpu", weights_only=False)
                self._grpo_lagrangian.load_state_dict(payload)
                print(f"Loaded GRPO-Lagrangian state from {lag_path}")
        except Exception as e:
            print(f"Loading grpo_lagrangian state error: {e}")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        reverse_global_idx = torch.argsort(global_idx)
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
        return global_idx, reverse_global_idx

    def _balance_gen_batch(self, batch: DataProto, mode):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        world_size = self.actor_rollout_wg.world_size
        # 初始化每张卡：当前负载（样本数）和索引列表
        gpu_loads = [0] * world_size
        global_partition_lst = [[] for _ in range(world_size)]

        curr = 0
        n_repeats = 0
        q_idx = 0
        if mode == "val":
            assert "data_source" in batch.non_tensor_batch.keys(), f"data_source key must be provided for test set"
            # 先把所有“题目组”拆成 (start_idx, size) 形式
            group_ranges: list[tuple[int, int, int]] = []
            last_key = f'{batch.non_tensor_batch["data_source"][0]}__{batch.non_tensor_batch["index"][0]}'
            for i in range(len(batch)):
                assert batch.non_tensor_batch["data_source"][i], f'data_source: {batch.non_tensor_batch["data_source"][i]} must be provided for test set'
                key = f'{batch.non_tensor_batch["data_source"][i]}__{batch.non_tensor_batch["index"][i]}'
                if key != last_key:
                    group_ranges.append((curr, n_repeats, q_idx))
                    curr += n_repeats
                    last_key = key
                    n_repeats = 0
                    q_idx += 1
                n_repeats += 1
            group_ranges.append((curr, n_repeats, q_idx))

            # 按组大小降序排列（把重复次数多的题先分配）
            group_ranges.sort(key=lambda x: (-x[1], x[2]), reverse=False)

            # 贪心装箱：对每个题目组，分配到当前负载最小的那张卡
            for start_idx, size, q_idx in group_ranges:
                tgt = min(range(world_size), key=lambda i: gpu_loads[i])  # 找到最空闲的 GPU
                global_partition_lst[tgt].extend(range(start_idx, start_idx + size))  # 分配 [start_idx, start_idx+size) 这段连续样本
                gpu_loads[tgt] += size

            global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
            batch.reorder(global_idx)
            reverse_global_idx = torch.argsort(global_idx)

            return global_idx, reverse_global_idx

        elif mode == "gen":
            assert "index" in batch.non_tensor_batch.keys() and "is_grpo" in batch.non_tensor_batch.keys(), f"index and is_grpo key must be provided for gen set"
            group_ranges: list[tuple[int, int, int]] = []
            last_key = f'{batch.non_tensor_batch["is_grpo"][0]}__{batch.non_tensor_batch["index"][0]}'
            for i in range(len(batch)):
                key = f'{batch.non_tensor_batch["is_grpo"][i]}__{batch.non_tensor_batch["index"][i]}'
                if key != last_key:
                    group_ranges.append((curr, int(batch.non_tensor_batch["is_grpo"][i - 1]), n_repeats))
                    curr += n_repeats
                    last_key = key
                    n_repeats = 0
                    q_idx += 1
                n_repeats += 1
            group_ranges.append((curr, int(batch.non_tensor_batch["is_grpo"][-1]), n_repeats))

            # 按组大小降序排列（把重复次数多的题先分配）
            group_ranges.sort(key=lambda x: (-x[1], -x[2]), reverse=False)

            # 贪心装箱：对每个题目组，分配到当前负载最小的那张卡
            for start_idx, is_grpo, size in group_ranges:
                tgt = min(range(world_size), key=lambda i: gpu_loads[i])  # 找到最空闲的 GPU
                global_partition_lst[tgt].extend(range(start_idx, start_idx + size))  # 分配 [start_idx, start_idx+size) 这段连续样本
                gpu_loads[tgt] += size
            max_size = max([load for load in gpu_loads])
            padding_sizes = [max_size - load for load in gpu_loads]

            is_padding = []
            for i in range(world_size):
                is_padding.extend([0] * len(global_partition_lst[i]))
                if padding_sizes[i] > 0:
                    global_partition_lst[i].extend([global_partition_lst[i][-1]] * padding_sizes[i])
                    is_padding.extend([1] * padding_sizes[i])

            global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
            batch.reorder(global_idx)
            batch.non_tensor_batch["is_padding"] = np.array(is_padding)
            reverse_global_idx = torch.argsort(global_idx)

            return global_idx, reverse_global_idx

        else:
            raise NotImplementedError(f"mode {mode} not supported in _balance_gen_batch")

    def log_and_save_metrics(self, metrics, logger, rollout_data_dir):
        logger.log(data=metrics, step=self.global_steps)
        if rollout_data_dir:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {type(value)}")
                    metrics[key] = value.item()
                elif isinstance(value, np.ndarray):
                    print(f"{key}: {type(value)}")
                    metrics[key] = value.tolist()
            filename = os.path.join(rollout_data_dir, f"metrics_{self.global_steps}.json")
            with open(filename, "w") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.sampling_cnt = 0
        self.generated_token_num = 0

        # load checkpoint before doing anything
        if not self.config.trainer.get("val_only", False):
            self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)
        if self.global_steps > 0 and self.global_steps % len(self.train_dataloader) == 0:
            current_epoch += 1

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self.data_filter.preprocess_epoch(epoch, dataset=None)

                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # generation accumulation
                    self.sampling_cnt += len(batch)
                    # self.generated_token_num += gen_batch_output.meta_info.pop("generated_token_num", None)
                    metrics["training/sampling_cnt"] = self.sampling_cnt
                    # metrics["batch/generated_token_num"] = self.generated_token_num

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        global_idx, reverse_global_idx = self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer, return_all=True
                            )
                        else:
                            reward_result = compute_reward(batch, self.reward_fn, return_all=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            batch.non_tensor_batch["extra_uid"] = np.arange(len(batch))
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            # entropys = old_log_prob.batch["entropys"]
                            # response_masks = batch.batch["response_mask"]
                            # actor_config = self.config.actor_rollout_ref.actor
                            # entropy_agg = agg_loss(
                            #     loss_mat=entropys,
                            #     loss_mask=response_masks,
                            #     loss_agg_mode=actor_config.loss_agg_mode,
                            #     loss_scale_factor=actor_config.loss_scale_factor,
                            # )
                            # old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            # metrics.update(old_log_prob_metrics)
                            # old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_result = ray.get(future_reward)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # GRPO-Lagrangian: R - λ p, buffer mean(p), update λ every z steps (overrides fixed ratio penalty)
                        penalty_cost_start_step = self.config.algorithm.get("penalty_cost_start_step", None)
                        clip_penalty_low = self.config.algorithm.get("clip_penalty_low", None)
                        clip_penalty_high = self.config.algorithm.get("clip_penalty_high", None)
                        normalize_mode = self.config.algorithm.get("normalize_mode", None)
                        normalize_tanh_k = self.config.algorithm.get("normalize_tanh_k", None)
                        penalty_truncated = self.config.algorithm.get("penalty_truncated", None)
                        lagrangian_apply_on = self._grpo_lagrangian.cfg.get("apply_on", "reward") if self._grpo_lagrangian is not None else "reward"
                        if self._grpo_lagrangian is not None and "rollout_log_probs" in batch.batch:
                            if lagrangian_apply_on != "adv":
                                # The operator owns raw mismatch extraction, sign split, dual updates,
                                # and reward shaping so the trainer stays on the main learning path.
                                lam_metrics = self._grpo_lagrangian.apply_to_batch(
                                    batch,
                                    penalty_cost_start_step=penalty_cost_start_step,
                                    global_step=self.global_steps,
                                    clip_penalty_low=clip_penalty_low,
                                    clip_penalty_high=clip_penalty_high,
                                    normalize_mode=normalize_mode,
                                    normalize_tanh_k=normalize_tanh_k,
                                    penalty_truncated=penalty_truncated,
                                )
                                metrics.update(lam_metrics)
                        else:
                            reward_ratio_penalty_cfg = self.config.algorithm.get("reward_ratio_penalty", None)
                            if (
                                reward_ratio_penalty_cfg
                                and reward_ratio_penalty_cfg.get("enable", False)
                                and "rollout_log_probs" in batch.batch
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import apply_ratio_penalty_to_reward

                                penalty_metrics = apply_ratio_penalty_to_reward(
                                    batch,
                                    ratio_type=reward_ratio_penalty_cfg.get("ratio_type", "seq_geo"),
                                    max_ratio=reward_ratio_penalty_cfg.get("max_ratio"),
                                    penalty_level=reward_ratio_penalty_cfg.get("penalty_level", "sequence"),
                                    normalize_by_response_length=reward_ratio_penalty_cfg.get("normalize_by_response_length", False),
                                    penalty_cost_start_step=penalty_cost_start_step,
                                    global_step=self.global_steps,
                                    clip_penalty_low=clip_penalty_low,
                                    clip_penalty_high=clip_penalty_high,
                                    normalize_mode=normalize_mode,
                                    normalize_tanh_k=normalize_tanh_k,
                                    penalty_truncated=penalty_truncated,
                                )
                                metrics.update(penalty_metrics)

                        # Rollout correction metrics: always record when rollout_log_probs present
                        # (trainer = full-batch π_old vs π_rollout; actor will record π_θ vs π_rollout)
                        if "rollout_log_probs" in batch.batch:
                            if (
                                rollout_corr_config is not None
                                and not bypass_recomputing_logprobs
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                    compute_mismatch_log_prob_by_old_prob_bucket_metrics,
                                    compute_mismatch_prob_by_old_prob_bucket_metrics,
                                    compute_mismatch_seq_frac_cnt_by_old_prob_bucket_metrics,
                                    compute_prob_ratio_by_old_prob_bucket_metrics,
                                    compute_seq_mismatch_log_prob_by_old_prob_bucket_metrics,
                                    compute_seq_mismatch_prob_by_old_prob_bucket_metrics,
                                    compute_seq_prob_ratio_by_old_prob_bucket_metrics,
                                    # compute_seq_sum_prob_ratio_by_old_prob_bucket_metrics,
                                    # compute_seq_sum_mismatch_log_prob_by_old_prob_bucket_metrics,
                                    # compute_seq_sum_mismatch_prob_by_old_prob_bucket_metrics,
                                    compute_mismatch_token_frac_cnt_by_old_prob_bucket_metrics,
                                    compute_ratio_by_length_bucket_metrics,
                                    compute_mismatch_vs_logits_metrics,
                                )

                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                                metrics.update(is_metrics)
                                # ratio 与 response 长度相关性：按长度分 4 桶，每桶统计 ratio_seq / ratio_seq_geo 的 mean、max、min
                                metrics.update(compute_ratio_by_length_bucket_metrics(batch, num_buckets=4))
                                # 不同 old token probability bucket 的 rollout/train mismatch（log-prob 与 prob）
                                metrics.update(compute_mismatch_token_frac_cnt_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_seq_frac_cnt_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_log_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_prob_ratio_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_seq_mismatch_log_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_seq_mismatch_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_seq_prob_ratio_by_old_prob_bucket_metrics(batch))
                                # metrics.update(compute_seq_sum_prob_ratio_by_old_prob_bucket_metrics(batch))
                                # metrics.update(compute_seq_sum_mismatch_log_prob_by_old_prob_bucket_metrics(batch))
                                # metrics.update(compute_seq_sum_mismatch_prob_by_old_prob_bucket_metrics(batch))
                                # 高 mismatch token 与 logits 衍生量（entropy / old_log_prob）的关联
                                metrics.update(compute_mismatch_vs_logits_metrics(batch))

                            else:
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_corr_metrics_only,
                                    compute_mismatch_log_prob_by_old_prob_bucket_metrics,
                                    compute_mismatch_prob_by_old_prob_bucket_metrics,
                                    compute_mismatch_seq_frac_cnt_by_old_prob_bucket_metrics,
                                    compute_prob_ratio_by_old_prob_bucket_metrics,
                                    compute_seq_mismatch_log_prob_by_old_prob_bucket_metrics,
                                    compute_seq_mismatch_prob_by_old_prob_bucket_metrics,
                                    compute_seq_prob_ratio_by_old_prob_bucket_metrics,
                                    # compute_seq_sum_prob_ratio_by_old_prob_bucket_metrics,
                                    # compute_seq_sum_mismatch_log_prob_by_old_prob_bucket_metrics,
                                    # compute_seq_sum_mismatch_prob_by_old_prob_bucket_metrics,
                                    compute_mismatch_token_frac_cnt_by_old_prob_bucket_metrics,
                                    compute_ratio_by_length_bucket_metrics,
                                    compute_mismatch_vs_logits_metrics,
                                )

                                metrics.update(compute_rollout_corr_metrics_only(batch))
                                metrics.update(compute_ratio_by_length_bucket_metrics(batch, num_buckets=4))
                                metrics.update(compute_mismatch_token_frac_cnt_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_seq_frac_cnt_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_log_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_prob_ratio_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_seq_mismatch_log_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_seq_mismatch_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_seq_prob_ratio_by_old_prob_bucket_metrics(batch))
                                # metrics.update(compute_seq_sum_prob_ratio_by_old_prob_bucket_metrics(batch))
                                # metrics.update(compute_seq_sum_mismatch_log_prob_by_old_prob_bucket_metrics(batch))
                                # metrics.update(compute_seq_sum_mismatch_prob_by_old_prob_bucket_metrics(batch))
                                metrics.update(compute_mismatch_vs_logits_metrics(batch))

                            # Log sequence-level ratio histograms directly to wandb (not through
                            # shared metrics dict, which gets json-serialized and sent to all loggers)
                            # Log sequence-level ratio histograms directly to wandb (not through
                            # shared metrics dict, which gets json-serialized and sent to all loggers)
                            import wandb

                            from verl.trainer.ppo.rollout_corr_helper import compute_ratio_histograms
                            hist_data = compute_ratio_histograms(batch)
                            wandb.log(hist_data, step=self.global_steps)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            global_step=self.global_steps,
                            grpo_lagrangian_operator=self._grpo_lagrangian,
                        )
                        if "adv_cost_metrics" in batch.meta_info:
                            metrics.update(batch.meta_info.pop("adv_cost_metrics"))
                        if "grpo_lagrangian_metrics" in batch.meta_info:
                            metrics.update(batch.meta_info.pop("grpo_lagrangian_metrics"))
                        if "rollout_log_probs" in batch.batch:
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_mismatch_log_prob_by_old_prob_and_adv_sign_metrics,
                                compute_mismatch_prob_by_old_prob_and_adv_sign_metrics,
                                compute_mismatch_seq_frac_cnt_by_old_prob_and_adv_sign_metrics,
                                compute_prob_ratio_by_old_prob_and_adv_sign_metrics,
                                compute_seq_mismatch_log_prob_by_old_prob_and_adv_sign_metrics,
                                compute_seq_mismatch_prob_by_old_prob_and_adv_sign_metrics,
                                compute_seq_prob_ratio_by_old_prob_and_adv_sign_metrics,
                                # compute_seq_sum_prob_ratio_by_old_prob_and_adv_sign_metrics,
                                # compute_seq_sum_mismatch_log_prob_by_old_prob_and_adv_sign_metrics,
                                # compute_seq_sum_mismatch_prob_by_old_prob_and_adv_sign_metrics,
                                compute_mismatch_token_frac_cnt_by_old_prob_and_adv_sign_metrics,
                            )

                            metrics.update(compute_mismatch_token_frac_cnt_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_mismatch_seq_frac_cnt_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_mismatch_log_prob_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_mismatch_prob_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_prob_ratio_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_seq_mismatch_log_prob_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_seq_mismatch_prob_by_old_prob_and_adv_sign_metrics(batch))
                            metrics.update(compute_seq_prob_ratio_by_old_prob_and_adv_sign_metrics(batch))
                            # metrics.update(compute_seq_sum_prob_ratio_by_old_prob_and_adv_sign_metrics(batch))
                            # metrics.update(compute_seq_sum_mismatch_log_prob_by_old_prob_and_adv_sign_metrics(batch))
                            # metrics.update(compute_seq_sum_mismatch_prob_by_old_prob_and_adv_sign_metrics(batch))

                    ### TODO: Compute statistics here
                    if self.config.trainer.balance_batch:
                        batch.reorder(reverse_global_idx)

                    keys_to_pop = []
                    with marked_timer("compute_metrics", timing_raw, color="red"):
                        # Get metrics and update data_filter
                        temp_batch = self._maybe_log_extra_reward_metrics(reward_result, batch.batch["token_level_scores"].sum(-1).tolist(), metrics)
                        batch = batch.union(temp_batch)
                        keys_to_pop.extend(list(temp_batch.batch.keys()))
                        data_metrics, proxies, _ = compute_data_metrics(
                            batch, self, None, None, epoch, use_critic=self.use_critic, config=self.config, batch_grad_direction=None
                        )
                        batch = batch.union(proxies)
                        keys_to_pop.extend(list(proxies.batch.keys()))
                        metrics.update(data_metrics)

                        batch.pop(batch_keys=keys_to_pop)
                        metrics.update(data_metrics)
                        ### TODO: End

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    actor_output = None
                    post_update_extra_tensors: dict[str, torch.Tensor] = {}
                    save_all_rollout_data = self.config.trainer.get("save_all_rollout_data", False)
                    compute_prob_shift = self.config.trainer.get("compute_prob_shift", False)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            batch.meta_info["save_all_rollout_data"] = save_all_rollout_data
                            batch.meta_info["compute_prob_shift"] = compute_prob_shift
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        post_update_metrics, post_update_extra_tensors = self._compute_post_update_actor_outputs(
                            batch=batch, timing_raw=timing_raw, save_all_rollout_data=save_all_rollout_data, compute_prob_shift=compute_prob_shift, actor_output=actor_output,
                        )
                        metrics.update(post_update_metrics)

                    # Log rollout generations and optionally dump full batch tensors.
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                        if save_all_rollout_data:
                            self._save_all_rollout_batch_tensors(
                                batch, timing_raw, rollout_data_dir, actor_output=actor_output, extra_tensors=post_update_extra_tensors
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                if self._grpo_lagrangian is not None:
                    metrics.update(self._grpo_lagrangian.maybe_update(self.global_steps))

                self.log_and_save_metrics(metrics, logger, rollout_data_dir)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
