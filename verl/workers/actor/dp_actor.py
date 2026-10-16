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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _safe_std_for_logging(values: torch.Tensor) -> torch.Tensor:
    # Dynamic batching can yield 0/1-sample micro-batches, where sample std is undefined.
    if values.numel() <= 1:
        return values.new_zeros(())
    return torch.std(values)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            chosen_logits: # (bs, response_len) 或 None，采样 token 的未缩放 logit（绝对尺度）
            max_logits: # (bs, response_len) 或 None，该位置 vocab 上 max logit（reduce_max）
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            chosen_logits = None
            max_logits = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    # 绝对尺度：采样 token 的 logit + 该位置 max logit（未除 temperature），用于 mismatch vs logits 关联分析
                    logits_slice = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    chosen_logits = torch.gather(
                        logits_slice, dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                    ).squeeze(-1)  # (bsz, response_length)
                    max_logits = logits_slice.max(dim=-1).values  # (bsz, response_length)

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, chosen_logits, max_logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        chosen_logits_lst = []
        max_logits_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, chosen_logits, max_logits = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if chosen_logits is not None:
                chosen_logits_lst.append(chosen_logits)
            if max_logits is not None:
                max_logits_lst.append(max_logits)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        chosen_logits_out = torch.concat(chosen_logits_lst, dim=0) if chosen_logits_lst else None
        max_logits_out = torch.concat(max_logits_lst, dim=0) if max_logits_lst else None

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if chosen_logits_out is not None:
                chosen_logits_out = restore_dynamic_batch(chosen_logits_out, batch_idx_list)
            if max_logits_out is not None:
                max_logits_out = restore_dynamic_batch(max_logits_out, batch_idx_list)

        return log_probs, entropys, chosen_logits_out, max_logits_out

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        need_response_minibatch_idx = bool(data.meta_info.get("save_all_rollout_data", False)) or bool(data.meta_info.get("compute_prob_shift", False))

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        if torch.distributed.get_rank() == 0:
            print(f"{on_policy=}")

        metrics = {}
        # Accumulate ratio tensors across micro batches (first epoch only) for full-batch std/median
        acc_token: list = []
        acc_seq: list = []
        acc_seq_geo: list = []
        acc_pi_train_seq: list = []
        acc_pi_rollout_seq: list = []
        acc_pi_train_seq_geo: list = []
        acc_pi_rollout_seq_geo: list = []
        acc_diff_seq_sum_add: list = []
        extra_tensors: dict[str, torch.Tensor] = {}

        if need_response_minibatch_idx:
            response_minibatch_idx = torch.empty((data.batch["old_log_probs"].shape[0],), dtype=torch.long)
            row_start = 0
            for batch_idx, mini_batch in enumerate(mini_batches):
                mini_batch_size = mini_batch.batch["old_log_probs"].shape[0]
                response_minibatch_idx[row_start : row_start + mini_batch_size] = batch_idx + 1
                row_start += mini_batch_size
            assert row_start == data.batch["old_log_probs"].shape[0]
            extra_tensors["response_minibatch_idx"] = response_minibatch_idx

        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    if torch.distributed.get_rank() == 0:
                        total_seqlen = mini_batch.batch["attention_mask"].sum().item()
                        print(f"{batch_idx=}, {len(mini_batches)=}, {total_seqlen=}, {max_token_len=}, pct: {(total_seqlen / max_token_len * 100):.2f}%")
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob, _, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Rollout_corr: offpolicy per-micro (append then mean); ratio stats full-batch (accumulate then compute once)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import (
                            compute_offpolicy_metrics,
                            compute_ratio_stats_from_tensors,
                            get_ratio_tensors_for_accumulation,
                        )

                        offpolicy = compute_offpolicy_metrics(
                            old_log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        for k, v in offpolicy.items():
                            micro_batch_metrics[f"rollout_corr_actor/{k}"] = v if not isinstance(v, torch.Tensor) else v.item()
                        if epoch_idx == 0:
                            (
                                t_flat,
                                r_seq,
                                r_geo,
                                pi_train,
                                pi_rollout,
                                pi_train_geo,
                                pi_rollout_geo,
                                diff_sum_add,
                            ) = get_ratio_tensors_for_accumulation(
                                log_prob, rollout_log_prob, response_mask
                            )
                            if t_flat.numel() > 0 or r_seq.numel() > 0:
                                # Keep rollout-corr full-batch accumulation on CPU to avoid extra GPU peak memory.
                                acc_token.append(t_flat.detach().cpu())
                                acc_seq.append(r_seq.detach().cpu())
                                acc_seq_geo.append(r_geo.detach().cpu())
                                acc_pi_train_seq.append(pi_train.detach().cpu())
                                acc_pi_rollout_seq.append(pi_rollout.detach().cpu())
                                acc_pi_train_seq_geo.append(pi_train_geo.detach().cpu())
                                acc_pi_rollout_seq_geo.append(pi_rollout_geo.detach().cpu())
                                acc_diff_seq_sum_add.append(diff_sum_add.detach().cpu())

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Keep compatibility with old actor diagnostics.
                    metric_old_log_prob = model_inputs["old_log_probs"]
                    probs_diff = (torch.exp(log_prob) - torch.exp(metric_old_log_prob)) * response_mask
                    probs_diff_abs = torch.abs(probs_diff)
                    micro_batch_metrics["actor/probs_diff_max"] = torch.max(probs_diff).detach().item()
                    micro_batch_metrics["actor/probs_diff_min"] = torch.min(probs_diff).detach().item()
                    micro_batch_metrics["actor/probs_diff_mean"] = torch.mean(probs_diff_abs).detach().item()
                    micro_batch_metrics["actor/probs_diff_std"] = _safe_std_for_logging(probs_diff_abs).detach().item()

                    log_probs_diff = (log_prob - metric_old_log_prob) * response_mask
                    micro_batch_metrics["actor/log_probs_diff_max"] = torch.max(log_probs_diff).detach().item()
                    micro_batch_metrics["actor/log_probs_diff_min"] = torch.min(log_probs_diff).detach().item()

                    if rollout_log_prob is not None:
                        seq_iw = ((log_prob - rollout_log_prob) * response_mask).sum(dim=-1)
                        seq_iw_abs = torch.abs(seq_iw)
                        micro_batch_metrics["actor/log_seq_iw_max"] = torch.max(seq_iw).detach().item()
                        micro_batch_metrics["actor/log_seq_iw_min"] = torch.min(seq_iw).detach().item()
                        micro_batch_metrics["actor/log_seq_iw_mean"] = torch.mean(seq_iw_abs).detach().item()
                        micro_batch_metrics["actor/log_seq_iw_std"] = _safe_std_for_logging(seq_iw_abs).detach().item()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                weights_l2_norm = 0.0
                for param in self.actor_module.parameters():
                    if param.requires_grad:
                        weights_l2_norm += torch.norm(param.detach(), p=2).item() ** 2
                mini_batch_metrics["actor/weights_l2_norm"] = weights_l2_norm
                append_to_dict(metrics, mini_batch_metrics)

        # Full-batch ratio stats (std/median are true batch stats; only first-epoch data)
        if acc_token or acc_seq:
            from verl.trainer.ppo.rollout_corr_helper import compute_ratio_stats_from_tensors

            # Aggregation is intentionally performed on CPU to reduce actor-side GPU memory peak.
            token_all = torch.cat(acc_token, dim=0) if acc_token else torch.empty(0)
            seq_all = torch.cat(acc_seq, dim=0) if acc_seq else torch.empty(0)
            seq_geo_all = torch.cat(acc_seq_geo, dim=0) if acc_seq_geo else torch.empty(0)
            pi_train_all = torch.cat(acc_pi_train_seq, dim=0) if acc_pi_train_seq else torch.empty(0)
            pi_rollout_all = torch.cat(acc_pi_rollout_seq, dim=0) if acc_pi_rollout_seq else torch.empty(0)
            pi_train_geo_all = torch.cat(acc_pi_train_seq_geo, dim=0) if acc_pi_train_seq_geo else torch.empty(0)
            pi_rollout_geo_all = torch.cat(acc_pi_rollout_seq_geo, dim=0) if acc_pi_rollout_seq_geo else torch.empty(0)
            diff_seq_sum_add_all = torch.cat(acc_diff_seq_sum_add, dim=0) if acc_diff_seq_sum_add else torch.empty(0)
            full_batch_ratio = compute_ratio_stats_from_tensors(
                token_all,
                seq_all,
                seq_geo_all,
                pi_train_seq=pi_train_all,
                pi_rollout_seq=pi_rollout_all,
                pi_train_seq_geo=pi_train_geo_all,
                pi_rollout_seq_geo=pi_rollout_geo_all,
                diff_seq_sum_add=diff_seq_sum_add_all,
            )
            full_batch_ratio_prefix = {f"rollout_corr_actor/{k}": [v] for k, v in full_batch_ratio.items()}
            append_to_dict(metrics, full_batch_ratio_prefix)
            # Explicitly drop full-batch aggregation buffers before leaving update_policy.
            del token_all, seq_all, seq_geo_all, pi_train_all, pi_rollout_all, pi_train_geo_all, pi_rollout_geo_all, diff_seq_sum_add_all

        acc_token.clear()
        acc_seq.clear()
        acc_seq_geo.clear()
        acc_pi_train_seq.clear()
        acc_pi_rollout_seq.clear()
        acc_pi_train_seq_geo.clear()
        acc_pi_rollout_seq_geo.clear()
        acc_diff_seq_sum_add.clear()
        self.actor_optimizer.zero_grad()
        if extra_tensors:
            return metrics, extra_tensors
        return metrics
