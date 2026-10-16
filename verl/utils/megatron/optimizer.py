# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import inspect
from dataclasses import fields, is_dataclass

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from verl.utils.logger import print_rank_0


def _get_optimizer_config_supported_args() -> set[str] | None:
    supported_args = set()

    try:
        supported_args.update(inspect.signature(OptimizerConfig).parameters)
    except (TypeError, ValueError):
        pass

    if not supported_args and is_dataclass(OptimizerConfig):
        supported_args.update(field.name for field in fields(OptimizerConfig))

    return supported_args or None


def _normalize_optimizer_kwarg_name(arg_name: str, supported_args: set[str] | None) -> str | None:
    if supported_args is None or arg_name in supported_args:
        return arg_name

    alias_candidates = {
        "eps": ("adam_eps",),
        "beta1": ("adam_beta1",),
        "beta2": ("adam_beta2",),
    }

    for candidate in alias_candidates.get(arg_name, ()):
        if candidate in supported_args:
            return candidate

    return None


def _filter_supported_optimizer_args(optim_args: dict, supported_args: set[str] | None) -> dict:
    if supported_args is None:
        return optim_args

    filtered_args = {}
    dropped_args = []
    remapped_args = {}

    for arg_name, arg_value in optim_args.items():
        normalized_name = _normalize_optimizer_kwarg_name(arg_name, supported_args)
        if normalized_name is None:
            dropped_args.append(arg_name)
            continue

        filtered_args[normalized_name] = arg_value
        if normalized_name != arg_name:
            remapped_args[arg_name] = normalized_name

    if remapped_args:
        print_rank_0(f"remapped megatron optimizer args: {remapped_args}")
    if dropped_args:
        print_rank_0(f"dropped unsupported megatron optimizer args: {sorted(dropped_args)}")

    return filtered_args


def init_megatron_optim_config(optim_config: dict, fp16: bool = False) -> OptimizerConfig:
    supported_args = _get_optimizer_config_supported_args()

    optim_args = {
        "optimizer": optim_config.optimizer,
        "lr": optim_config.lr,
        "min_lr": optim_config.min_lr,
        "clip_grad": optim_config.clip_grad,
        "weight_decay": optim_config.weight_decay,
        "use_distributed_optimizer": True,
    }
    beta1, beta2 = optim_config.betas
    if supported_args is None or {"adam_beta1", "adam_beta2"}.issubset(supported_args):
        optim_args["adam_beta1"] = beta1
        optim_args["adam_beta2"] = beta2
    elif {"beta1", "beta2"}.issubset(supported_args):
        optim_args["beta1"] = beta1
        optim_args["beta2"] = beta2
    elif "betas" in supported_args:
        optim_args["betas"] = (beta1, beta2)

    if fp16:
        optim_args.update(
            {
                "bf16": False,
                "fp16": True,
                "params_dtype": torch.float16,
                "initial_loss_scale": 32768,
                "min_loss_scale": 1,
                "use_precision_aware_optimizer": True,
                "store_param_remainders": False,
            }
        )
    else:  # bf16 mode
        optim_args.update(
            {
                "bf16": True,
                "params_dtype": torch.bfloat16,
            }
        )
    override_config = optim_config.get("override_optimizer_config", {})
    if override_config:
        for k, v in override_config.items():
            optim_args[k] = v

    optim_args = _filter_supported_optimizer_args(optim_args, supported_args)
    print_rank_0(f"optimizer config after override: {optim_args}")

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    lr_decay_steps = config.lr_decay_steps
    lr_warmup_steps = config.lr_warmup_steps
    if config.get("lr_decay_steps", None) is None:
        lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
