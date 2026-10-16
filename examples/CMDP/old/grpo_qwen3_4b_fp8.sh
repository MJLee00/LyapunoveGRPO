#!/usr/bin/env bash

set -x

REWARD_RATIO_PENALTY_ENABLE=${REWARD_RATIO_PENALTY_ENABLE:-false}
REWARD_RATIO_PENALTY_RATIO_TYPE=${REWARD_RATIO_PENALTY_RATIO_TYPE:-"token"}

if [ -z "${EXPERIMENT_NAME}" ]; then
  if [ "${REWARD_RATIO_PENALTY_ENABLE}" = "true" ]; then
    EXPERIMENT_NAME="grpo_fp8_qwen34b_8k_cost_penalty_${REWARD_RATIO_PENALTY_RATIO_TYPE}"
  else
    EXPERIMENT_NAME="grpo_fp8_qwen34b_8k_cost_no_penalty"
  fi
fi

export JUICEFS=/mnt/ws-jfs/dsh-jfs
export NEW_HOME=$JUICEFS/lrz
export NEW_CONDA_HOME=$NEW_HOME/miniforge3
export BASE_PATH=$NEW_HOME/verl
export MODEL_BASE_PATH=$NEW_HOME/hf_models
export MODEL_PATH=$MODEL_BASE_PATH/Qwen3-4B-Base

eval "$($NEW_CONDA_HOME/bin/conda shell.bash hook)"
conda activate lv
cd $BASE_PATH

# WANDB_API_KEY must be provided through the environment; no credential is stored here.

project_name='CMDP'
adv_estimator=grpo
loss_mode=vanilla
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.2
loss_agg_mode="token-mean"

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))
train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=64

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${BASE_PATH}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${EXPERIMENT_NAME}"}
ROLLOUT_DIR=${VAL_DIR:-"${RAY_DATA_HOME}/rollout/${project_name}/${EXPERIMENT_NAME}"}
VAL_DIR=${VAL_DIR:-"${RAY_DATA_HOME}/val/${project_name}/${EXPERIMENT_NAME}"}
REWARD_DIR=${REWARD_DIR:-"${RAY_DATA_HOME}/reward/${project_name}/${EXPERIMENT_NAME}"}
WANDB_DIR_CUSTOM=${WANDB_DIR_CUSTOM:-"${RAY_DATA_HOME}/wandb_dir/${project_name}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${RAY_DATA_HOME}/logs/${project_name}/${EXPERIMENT_NAME}"}
mkdir -p ${LOG_DIR}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/train/dapo_train.parquet"}
# AIME24@32, AIME25@32, AMC23@32, MATH500@4, Minerva@4, Olympiad@4
# 30 x 32,   30 x 32,   40 x 32,  500 x 4,   272 x 4,   674 x 4   = 8984 samples
TEST_FILE=${TEST_FILE:-"['${RAY_DATA_HOME}/data/eval/aime24_32.parquet', '${RAY_DATA_HOME}/data/eval/aime25_32.parquet', '${RAY_DATA_HOME}/data/eval/amc23_32.parquet', '${RAY_DATA_HOME}/data/eval/math500_4.parquet', '${RAY_DATA_HOME}/data/eval/minerva_4.parquet', '${RAY_DATA_HOME}/data/eval/olympiad_4.parquet']"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.7
val_top_p=1.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 4))
max_model_len=$((max_prompt_length + max_response_length * 1))
max_num_seqs=$((1024 * 2))
offload=False
gen_tp=1
fsdp_size=-1
free_cache_engine=True
disable_log_stats=True
max_actor_ckpt_to_keep=1

balance_val_batch=True

val_only=$1
val_ckpt=$2
extra_args=()
if [ -z "$val_only" ]; then
    val_only=False
    extra_args+=(trainer.logger=['console','wandb'])
else
    echo "Running validation only: val_ckpt: ${val_ckpt}"
    val_only=True
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 32))
    if [ -z "$val_ckpt" ]; then
        val_ckpt=0
    fi
    extra_args+=(trainer.logger=['console'] trainer.val_only=True +trainer.val_ckpt=${val_ckpt})
fi

# Print out all env variables
printenv

export WANDB_MODE=online
export WANDB_DIR=${WANDB_DIR_CUSTOM}
export WANDB_CACHE_DIR=${WANDB_DIR_CUSTOM}
export WANDB_CONFIG_DIR=${WANDB_DIR_CUSTOM}
export WANDB_DATA_DIR=${WANDB_DIR_CUSTOM}
export WANDB_ARTIFACT_DIR=${WANDB_DIR_CUSTOM}
export WANDB_ARTIFACT_LOCATION=${WANDB_DIR_CUSTOM}
mkdir -p "${WANDB_DIR_CUSTOM}"
${NEW_CONDA_HOME}/envs/lv/bin/python -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    +actor_rollout_ref.rollout.quantization=fp8 \
    algorithm.reward_ratio_penalty.enable="${REWARD_RATIO_PENALTY_ENABLE}" \
    algorithm.reward_ratio_penalty.ratio_type="${REWARD_RATIO_PENALTY_RATIO_TYPE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.enforce_eager=${free_cache_engine} \
    actor_rollout_ref.rollout.free_cache_engine=${free_cache_engine} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.disable_log_stats=${disable_log_stats} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.engine_kwargs.balance_val_batch=${balance_val_batch} \
    actor_rollout_ref.nccl_timeout=7200 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    reward_model.reward_manager=rllm_skywork \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.reward_type=v2_math-verify \
    trainer.total_epochs=30 \
    trainer.total_training_steps=801 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    +trainer.wandb_mode=${WANDB_MODE} \
    trainer.log_val_generations=5 \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.validation_data_dir="${VAL_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.save_freq=20 \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    "${extra_args[@]}" 2>&1 | tee ${LOG_DIR}/$(date +%m%d-%H%M%S).log
