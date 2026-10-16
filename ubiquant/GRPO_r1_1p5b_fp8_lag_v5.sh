#!/usr/bin/env bash

set -x

# shared params
GRPO_LAGRANGIAN_ENABLE=${GRPO_LAGRANGIAN_ENABLE:-true}
GRPO_LAGRANGIAN_Z=${GRPO_LAGRANGIAN_Z:-1}
GRPO_LAGRANGIAN_BUFFER=${GRPO_LAGRANGIAN_BUFFER:-1}
GRPO_LAGRANGIAN_LAMBDA_MAX=${GRPO_LAGRANGIAN_LAMBDA_MAX:-1.0}
GRPO_LAGRANGIAN_LAMBDA_MIN=${GRPO_LAGRANGIAN_LAMBDA_MIN:-0.0}
GRPO_LAGRANGIAN_RATIO=${GRPO_LAGRANGIAN_RATIO:-seq_geo_no_exp}
GRPO_LAGRANGIAN_OPERATOR=${GRPO_LAGRANGIAN_OPERATOR:-dual}
GRPO_LAGRANGIAN_BUFFER_MODE=${GRPO_LAGRANGIAN_BUFFER_MODE:-mean}
GRPO_LAGRANGIAN_APPLY_ON=${GRPO_LAGRANGIAN_APPLY_ON:-reward}
GRPO_LAGRANGIAN_PENALTY_LEVEL=${GRPO_LAGRANGIAN_PENALTY_LEVEL:-sequence}
GRPO_LAGRANGIAN_NORMALIZE_BY_RESPONSE_LENGTH=${GRPO_LAGRANGIAN_NORMALIZE_BY_RESPONSE_LENGTH:-false}
NORMALIZE_MODE=${NORMALIZE_MODE:-sigmoid}
DTYPE=${DTYPE:-"bfloat16"}
SAVE_ALL_ROLLOUT_DATA=${SAVE_ALL_ROLLOUT_DATA:-false}
COMPUTE_PROB_SHIFT=${COMPUTE_PROB_SHIFT:-false}
SAVE_CKPT=${SAVE_CKPT:-false}
clean=${clean:-0}

# unshared
GRPO_LAGRANGIAN_LAMBDA_INIT=${GRPO_LAGRANGIAN_LAMBDA_INIT:-0.4}
GRPO_LAGRANGIAN_LAMBDA_LR=${GRPO_LAGRANGIAN_LAMBDA_LR:-1e1}
GRPO_LAGRANGIAN_C=${GRPO_LAGRANGIAN_C:--0.00050}

GRPO_LAGRANGIAN_ENABLE2=${GRPO_LAGRANGIAN_ENABLE2:-false}
GRPO_LAGRANGIAN_LAMBDA_INIT2=${GRPO_LAGRANGIAN_LAMBDA_INIT2:-0.4}
GRPO_LAGRANGIAN_LAMBDA_LR2=${GRPO_LAGRANGIAN_LAMBDA_LR2:-1e1}
GRPO_LAGRANGIAN_C2=${GRPO_LAGRANGIAN_C2:--0.00050}

if [ -z "${EXPERIMENT_NAME}" ]; then
  if [ "${DTYPE}" != "bfloat16" ]; then
    EXPERIMENT_NAME="v5_r1-1p5b_fp16"
  elif [ "${GRPO_LAGRANGIAN_ENABLE2}" = "true" ]; then
    EXPERIMENT_NAME="v5_r1-1p5b_fp8_lag_${GRPO_LAGRANGIAN_RATIO}_c${GRPO_LAGRANGIAN_C}_init${GRPO_LAGRANGIAN_LAMBDA_INIT}_lr${GRPO_LAGRANGIAN_LAMBDA_LR}_buf${GRPO_LAGRANGIAN_BUFFER_MODE}_norm${NORMALIZE_MODE}_lag2_c${GRPO_LAGRANGIAN_C2}_init${GRPO_LAGRANGIAN_LAMBDA_INIT2}_lr${GRPO_LAGRANGIAN_LAMBDA_LR2}"
  else
    EXPERIMENT_NAME="v5_r1-1p5b_fp8_lag_${GRPO_LAGRANGIAN_RATIO}_c${GRPO_LAGRANGIAN_C}_init${GRPO_LAGRANGIAN_LAMBDA_INIT}_lr${GRPO_LAGRANGIAN_LAMBDA_LR}_buf${GRPO_LAGRANGIAN_BUFFER_MODE}_norm${NORMALIZE_MODE}"
  fi
fi
if [ "${GRPO_LAGRANGIAN_APPLY_ON}" = "adv" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_adv"
fi
if [ "${SAVE_ALL_ROLLOUT_DATA}" = "true" ] && [ "${COMPUTE_PROB_SHIFT}" = "true" ] && [[ "${EXPERIMENT_NAME}" != *"_log" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_log"
fi

export BASE_PATH="${BASE_PATH:-/workspace/verl}"
export MODEL_PATH="${MODEL_PATH:-/path/to/DeepSeek-R1-Distill-Qwen-1.5B}"

TRAIN_FILE=${TRAIN_FILE:-"${BASE_PATH}/data/train/math_1460.parquet"}
# AIME24@32, AIME25@32, AMC23@32, MATH500@4, Minerva@4, Olympiad@4
# 30 x 32,   30 x 32,   40 x 32,  500 x 4,   272 x 4,   674 x 4   = 8984 samples
TEST_FILE=${TEST_FILE:-"['${BASE_PATH}/data/eval/aime24_32.parquet', '${BASE_PATH}/data/eval/aime25_32.parquet', '${BASE_PATH}/data/eval/amc23_32.parquet', '${BASE_PATH}/data/eval/math500_4.parquet', '${BASE_PATH}/data/eval/minerva_4.parquet', '${BASE_PATH}/data/eval/olympiad_4.parquet']"}

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
clip_ratio_high=0.24
loss_agg_mode="seq-mean-token-mean"

# Rollout Correction parameters for FP8 rollout
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))
train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=16

# Ray
#WORKING_DIR=${WORKING_DIR:-"${PWD}"}
#echo "WORKING_DIR: ${WORKING_DIR}"
#RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env_fp8.yaml"}
#echo "RUNTIME_ENV: ${RUNTIME_ENV}"
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
# 如果 LOG_DIR 存在就删除里面的所有文件
if [ -d "${LOG_DIR}" ] && [ "${clean}" = "1" ]; then
    python ${BASE_PATH}/examples/CMDP/clean.py -l "${LOG_DIR}"
fi
mkdir -p ${LOG_DIR}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.7
val_top_p=1.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=32768
infer_ppo_max_token_len=32768
max_num_batched_tokens=20480
max_model_len=$((max_prompt_length + max_response_length * 1))
max_num_seqs=$((1024 * 1))
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
if [ $SAVE_ALL_ROLLOUT_DATA = "true" ]; then
    extra_args+=(+trainer.save_all_rollout_data=True)
fi
if [ $COMPUTE_PROB_SHIFT = "true" ]; then
    extra_args+=(+trainer.compute_prob_shift=True)
fi
if [ $SAVE_CKPT = "true" ]; then
    extra_args+=(+trainer.save_ckpt=True actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'])
else
    extra_args+=(actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'])
fi
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

# Set Flash-RL environment variables
export VERL_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

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
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
-- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.seed=42 \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.eps=1e-15 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.dtype=${DTYPE} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.free_cache_engine=${free_cache_engine} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
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
    +actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.normalize_mode="${NORMALIZE_MODE}" \
    algorithm.grpo_lagrangian.enable="${GRPO_LAGRANGIAN_ENABLE}" \
    algorithm.grpo_lagrangian.constraint_budget_c="${GRPO_LAGRANGIAN_C}" \
    algorithm.grpo_lagrangian.operator_update_every_z="${GRPO_LAGRANGIAN_Z}" \
    algorithm.grpo_lagrangian.buffer_size="${GRPO_LAGRANGIAN_BUFFER}" \
    algorithm.grpo_lagrangian.lambda_lr="${GRPO_LAGRANGIAN_LAMBDA_LR}" \
    algorithm.grpo_lagrangian.lambda_max="${GRPO_LAGRANGIAN_LAMBDA_MAX}" \
    algorithm.grpo_lagrangian.lambda_init="${GRPO_LAGRANGIAN_LAMBDA_INIT}" \
    algorithm.grpo_lagrangian.lambda_min="${GRPO_LAGRANGIAN_LAMBDA_MIN}" \
    algorithm.grpo_lagrangian.ratio_type="${GRPO_LAGRANGIAN_RATIO}" \
    algorithm.grpo_lagrangian.operator_type="${GRPO_LAGRANGIAN_OPERATOR}" \
    algorithm.grpo_lagrangian.buffer_mode="${GRPO_LAGRANGIAN_BUFFER_MODE}" \
    +algorithm.grpo_lagrangian.apply_on="${GRPO_LAGRANGIAN_APPLY_ON}" \
    +algorithm.grpo_lagrangian.penalty_level="${GRPO_LAGRANGIAN_PENALTY_LEVEL}" \
    +algorithm.grpo_lagrangian.normalize_by_response_length="${GRPO_LAGRANGIAN_NORMALIZE_BY_RESPONSE_LENGTH}" \
    algorithm.grpo_lagrangian.enable_lambda2="${GRPO_LAGRANGIAN_ENABLE2}" \
    algorithm.grpo_lagrangian.lambda_lr2="${GRPO_LAGRANGIAN_LAMBDA_LR2}" \
    algorithm.grpo_lagrangian.lambda_init2="${GRPO_LAGRANGIAN_LAMBDA_INIT2}" \
    algorithm.grpo_lagrangian.constraint_budget_c2="${GRPO_LAGRANGIAN_C2}" \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    reward_model.reward_manager=rllm_skywork \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.reward_type=v2_math-verify-01 \
    trainer.total_epochs=50 \
    trainer.total_training_steps=1001 \
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
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    +trainer.dump_high_diff_tokens=False \
    +trainer.dump_high_diff_dir="${CKPTS_DIR}/logp_diff" \
    actor_rollout_ref.rollout.enforce_eager=False \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    "${extra_args[@]}" 2>&1 | tee ${LOG_DIR}/$(date +%m%d-%H%M%S).log
