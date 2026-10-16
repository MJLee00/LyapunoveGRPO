#!/usr/bin/env bash

set -euo pipefail
set -x

# shared params
GRPO_LAGRANGIAN_ENABLE=${GRPO_LAGRANGIAN_ENABLE:-true}
GRPO_LAGRANGIAN_Z=${GRPO_LAGRANGIAN_Z:-1}
GRPO_LAGRANGIAN_BUFFER=${GRPO_LAGRANGIAN_BUFFER:-1}
GRPO_LAGRANGIAN_LAMBDA_MAX=${GRPO_LAGRANGIAN_LAMBDA_MAX:-1.0}
GRPO_LAGRANGIAN_LAMBDA_MIN=${GRPO_LAGRANGIAN_LAMBDA_MIN:-0.0}
GRPO_LAGRANGIAN_RATIO=${GRPO_LAGRANGIAN_RATIO:-token_prob_diff}
GRPO_LAGRANGIAN_OPERATOR=${GRPO_LAGRANGIAN_OPERATOR:-dual}
GRPO_LAGRANGIAN_BUFFER_MODE=${GRPO_LAGRANGIAN_BUFFER_MODE:-abs_mean}
GRPO_LAGRANGIAN_APPLY_ON=${GRPO_LAGRANGIAN_APPLY_ON:-adv}
GRPO_LAGRANGIAN_PENALTY_LEVEL=${GRPO_LAGRANGIAN_PENALTY_LEVEL:-token}
GRPO_LAGRANGIAN_NORMALIZE_BY_RESPONSE_LENGTH=${GRPO_LAGRANGIAN_NORMALIZE_BY_RESPONSE_LENGTH:-false}
NORMALIZE_MODE=${NORMALIZE_MODE:-absneg_0.0_none}
DTYPE=${DTYPE:-"bfloat16"}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
SAVE_ALL_ROLLOUT_DATA=${SAVE_ALL_ROLLOUT_DATA:-false}
COMPUTE_PROB_SHIFT=${COMPUTE_PROB_SHIFT:-false}
SAVE_CKPT=${SAVE_CKPT:-false}
DATA_TYPE=${DATA_TYPE:-"dapo"}
clean=${clean:-0}

# unshared
GRPO_LAGRANGIAN_LAMBDA_INIT=${GRPO_LAGRANGIAN_LAMBDA_INIT:-0.1}
GRPO_LAGRANGIAN_LAMBDA_LR=${GRPO_LAGRANGIAN_LAMBDA_LR:-1e0}
GRPO_LAGRANGIAN_C=${GRPO_LAGRANGIAN_C:-0.0040}

GRPO_LAGRANGIAN_ENABLE2=${GRPO_LAGRANGIAN_ENABLE2:-false}
GRPO_LAGRANGIAN_LAMBDA_INIT2=${GRPO_LAGRANGIAN_LAMBDA_INIT2:-0.0}
GRPO_LAGRANGIAN_LAMBDA_LR2=${GRPO_LAGRANGIAN_LAMBDA_LR2:-1e0}
GRPO_LAGRANGIAN_C2=${GRPO_LAGRANGIAN_C2:-0.0040}

if [ -z "${EXPERIMENT_NAME:-}" ]; then
  if [ "${DTYPE}" != "bfloat16" ]; then
    EXPERIMENT_NAME="v5_qwen3-8b_fp16"
  elif [ "${GRPO_LAGRANGIAN_ENABLE2}" = "true" ]; then
    EXPERIMENT_NAME="v5_qwen3-8b_lag_${GRPO_LAGRANGIAN_RATIO}_c${GRPO_LAGRANGIAN_C}_init${GRPO_LAGRANGIAN_LAMBDA_INIT}_lr${GRPO_LAGRANGIAN_LAMBDA_LR}_buf${GRPO_LAGRANGIAN_BUFFER_MODE}_norm${NORMALIZE_MODE}_lag2_c${GRPO_LAGRANGIAN_C2}_init${GRPO_LAGRANGIAN_LAMBDA_INIT2}_lr${GRPO_LAGRANGIAN_LAMBDA_LR2}"
  else
    EXPERIMENT_NAME="v5_qwen3-8b_lag_${GRPO_LAGRANGIAN_RATIO}_c${GRPO_LAGRANGIAN_C}_init${GRPO_LAGRANGIAN_LAMBDA_INIT}_lr${GRPO_LAGRANGIAN_LAMBDA_LR}_buf${GRPO_LAGRANGIAN_BUFFER_MODE}_norm${NORMALIZE_MODE}"
  fi
fi
if [ "${GRPO_LAGRANGIAN_APPLY_ON}" = "adv" ] && [[ "${EXPERIMENT_NAME}" != *"_bf16" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_adv"
fi
if [ "${SAVE_ALL_ROLLOUT_DATA}" = "true" ] && [ "${COMPUTE_PROB_SHIFT}" = "true" ] && [[ "${EXPERIMENT_NAME}" != *"_log" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_log"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export BASE_PATH="${BASE_PATH:-${REPO_ROOT}}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-${BASE_PATH}/models}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_BASE_PATH}/Qwen3-8B}"

if [ "${DATA_TYPE}" = "dapo_2080" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_2080"
    TRAIN_FILE="${BASE_PATH}/data/train/dapo_train_qwen3-8b_20_80_6170.parquet"
elif [ "${DATA_TYPE}" = "dapo_0100" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_0100"
    TRAIN_FILE="${BASE_PATH}/data/train/dapo_train_qwen3-8b_0_100_13845.parquet"
else
    TRAIN_FILE=${TRAIN_FILE:-"${BASE_PATH}/data/train/dapo_train.parquet"}
fi
# AIME24@32, AIME25@32, AMC23@32, MATH500@4, Minerva@4, Olympiad@4
# 30 x 32,   30 x 32,   40 x 32,  500 x 4,   272 x 4,   674 x 4   = 8984 samples
TEST_FILE=${TEST_FILE:-"['${BASE_PATH}/data/eval/aime24_32.parquet', '${BASE_PATH}/data/eval/aime25_32.parquet', '${BASE_PATH}/data/eval/amc23_32.parquet', '${BASE_PATH}/data/eval/math500_4.parquet', '${BASE_PATH}/data/eval/minerva_4.parquet', '${BASE_PATH}/data/eval/olympiad_4.parquet']"}

if [ -n "${CONDA_HOME:-}" ] && [ -x "${CONDA_HOME}/bin/conda" ]; then
    eval "$(${CONDA_HOME}/bin/conda shell.bash hook)"
    conda activate "${CONDA_ENV:-lv}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN is not executable" >&2
    exit 1
fi
cd "${BASE_PATH}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_ENABLE_MONITORING=0   # 可选：彻底关掉 watchdog

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

max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 1))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}

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
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 4))
actor_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU:-${actor_ppo_max_token_len}}
infer_ppo_max_token_len=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${infer_ppo_max_token_len}}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-${max_num_batched_tokens}}
max_model_len=$((max_prompt_length + max_response_length * 1))
max_num_seqs=${MAX_NUM_SEQS:-$((1024 * 1))}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
enforce_eager=${ENFORCE_EAGER:-false}
total_training_steps=${TOTAL_TRAINING_STEPS:-301}
test_freq=${TEST_FREQ:-20}
offload=${OFFLOAD:-False}
gen_tp=1
fsdp_size=-1
free_cache_engine=True
disable_log_stats=True
max_actor_ckpt_to_keep=1

balance_val_batch=True

val_only="${1:-}"
val_ckpt="${2:-}"
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
    val_before_train=False
else
    echo "Running validation only: val_ckpt: ${val_ckpt}"
    val_only=True
    val_before_train=True
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 32))
    infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
    max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))
    max_model_len=$((max_prompt_length + max_response_length * 1))
    if [ -z "$val_ckpt" ]; then
        val_ckpt=0
    fi
    extra_args+=(trainer.logger=['console'] trainer.val_only=True +trainer.val_ckpt=${val_ckpt})
    if [ "$1" == "ood" ]; then
        TEST_FILE="['${RAY_DATA_HOME}/data/eval/arc-c_1.parquet', '${RAY_DATA_HOME}/data/eval/gpqa-diamond_4.parquet', '${RAY_DATA_HOME}/data/eval/mmlu-pro_1.parquet']"
    elif [ "$1" == "64" ]; then
        TEST_FILE="['${RAY_DATA_HOME}/data/eval/aime24_64.parquet', '${RAY_DATA_HOME}/data/eval/aime25_64.parquet', '${RAY_DATA_HOME}/data/eval/amc23_64.parquet', '${RAY_DATA_HOME}/data/eval/math500_4.parquet', '${RAY_DATA_HOME}/data/eval/minerva_8.parquet', '${RAY_DATA_HOME}/data/eval/olympiad_4.parquet']"
    else
        TEST_FILE="['${RAY_DATA_HOME}/data/eval/aime24_32.parquet', '${RAY_DATA_HOME}/data/eval/aime25_32.parquet', '${RAY_DATA_HOME}/data/eval/amc23_32.parquet', '${RAY_DATA_HOME}/data/eval/math500_4.parquet', '${RAY_DATA_HOME}/data/eval/minerva_4.parquet', '${RAY_DATA_HOME}/data/eval/olympiad_4.parquet']"
    fi
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR_CUSTOM}"
export WANDB_CACHE_DIR="${WANDB_DIR_CUSTOM}"
export WANDB_CONFIG_DIR="${WANDB_DIR_CUSTOM}"
export WANDB_DATA_DIR="${WANDB_DIR_CUSTOM}"
export WANDB_ARTIFACT_DIR="${WANDB_DIR_CUSTOM}"
export WANDB_ARTIFACT_LOCATION="${WANDB_DIR_CUSTOM}"
mkdir -p "${WANDB_DIR_CUSTOM}"
echo "REPRO_CONFIG repo=${BASE_PATH} model=${MODEL_PATH} data=${TRAIN_FILE} python=${PYTHON_BIN} wandb_mode=${WANDB_MODE}"
echo "REPRO_CONFIG attn_implementation=${ATTN_IMPLEMENTATION}"
echo "REPRO_CONFIG use_remove_padding=${USE_REMOVE_PADDING}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "REPRO_CONFIG wandb_api_key=provided"
else
    echo "REPRO_CONFIG wandb_api_key=absent"
fi
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
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
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
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
    actor_rollout_ref.actor.fsdp_config.dtype=${DTYPE} \
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
    actor_rollout_ref.rollout.dtype=${DTYPE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
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
    reward_model.reward_manager=rllm_skywork \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.reward_type=v2_math-verify-01 \
    trainer.total_epochs=50 \
    trainer.total_training_steps=${total_training_steps} \
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
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    "${extra_args[@]}" 2>&1 | tee "${LOG_DIR}/$(date +%m%d-%H%M%S).log"
