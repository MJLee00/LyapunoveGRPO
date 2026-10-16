#!/usr/bin/env bash
# Paper-aligned DC-GRPO adaptation: Qwen3-8B on one Roihu GH200.
# Full DAPO-Math-17K data is used; no debug row/step cap is applied here.
#SBATCH --job-name=dcmdp-qwen3-8b
#SBATCH --account=project_2008161
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=212G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/project_2015042/mingyl/LyapunoveGRPO
ENV_ROOT=/projappl/project_2015042/mingyl/LyapunoveGRPO
IMAGE=/scratch/project_2015042/renpei/rlsd_runtime/vllm-0.11.0-arm64.sif
RUN_ROOT="${PROJECT_ROOT}/run/dcmdp-qwen3-8b-${SLURM_JOB_ID}"
CACHE_ROOT="${RUN_ROOT}/cache"
RAY_ROOT="${PROJECT_ROOT}/r/${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}" "${RAY_ROOT}" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${ENV_ROOT}/hf/hub" "${ENV_ROOT}/hf/datasets"
exec > >(tee "${RUN_ROOT}/stdout.log") 2> >(tee "${RUN_ROOT}/stderr.log" >&2)

echo "REPRODUCTION_START job=${SLURM_JOB_ID} host=$(hostname)"
echo "REPRODUCTION_ROOT=${PROJECT_ROOT}"
echo "REPRODUCTION_ENV_ROOT=${ENV_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Secrets are forwarded only through the process environment and are never printed.
if [ -n "${HF_TOKEN:-}" ]; then
    export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
fi

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    --env PYTHONPATH=/envroot/env/site:/workspace/verl \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env GCC=/usr/bin/gcc \
    --env CUDAHOSTCXX=/usr/bin/g++ \
    --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env TMPDIR=/workspace/r/${SLURM_JOB_ID} \
    --env RAY_TMPDIR=/workspace/r/${SLURM_JOB_ID} \
    --env TORCHINDUCTOR_CACHE_DIR=/workspace/run/dcmdp-qwen3-8b-${SLURM_JOB_ID}/cache/torchinductor \
    --env TRITON_CACHE_DIR=/workspace/run/dcmdp-qwen3-8b-${SLURM_JOB_ID}/cache/triton \
    --env HF_HOME=/envroot/hf \
    --env HF_HUB_CACHE=/envroot/hf/hub \
    --env HF_DATASETS_CACHE=/envroot/hf/datasets \
    --env BASE_PATH=/workspace/verl \
    --env MODEL_BASE_PATH=/envroot/hf/models \
    --env MODEL_PATH=Qwen/Qwen3-8B \
    --env TRAIN_FILE=/workspace/verl/data/train/dapo_train.parquet \
    --env DATA_TYPE=dapo \
    --env RAY_DATA_HOME=/workspace/run/dcmdp-qwen3-8b-${SLURM_JOB_ID} \
    --env EXPERIMENT_NAME=roihu_qwen3-8b_dapo_dcmdp_v0.7.0dev_torch2.8_vllm0.11 \
    --env PYTHON_BIN=python3 \
    --env CONDA_HOME= \
    --env NNODES=1 \
    --env NGPUS_PER_NODE=1 \
    --env OFFLOAD=false \
    --env OFFLOAD_POLICY=true \
    --env SAVE_CKPT=true \
    --env WANDB_MODE=offline \
    --env ATTN_IMPLEMENTATION=sdpa \
    --env USE_REMOVE_PADDING=false \
    --env MAX_RESPONSE_LENGTH=2048 \
    --env TRAIN_PROMPT_BSZ=16 \
    --env N_RESP_PER_PROMPT=2 \
    --env TRAIN_PROMPT_MINI_BSZ=4 \
    --env GPU_MEMORY_UTILIZATION=0.20 \
    --env ENFORCE_EAGER=true \
    --env PPO_MAX_TOKEN_LEN_PER_GPU=4096 \
    --env LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
    --env MAX_NUM_BATCHED_TOKENS=4096 \
    --env MAX_NUM_SEQS=32 \
    --env GRPO_LAGRANGIAN_C=0.0037 \
    --env GRPO_LAGRANGIAN_LAMBDA_INIT=0.1 \
    --env GRPO_LAGRANGIAN_LAMBDA_LR=1.0 \
    --env GRPO_LAGRANGIAN_BUFFER=1 \
    --env GRPO_LAGRANGIAN_Z=1 \
    --env GRPO_LAGRANGIAN_RATIO=token_prob_diff \
    --env NORMALIZE_MODE=absneg_0.0_none \
    "${IMAGE}" \
    python3 /workspace/verl/scripts/repro/check_training_import.py

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    --env PYTHONPATH=/envroot/env/site:/workspace/verl \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env GCC=/usr/bin/gcc \
    --env CUDAHOSTCXX=/usr/bin/g++ \
    --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env TMPDIR=/workspace/r/${SLURM_JOB_ID} \
    --env RAY_TMPDIR=/workspace/r/${SLURM_JOB_ID} \
    --env TORCHINDUCTOR_CACHE_DIR=/workspace/run/dcmdp-qwen3-8b-${SLURM_JOB_ID}/cache/torchinductor \
    --env TRITON_CACHE_DIR=/workspace/run/dcmdp-qwen3-8b-${SLURM_JOB_ID}/cache/triton \
    --env HF_HOME=/envroot/hf \
    --env HF_HUB_CACHE=/envroot/hf/hub \
    --env HF_DATASETS_CACHE=/envroot/hf/datasets \
    --env BASE_PATH=/workspace/verl \
    --env MODEL_BASE_PATH=/envroot/hf/models \
    --env MODEL_PATH=Qwen/Qwen3-8B \
    --env TRAIN_FILE=/workspace/verl/data/train/dapo_train.parquet \
    --env DATA_TYPE=dapo \
    --env RAY_DATA_HOME=/workspace/run/dcmdp-qwen3-8b-${SLURM_JOB_ID} \
    --env EXPERIMENT_NAME=roihu_qwen3-8b_dapo_dcmdp_v0.7.0dev_torch2.8_vllm0.11 \
    --env PYTHON_BIN=python3 \
    --env CONDA_HOME= \
    --env NNODES=1 \
    --env NGPUS_PER_NODE=1 \
    --env OFFLOAD=false \
    --env OFFLOAD_POLICY=true \
    --env SAVE_CKPT=true \
    --env WANDB_MODE=offline \
    --env ATTN_IMPLEMENTATION=sdpa \
    --env USE_REMOVE_PADDING=false \
    --env MAX_RESPONSE_LENGTH=2048 \
    --env TRAIN_PROMPT_BSZ=16 \
    --env N_RESP_PER_PROMPT=2 \
    --env TRAIN_PROMPT_MINI_BSZ=4 \
    --env GPU_MEMORY_UTILIZATION=0.20 \
    --env ENFORCE_EAGER=true \
    --env PPO_MAX_TOKEN_LEN_PER_GPU=4096 \
    --env LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
    --env MAX_NUM_BATCHED_TOKENS=4096 \
    --env MAX_NUM_SEQS=32 \
    --env GRPO_LAGRANGIAN_C=0.0037 \
    --env GRPO_LAGRANGIAN_LAMBDA_INIT=0.1 \
    --env GRPO_LAGRANGIAN_LAMBDA_LR=1.0 \
    --env GRPO_LAGRANGIAN_BUFFER=1 \
    --env GRPO_LAGRANGIAN_Z=1 \
    --env GRPO_LAGRANGIAN_RATIO=token_prob_diff \
    --env NORMALIZE_MODE=absneg_0.0_none \
    "${IMAGE}" \
    bash /workspace/verl/examples/CMDP/GRPO_qwen3_8b_lag_v5.sh

touch "${RUN_ROOT}/COMPLETED"
echo "REPRODUCTION_COMPLETED=1 job=${SLURM_JOB_ID}"
