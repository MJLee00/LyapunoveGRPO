#!/usr/bin/env bash
# Matched AIME24 pilot: vanilla GRPO versus DC-GRPO on Qwen3-8B.
# Training data stays the official DAPO-Math-17K; AIME24 is the only validation set.
# Submit with --export=ALL,METHOD=grpo or --export=ALL,METHOD=dcmdp.
# The default 20-step pilot is intentionally bounded for one GH200; set
# TOTAL_TRAINING_STEPS=301 for the paper-length schedule.
#SBATCH --job-name=aime24-qwen3-8b
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
METHOD="${METHOD:-dcmdp}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
TEST_FREQ="${TEST_FREQ:-10}"

case "${METHOD}" in
    grpo)
        LAGRANGIAN_ENABLE=false
        METHOD_TAG=grpo
        ;;
    dcmdp)
        LAGRANGIAN_ENABLE=true
        METHOD_TAG=dcmdp
        ;;
    *)
        echo "ERROR: METHOD must be grpo or dcmdp, got ${METHOD}" >&2
        exit 2
        ;;
esac

RUN_ROOT="${PROJECT_ROOT}/run/aime24-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}"
CACHE_ROOT="${RUN_ROOT}/cache"
mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/r/${SLURM_JOB_ID}" \
    "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" \
    "${ENV_ROOT}/hf/hub" "${ENV_ROOT}/hf/datasets"
exec > >(tee "${RUN_ROOT}/stdout.log") 2> >(tee "${RUN_ROOT}/stderr.log" >&2)

echo "AIME24_COMPARE_START job=${SLURM_JOB_ID} method=${METHOD_TAG} host=$(hostname)"
echo "AIME24_COMPARE_CONFIG train=dapo_train.parquet eval=aime24_32.parquet steps=${TOTAL_TRAINING_STEPS} test_freq=${TEST_FREQ}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ -n "${HF_TOKEN:-}" ]; then
    export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
fi

COMMON_ENV=(
    --env PYTHONPATH=/envroot/env/site:/workspace/verl
    --env CC=/usr/bin/gcc
    --env CXX=/usr/bin/g++
    --env GCC=/usr/bin/gcc
    --env CUDAHOSTCXX=/usr/bin/g++
    --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    --env TMPDIR=/workspace/r/${SLURM_JOB_ID}
    --env RAY_TMPDIR=/workspace/r/${SLURM_JOB_ID}
    --env TORCHINDUCTOR_CACHE_DIR=/workspace/run/aime24-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}/cache/torchinductor
    --env TRITON_CACHE_DIR=/workspace/run/aime24-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}/cache/triton
    --env HF_HOME=/envroot/hf
    --env HF_HUB_CACHE=/envroot/hf/hub
    --env HF_DATASETS_CACHE=/envroot/hf/datasets
    --env BASE_PATH=/workspace/verl
    --env MODEL_BASE_PATH=/envroot/hf/models
    --env MODEL_PATH=Qwen/Qwen3-8B
    --env TRAIN_FILE=/workspace/verl/data/train/dapo_train.parquet
    --env TEST_FILE="['/workspace/verl/data/eval/aime24_32.parquet']"
    --env DATA_TYPE=dapo
    --env RAY_DATA_HOME=/workspace/run/aime24-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}
    --env EXPERIMENT_NAME=roihu_qwen3-8b_dapo_aime24_${METHOD_TAG}_v0.7.0dev_torch2.8_vllm0.11
    --env PYTHON_BIN=python3
    --env CONDA_HOME=
    --env NNODES=1
    --env NGPUS_PER_NODE=1
    --env OFFLOAD=false
    --env OFFLOAD_POLICY=true
    --env SAVE_CKPT=false
    --env WANDB_MODE=offline
    --env ATTN_IMPLEMENTATION=sdpa
    --env USE_REMOVE_PADDING=false
    # Single-GH200 adaptation: leave headroom for actor log-prob/entropy logits.
    # Single-GH200 gate: keep ample headroom for FSDP actor-update peaks.
    --env MAX_RESPONSE_LENGTH=2048
    --env TRAIN_PROMPT_BSZ=16
    --env N_RESP_PER_PROMPT=2
    --env TRAIN_PROMPT_MINI_BSZ=4
    --env GPU_MEMORY_UTILIZATION=0.20
    --env ENFORCE_EAGER=true
    --env PPO_MAX_TOKEN_LEN_PER_GPU=4096
    --env LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
    --env MAX_NUM_BATCHED_TOKENS=4096
    --env MAX_NUM_SEQS=32
    --env GRPO_LAGRANGIAN_ENABLE=${LAGRANGIAN_ENABLE}
    --env GRPO_LAGRANGIAN_C=0.0037
    --env GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
    --env GRPO_LAGRANGIAN_LAMBDA_LR=1.0
    --env GRPO_LAGRANGIAN_BUFFER=1
    --env GRPO_LAGRANGIAN_Z=1
    --env GRPO_LAGRANGIAN_RATIO=token_prob_diff
    --env NORMALIZE_MODE=absneg_0.0_none
    --env TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
    --env TEST_FREQ=${TEST_FREQ}
)

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    "${COMMON_ENV[@]}" \
    "${IMAGE}" \
    python3 /workspace/verl/scripts/repro/check_training_import.py

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    "${COMMON_ENV[@]}" \
    "${IMAGE}" \
    bash /workspace/verl/examples/CMDP/GRPO_qwen3_8b_lag_v5.sh

touch "${RUN_ROOT}/COMPLETED"
echo "AIME24_COMPARE_COMPLETED=1 job=${SLURM_JOB_ID} method=${METHOD_TAG}"
