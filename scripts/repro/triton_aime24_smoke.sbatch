#!/bin/bash -l
# Triton-only environment/GPU smoke gate for the official CMDP launcher.
# This is deliberately a one-H200, short debug run; it is not a paper result.
# The paper comparison must use the same launcher with NNODES=1, NGPUS_PER_NODE=8
# after this gate and after the Triton environment has been verified.
# Do not submit until `aalto_users` and the H200 short partition are confirmed.
# The remote project and log directories must exist before sbatch opens these paths.
#SBATCH --job-name=aime24-triton-smoke
#SBATCH --account=aalto_users
# ellis is the requested H200 queue; it shares the H200 node pool with short.
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Qwen3-8B actor initialization follows the official FP32-parameter/BF16-
# mixed-precision path and peaks above 96G CPU RAM before FSDP is established.
# Reserve enough CPU memory without changing GPU count or algorithm settings.
#SBATCH --mem=256G
#SBATCH --gpus=h200:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/work/lim7/LyapunoveGRPO/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/work/lim7/LyapunoveGRPO/logs/slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lim7/LyapunoveGRPO}"
JOB_ID="${SLURM_JOB_ID:-local}"
METHOD="${METHOD:-dcmdp}"
RUN_ROOT="${PROJECT_ROOT}/run/aime24-triton-smoke-${METHOD}-qwen3-8b-${JOB_ID}"
SHARED_HF_CACHE="${SHARED_HF_CACHE:-/scratch/shareddata/dldata/huggingface-hub-cache}"
QWEN3_MODEL_SNAPSHOT="${QWEN3_MODEL_SNAPSHOT:-${SHARED_HF_CACHE}/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5}"
MODEL_PATH="${MODEL_PATH:-${QWEN3_MODEL_SNAPSHOT}}"
RUNTIME_PYTHON_DIR="${RUNTIME_PYTHON_DIR:-${PROJECT_ROOT}/runtime/python}"

case "${METHOD}" in
    grpo) LAGRANGIAN_ENABLE=false ;;
    dcmdp) LAGRANGIAN_ENABLE=true ;;
    *) echo "ERROR: METHOD must be grpo or dcmdp, got ${METHOD}" >&2; exit 2 ;;
esac

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_ROOT}/hydra" "${RUN_ROOT}/hf"
exec > >(tee "${RUN_ROOT}/stdout.log") 2> >(tee "${RUN_ROOT}/stderr.log" >&2)

# Triton's generic module currently resolves to 2026.1 (vLLM 0.19.1); the
# paper-compatible environment is explicitly pinned to 2025.4 (vLLM 0.11.0).
module load scicomp-llm-env/2025.4
export BASE_PATH="${PROJECT_ROOT}"
export MODEL_BASE_PATH="${PROJECT_ROOT}/models"
export MODEL_PATH
export HF_HOME="${RUN_ROOT}/hf"
export HF_HUB_CACHE="${SHARED_HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${RUNTIME_PYTHON_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export RAY_DATA_HOME="${RUN_ROOT}"
export HYDRA_RUN_DIR="${RUN_ROOT}/hydra"
export WANDB_MODE=offline
# The cluster runtime's vendored W&B 0.30.0 is missing its optional
# OpenTelemetry dependencies and scratch quota is full. Console-only logging
# preserves the training algorithm without requiring credentials or network I/O.
export DISABLE_WANDB=true
export EXPERIMENT_NAME="triton_qwen3-8b_dapo2080_aime24_${METHOD}_smoke_v0.7.0dev"
export TRAIN_FILE="${PROJECT_ROOT}/data/train/dapo_train_qwen3-8b_20_80_6170.parquet"
export TEST_FILE="['${PROJECT_ROOT}/data/eval/aime24_32.parquet']"
export DATA_TYPE=dapo_2080
export DTYPE=bfloat16
export NNODES=1
export NGPUS_PER_NODE=1

# Keep the smoke bounded while exercising the actual official training path.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-1}"
export MAX_RESPONSE_LENGTH=2048
export TRAIN_PROMPT_BSZ=16
export N_RESP_PER_PROMPT=2
export TRAIN_PROMPT_MINI_BSZ=4
export GPU_MEMORY_UTILIZATION=0.30
export ENFORCE_EAGER=true
export USE_TORCH_COMPILE=false
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=32
export OFFLOAD_POLICY=true

# Keep the official BF16/FlashAttention path explicit. If the environment lacks
# flash-attn, fail the gate instead of silently producing a mismatched run.
export ATTN_IMPLEMENTATION=flash_attention_2
export USE_REMOVE_PADDING=true
export GRPO_LAGRANGIAN_ENABLE="${LAGRANGIAN_ENABLE}"
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1.0
export GRPO_LAGRANGIAN_BUFFER=1
export GRPO_LAGRANGIAN_Z=1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "MODEL_NOT_FOUND path=${MODEL_PATH}" >&2
    exit 3
fi

echo "TRITON_SMOKE_START job=${JOB_ID} method=${METHOD} host=$(hostname)"
echo "TRITON_SMOKE_CONFIG launcher=examples/CMDP/GRPO_qwen3_8b_lag_v5.sh model=${MODEL_PATH} data=${TRAIN_FILE} eval=${TEST_FILE}"
echo "TRITON_SMOKE_CONFIG steps=${TOTAL_TRAINING_STEPS} gpus=${CUDA_VISIBLE_DEVICES:-unset} dtype=${DTYPE} attn=${ATTN_IMPLEMENTATION}"
scontrol show job "${JOB_ID}" || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import importlib.metadata as metadata
import torch

required = {"vllm": "0.11.0"}
for name, expected in required.items():
    actual = metadata.version(name)
    if actual != expected:
        raise SystemExit(f"VERSION_MISMATCH {name}={actual} expected={expected}")
try:
    import flash_attn  # noqa: F401
except Exception as exc:
    raise SystemExit(f"FLASH_ATTN_IMPORT_FAILED {exc}") from exc
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA_NOT_VISIBLE")
print(f"TRITON_ENV_OK torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
PY

python scripts/repro/validate_dcmdp_setup.py --repo-root "${PROJECT_ROOT}"
python scripts/repro/check_training_import.py
python -m verl.trainer.main_ppo --help >/dev/null

bash "${PROJECT_ROOT}/examples/CMDP/GRPO_qwen3_8b_lag_v5.sh"

touch "${RUN_ROOT}/COMPLETED"
echo "TRITON_SMOKE_COMPLETED=1 job=${JOB_ID} method=${METHOD}"
