#!/bin/bash -l
# Paper-aligned AIME24 comparison on one Triton H200 node.
# The method switch is the only algorithmic difference: vanilla GRPO versus
# the repository's DC-GRPO/Lagrangian path.  The official CMDP launcher owns
# the training arguments; this wrapper only supplies Triton resources, paths,
# and the pinned vLLM environment.
#
# Submit one method at a time:
#   sbatch --parsable --export=ALL,METHOD=dcmdp scripts/repro/triton_aime24_paper_8gpu.sbatch
#   sbatch --parsable --export=ALL,METHOD=grpo  scripts/repro/triton_aime24_paper_8gpu.sbatch

#SBATCH --job-name=aime24-triton-8gpu
#SBATCH --account=aalto_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=h200:8
#SBATCH --mem=1024G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/work/lim7/LyapunoveGRPO/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/work/lim7/LyapunoveGRPO/logs/slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/work/lim7/LyapunoveGRPO}"
JOB_ID="${SLURM_JOB_ID:-local}"
METHOD="${METHOD:-dcmdp}"
RUNTIME_PYTHON_DIR="${RUNTIME_PYTHON_DIR:-${PROJECT_ROOT}/runtime/python}"
SHARED_HF_CACHE="${SHARED_HF_CACHE:-/scratch/shareddata/dldata/huggingface-hub-cache}"
QWEN3_MODEL_SNAPSHOT="${QWEN3_MODEL_SNAPSHOT:-${SHARED_HF_CACHE}/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5}"

case "${METHOD}" in
    grpo) LAGRANGIAN_ENABLE=false ;;
    dcmdp) LAGRANGIAN_ENABLE=true ;;
    *) echo "ERROR: METHOD must be grpo or dcmdp, got ${METHOD}" >&2; exit 2 ;;
esac

# Every run is immutable and job-qualified.  No existing checkpoint/result or
# log directory is reused.
RUN_ROOT="${PROJECT_ROOT}/run/aime24-paper-8gpu-${METHOD}-qwen3-8b-${JOB_ID}"
mkdir -p "${PROJECT_ROOT}/logs" "${RUN_ROOT}/hydra" "${RUN_ROOT}/hf" \
    "${RUN_ROOT}/cache/torchinductor" "${RUN_ROOT}/cache/triton"
exec > >(tee "${RUN_ROOT}/stdout.log") 2> >(tee "${RUN_ROOT}/stderr.log" >&2)

module load scicomp-llm-env/2025.4

export BASE_PATH="${PROJECT_ROOT}"
export MODEL_PATH="${QWEN3_MODEL_SNAPSHOT}"
export MODEL_BASE_PATH="${PROJECT_ROOT}/models"
export RUNTIME_PYTHON_DIR
export PYTHONPATH="${RUNTIME_PYTHON_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export HF_HOME="${RUN_ROOT}/hf"
export HF_HUB_CACHE="${SHARED_HF_CACHE}/hub"
export HF_DATASETS_CACHE="${RUN_ROOT}/hf/datasets"
export RAY_DATA_HOME="${RUN_ROOT}"
export HYDRA_RUN_DIR="${RUN_ROOT}/hydra"
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${RUN_ROOT}/cache/triton"
export WANDB_MODE=offline
# The pinned cluster environment has no usable OpenTelemetry/W&B network
# backend. Console-only logging preserves the experiment and avoids secrets.
export DISABLE_WANDB=true

export DATA_TYPE=dapo_2080
export TRAIN_FILE="${PROJECT_ROOT}/data/train/dapo_train_qwen3-8b_20_80_6170.parquet"
export TEST_FILE="['${PROJECT_ROOT}/data/eval/aime24_32.parquet']"
export DTYPE=bfloat16
export NNODES=1
export NGPUS_PER_NODE=8
export OFFLOAD=false
export OFFLOAD_POLICY=false
export USE_TORCH_COMPILE=true
export ATTN_IMPLEMENTATION=flash_attention_2
export USE_REMOVE_PADDING=true
export GPU_MEMORY_UTILIZATION=0.80
export ENFORCE_EAGER=false
export MAX_RESPONSE_LENGTH=8192
export TRAIN_PROMPT_BSZ=64
export N_RESP_PER_PROMPT=8
export TRAIN_PROMPT_MINI_BSZ=16
export PPO_MAX_TOKEN_LEN_PER_GPU=36864
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
export MAX_NUM_BATCHED_TOKENS=36864
export MAX_NUM_SEQS=1024
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-301}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_CKPT=false
export EXPERIMENT_NAME="triton_qwen3-8b_dapo2080_aime24_${METHOD}_paper_v0.7.0dev"

echo "AIME24_TRITON_PAPER_START job=${JOB_ID} method=${METHOD} host=$(hostname)"
echo "AIME24_TRITON_PAPER_CONFIG model=${MODEL_PATH} train=${TRAIN_FILE} eval=${TEST_FILE}"
echo "AIME24_TRITON_PAPER_CONFIG steps=${TOTAL_TRAINING_STEPS} nodes=${NNODES} gpus=${NGPUS_PER_NODE} dtype=${DTYPE}"
scontrol show job "${JOB_ID}" || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "MODEL_NOT_FOUND path=${MODEL_PATH}" >&2
    exit 3
fi
if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${PROJECT_ROOT}/data/eval/aime24_32.parquet" ]; then
    echo "DATA_NOT_FOUND train=${TRAIN_FILE} eval=${PROJECT_ROOT}/data/eval/aime24_32.parquet" >&2
    exit 4
fi

python - <<'PY'
import importlib.metadata as metadata
import torch

expected = {"vllm": "0.11.0"}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise SystemExit(f"VERSION_MISMATCH {name}={actual} expected={version}")
try:
    import flash_attn  # noqa: F401
except Exception as exc:
    raise SystemExit(f"FLASH_ATTN_IMPORT_FAILED {exc}") from exc
if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit(f"CUDA_NOT_VISIBLE count={torch.cuda.device_count()}")
print(f"TRITON_ENV_OK torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
PY

python scripts/repro/validate_dcmdp_setup.py --repo-root "${PROJECT_ROOT}"
python scripts/repro/check_training_import.py

# The official paper launcher supplies the optimizer, rollout, evaluator, and
# DC-GRPO parameters.  The only method difference is the Lagrangian enable bit.
export GRPO_LAGRANGIAN_ENABLE="${LAGRANGIAN_ENABLE}"
bash "${PROJECT_ROOT}/examples/CMDP/GRPO_qwen3_8b_lag_v5.sh"

touch "${RUN_ROOT}/COMPLETED"
echo "AIME24_TRITON_PAPER_COMPLETED=1 job=${JOB_ID} method=${METHOD}"
