#!/usr/bin/env bash
# Paper-aligned AIME24 comparison on two Roihu nodes (4 GH200 GPUs per node).
# The method switch is the only algorithmic difference: GRPO vs DC-GRPO.
# Roihu uses SDPA/remove-padding=false because the pinned ARM64 image has no
# flash-attn module; all paper training, rollout, optimizer, and evaluation
# budgets remain at their README/Table-1 values.
# Submit one method at a time with --export=ALL,METHOD=dcmdp or grpo.
# This script is prepared for gpularge and must not be submitted until the
# account has confirmed gpularge/scalability access.
#SBATCH --job-name=aime24-paper-8gpu
#SBATCH --account=project_2008161
#SBATCH --partition=gpularge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=848G
#SBATCH --time=36:00:00
#SBATCH --output=/projappl/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.out
#SBATCH --error=/projappl/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/project_2015042/mingyl/LyapunoveGRPO
ENV_ROOT=/projappl/project_2015042/mingyl/LyapunoveGRPO
IMAGE=/scratch/project_2015042/renpei/rlsd_runtime/vllm-0.11.0-arm64.sif
METHOD="${METHOD:-dcmdp}"
METHOD_TAG="${METHOD}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-301}"
TEST_FREQ="${TEST_FREQ:-20}"

case "${METHOD}" in
    grpo) LAGRANGIAN_ENABLE=false ;;
    dcmdp) LAGRANGIAN_ENABLE=true ;;
    *) echo "ERROR: METHOD must be grpo or dcmdp, got ${METHOD}" >&2; exit 2 ;;
esac

RUNTIME_ROOT="${ENV_ROOT}/runtime"
RUN_ROOT="${RUNTIME_ROOT}/aime24-paper-8gpu-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}"
RAY_ROOT="${ENV_ROOT}/ray/${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}/cache/torchinductor" "${RUN_ROOT}/cache/triton" \
    "${RAY_ROOT}/head" "${RAY_ROOT}/worker" "${ENV_ROOT}/hf/hub" "${ENV_ROOT}/hf/datasets"
exec > >(tee "${RUN_ROOT}/stdout.log") 2> >(tee "${RUN_ROOT}/stderr.log" >&2)

echo "AIME24_PAPER_START job=${SLURM_JOB_ID} method=${METHOD_TAG} host=$(hostname)"
echo "AIME24_PAPER_CONFIG train=dapo_train_qwen3-8b_20_80_6170.parquet eval=aime24_32.parquet steps=${TOTAL_TRAINING_STEPS}"
echo "AIME24_PAPER_CONFIG nodes=${SLURM_JOB_NUM_NODES} gpus_per_node=${SLURM_GPUS_PER_NODE:-4}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

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
    --env GCC=/usr/bin/g++
    --env CUDAHOSTCXX=/usr/bin/g++
    --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    --env TMPDIR=/envroot/ray/${SLURM_JOB_ID}
    --env RAY_TMPDIR=/envroot/ray/${SLURM_JOB_ID}
    --env HF_HOME=/envroot/hf
    --env HF_HUB_CACHE=/envroot/hf/hub
    --env HF_DATASETS_CACHE=/envroot/hf/datasets
    --env BASE_PATH=/workspace/verl
    --env MODEL_BASE_PATH=/envroot/hf/models
    --env MODEL_PATH=Qwen/Qwen3-8B
    --env DTYPE=bfloat16
    --env TRAIN_FILE=/workspace/verl/data/train/dapo_train_qwen3-8b_20_80_6170.parquet
    --env TEST_FILE="['/workspace/verl/data/eval/aime24_32.parquet']"
    --env DATA_TYPE=dapo_2080
    --env RAY_DATA_HOME=/envroot/runtime/aime24-paper-8gpu-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}
    --env HYDRA_RUN_DIR=/envroot/runtime/aime24-paper-8gpu-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}/hydra
    --env EXPERIMENT_NAME=roihu_qwen3-8b_dapo2080_aime24_${METHOD_TAG}_paper_v0.7.0dev
    --env PYTHON_BIN=python3
    --env CONDA_HOME=
    --env NNODES=2
    --env NGPUS_PER_NODE=4
    --env OFFLOAD=false
    --env OFFLOAD_POLICY=false
    --env USE_TORCH_COMPILE=true
    --env SAVE_CKPT=false
    --env WANDB_MODE=offline
    --env ATTN_IMPLEMENTATION=sdpa
    --env USE_REMOVE_PADDING=false
    --env MAX_RESPONSE_LENGTH=8192
    --env TRAIN_PROMPT_BSZ=64
    --env N_RESP_PER_PROMPT=8
    --env TRAIN_PROMPT_MINI_BSZ=16
    --env GPU_MEMORY_UTILIZATION=0.70
    --env ENFORCE_EAGER=false
    --env GRPO_LAGRANGIAN_ENABLE=${LAGRANGIAN_ENABLE}
    --env GRPO_LAGRANGIAN_C=0.0037
    --env GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
    --env GRPO_LAGRANGIAN_LAMBDA_LR=1.0
    --env GRPO_LAGRANGIAN_BUFFER=1
    --env GRPO_LAGRANGIAN_Z=1
    --env GRPO_LAGRANGIAN_RATIO=token_prob_diff
    --env GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
    --env NORMALIZE_MODE=absneg_0.0_none
    --env TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
    --env TEST_FREQ=${TEST_FREQ}
)

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [ "${#NODES[@]}" -ne 2 ]; then
    echo "ERROR: expected exactly two allocated nodes, got ${#NODES[@]}" >&2
    exit 3
fi
HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" hostname -I | awk '{print $1}')"
WORKER_IP="$(srun --nodes=1 --ntasks=1 --nodelist="${WORKER_NODE}" hostname -I | awk '{print $1}')"
RAY_ADDRESS="${HEAD_IP}:6379"
export RAY_ADDRESS
echo "RAY_HEAD=${HEAD_NODE} ${HEAD_IP}:6379"
echo "RAY_WORKER=${WORKER_NODE} ${WORKER_IP}:6379"

cleanup() {
    set +e
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${WORKER_NODE}" \
        apptainer exec "${COMMON_ENV[@]}" "${IMAGE}" ray stop --force >/dev/null 2>&1
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
        apptainer exec "${COMMON_ENV[@]}" "${IMAGE}" ray stop --force >/dev/null 2>&1
    [ -n "${HEAD_RAY_PID:-}" ] && kill "${HEAD_RAY_PID}" >/dev/null 2>&1 || true
    [ -n "${WORKER_RAY_PID:-}" ] && kill "${WORKER_RAY_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    apptainer exec "${COMMON_ENV[@]}" "${IMAGE}" \
    ray start --head --node-ip-address="${HEAD_IP}" --port=6379 \
    --num-cpus=288 --num-gpus=4 --disable-usage-stats --temp-dir="/envroot/ray/${SLURM_JOB_ID}/head" --block \
    >"${RUN_ROOT}/ray-head.log" 2>&1 &
HEAD_RAY_PID=$!

srun --overlap --nodes=1 --ntasks=1 --nodelist="${WORKER_NODE}" \
    apptainer exec "${COMMON_ENV[@]}" "${IMAGE}" \
    ray start --address="${RAY_ADDRESS}" --node-ip-address="${WORKER_IP}" \
    --num-cpus=288 --num-gpus=4 --disable-usage-stats --temp-dir="/envroot/ray/${SLURM_JOB_ID}/worker" --block \
    >"${RUN_ROOT}/ray-worker.log" 2>&1 &
WORKER_RAY_PID=$!

CLUSTER_READY=false
for attempt in $(seq 1 60); do
    if srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
        apptainer exec "${COMMON_ENV[@]}" --env RAY_ADDRESS="${RAY_ADDRESS}" "${IMAGE}" \
        python3 -c 'import os, ray; ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n["Alive"]]; print(f"RAY_ALIVE_NODES={len(nodes)}"); ray.shutdown(); raise SystemExit(0 if len(nodes) == 2 else 1)' \
        >/dev/null 2>&1; then
        CLUSTER_READY=true
        break
    fi
    sleep 5
done
if [ "${CLUSTER_READY}" != true ]; then
    echo "ERROR: Ray did not report two live nodes" >&2
    exit 4
fi
echo "RAY_CLUSTER_READY=1 nodes=2 gpus=8"

srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    apptainer exec "${COMMON_ENV[@]}" --env RAY_ADDRESS="${RAY_ADDRESS}" "${IMAGE}" \
    python3 /workspace/verl/scripts/repro/check_training_import.py

srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    apptainer exec "${COMMON_ENV[@]}" --env RAY_ADDRESS="${RAY_ADDRESS}" "${IMAGE}" \
    bash /workspace/verl/examples/CMDP/GRPO_qwen3_8b_lag_v5.sh

touch "${RUN_ROOT}/COMPLETED"
echo "AIME24_PAPER_COMPLETED=1 job=${SLURM_JOB_ID} method=${METHOD_TAG}"
