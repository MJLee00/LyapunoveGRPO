#!/usr/bin/env bash
# Roihu environment/GPU smoke test. This is intentionally capped at five minutes.
# The full reproduction is submitted separately to gpumedium after this check.
#SBATCH --job-name=dcmdp-smoke
#SBATCH --account=project_2008161
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=212G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/project_2015042/mingyl/LyapunoveGRPO
ENV_ROOT=/projappl/project_2015042/mingyl/LyapunoveGRPO
IMAGE=/scratch/project_2015042/renpei/rlsd_runtime/vllm-0.11.0-arm64.sif
RUN_ROOT="${PROJECT_ROOT}/run/smoke/${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/stdout.log") 2> >(tee "${RUN_ROOT}/stderr.log" >&2)

echo "SMOKE_START job=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ -n "${HF_TOKEN:-}" ]; then
    export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
fi

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    --env PYTHONPATH=/envroot/env/site:/workspace/verl \
    "${IMAGE}" \
    python3 /workspace/verl/scripts/repro/check_versions.py

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    --env PYTHONPATH=/envroot/env/site:/workspace/verl \
    "${IMAGE}" \
    python3 /workspace/verl/scripts/repro/validate_dcmdp_setup.py --repo-root /workspace/verl

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    --env PYTHONPATH=/envroot/env/site:/workspace/verl \
    "${IMAGE}" \
    python3 /workspace/verl/scripts/repro/check_training_import.py

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${ENV_ROOT}:/envroot" \
    --env PYTHONPATH=/envroot/env/site:/workspace/verl \
    "${IMAGE}" \
    python3 -m verl.trainer.main_ppo --help >/dev/null
echo "CONFIG_DEFAULTS=PASS"

touch "${RUN_ROOT}/COMPLETED"
echo "SMOKE_COMPLETED=1 job=${SLURM_JOB_ID}"
