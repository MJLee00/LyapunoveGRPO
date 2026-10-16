---
name: ubiquant-rl-training
description: Set up VERL RL training with the published torressliu Docker image and run ubiquant GRPO scripts. Use when users ask to prepare the Ubiquant Docker environment, run GRPO_qwen3_8b_fp8_lag_v5.sh or GRPO_r1_1p5b_fp8_lag_v5.sh, or troubleshoot missing model/data path issues.
---

# Ubiquant RL Training

## Purpose

Use this skill to run Ubiquant VERL RL training quickly with the published Docker image and execute both maintained scripts in sequence:

- `ubiquant/GRPO_qwen3_8b_fp8_lag_v5.sh`
- `ubiquant/GRPO_r1_1p5b_fp8_lag_v5.sh`

## Inputs To Collect First

Collect these values before execution:

- `MODEL_PATH` (required): absolute path to downloaded model weights
- `TRAIN_FILE` (required): parquet file path for training
- `TEST_FILE` (required): eval parquet list string or single file list
- `NGPUS_PER_NODE` (optional, default `8`)
- `NNODES` (optional, default `1`)
- run mode: execute both scripts sequentially after environment setup

Important: model weights should be downloaded in advance. Do not assume automatic online model download.

## Standard Workflow

1. From repository root, pull and start container:

```bash
docker pull torressliu/verl:vllm011-ubiquant-latest
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v "$(pwd)":/workspace/verl --name verl-ubiquant torressliu/verl:vllm011-ubiquant-latest sleep infinity
docker start verl-ubiquant
docker exec -it verl-ubiquant bash
```

2. Inside container, install editable package:

```bash
cd /workspace/verl
pip3 install --no-deps -e .
```

3. Export runtime variables:

```bash
export BASE_PATH=/workspace/verl
export MODEL_PATH=/abs/path/to/model
export TRAIN_FILE=/abs/path/to/train.parquet
export TEST_FILE="['/abs/path/to/aime24_32.parquet']"
export NGPUS_PER_NODE=8
export NNODES=1
```

4. Ensure model weights are available at `MODEL_PATH`:

```bash
ls -lah "${MODEL_PATH}"
```

If the directory does not exist, download first (example):

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /abs/path/to/DeepSeek-R1-Distill-Qwen-1.5B
export MODEL_PATH=/abs/path/to/DeepSeek-R1-Distill-Qwen-1.5B
```

5. Run both scripts sequentially:

```bash
bash ubiquant/GRPO_qwen3_8b_fp8_lag_v5.sh
bash ubiquant/GRPO_r1_1p5b_fp8_lag_v5.sh
```

## Validation Checklist

- Container is running: `docker ps` contains `verl-ubiquant`
- Code import works: `python3 -c "import verl; print('ok')"`
- Model path is valid: `ls -lah "${MODEL_PATH}"` succeeds
- Training started: script log shows `ray job submit` and `python3 -m verl.trainer.main_ppo`
- No path errors for `MODEL_PATH`, `TRAIN_FILE`, or eval files

## Troubleshooting

- **Model path not found**: verify `MODEL_PATH` points to existing local weights; re-export and rerun.
- **Data file not found**: verify parquet paths and permissions; use absolute paths.
- **No GPU inside container**: verify NVIDIA Container Toolkit and rerun with `--gpus all`.
- **Ray submission fails**: confirm Ray runtime in image and that local endpoint used by script is reachable.

## Safety Notes

- Do not commit real `WANDB_API_KEY` values.
- Prefer environment overrides over editing script defaults.
