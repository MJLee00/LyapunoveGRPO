# Ubiquant RL 训练快速开始

本目录用于说明：如何基于我们发布的 Docker 镜像快速搭建 VERL 训练环境，并运行以下两个 RL 训练脚本：

- `ubiquant/GRPO_qwen3_8b_fp8_lag_v5.sh`
- `ubiquant/GRPO_r1_1p5b_fp8_lag_v5.sh`

## 1) 使用我们发布的镜像

- Docker Hub: https://hub.docker.com/r/torressliu/verl
- 推荐 tag: `torressliu/verl:vllm011-ubiquant-latest`

该镜像已包含所需训练环境，通常不需要再单独安装 vLLM 或 SGLang。

## 2) 启动容器并安装当前代码

请在仓库根目录执行（确保当前目录包含 `ubiquant/`、`verl/`、`data/` 等）：

```bash
docker pull torressliu/verl:vllm011-ubiquant-latest
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v "$(pwd)":/workspace/verl --name verl-ubiquant torressliu/verl:vllm011-ubiquant-latest sleep infinity
docker start verl-ubiquant
docker exec -it verl-ubiquant bash
```

进入容器后执行：

```bash
cd /workspace/verl
```

## 3) 运行训练脚本

两个脚本依赖以下关键路径，请先按服务器环境配置：

- `BASE_PATH`：代码根目录（容器内通常为 `/workspace/verl`）
- `MODEL_PATH`：本地模型目录
- `TRAIN_FILE` / `TEST_FILE`：训练和评估 parquet 数据
- `NNODES` / `NGPUS_PER_NODE`：分布式训练资源


说明：建议提前下载好模型权重并设置到 `MODEL_PATH`。这两个脚本默认按本地路径读取模型，不保证自动在线下载。

运行 8B 脚本：

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export EXPERIMENT_NAME=v5_qwen3-8b_fp8_bf16
bash ubiquant/GRPO_qwen3_8b_fp8_lag_v5.sh
```

运行 R1-1.5B 脚本：

```bash
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
export EXPERIMENT_NAME=v5_r1-1p5b_fp8_bf16
bash ubiquant/GRPO_r1_1p5b_fp8_lag_v5.sh
```

## 备注

- 脚本通过 `ray job submit` 提交任务，请确保容器内 Ray 可用。
- 脚本中的默认路径很多是占位或集群路径，不修改会导致模型或数据找不到。
