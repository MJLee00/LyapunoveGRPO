- fp8使用环境需要：
    - vllm=0.11.0：需要安装deep-gemm，然后直接运行即可（开头输出可以看到会加载deepgemm）
    - vllm>0.11.0：可以不安装deep-gemm（不安装的话会通过cutlass来做fp8）

# GRPO

eval
```bash
export GRPO_LAGRANGIAN_C=0.0050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh True 301
```
```bash
export EXPERIMENT_NAME=v5_qwen3-30ba3b_fp8_bf16
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh True 301
```

```bash
export GRPO_LAGRANGIAN_C=0.0050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh True 401
```
```bash
export EXPERIMENT_NAME=v5_qwen3-30ba3b_bf16
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh True 401
```

```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=3e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_ratio
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 240
```
```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=3e-1
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 180
```

```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=3e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh True 80
```
```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh True 80
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-30ba3b_fp8_bf16
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh True 80
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh True 60
```

## Qwen3-8B (paper)

### main (Tab.1, BF16)

DCMDP main (c=0.0037, lambda_init=0.1, lambda_lr=1.0)
baseline
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 220

export EXPERIMENT_NAME=v5_qwen3-8b_bf16
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 200

export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_ratio
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 280

export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 140
```

### ablation: lambda_init (fix c=0.0037, lambda_lr=1.0)

lambda_init=0.05
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 300

export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 200

export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.3
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 300
```

### ablation: lambda_lr (fix c=0.0037, lambda_init=0.1)

lambda_lr=5e-1
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 300

export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1.5e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 240
```

### ablation: c (fix lambda_init=0.1, lambda_lr=1.0)

c=0.0030
```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 280

export GRPO_LAGRANGIAN_C=0.0033
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 160

export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 220

export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 260

export GRPO_LAGRANGIAN_C=0.0043
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh True 280
```


train

## 5.7

```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=3e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh
```

## 5.6

```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=3e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_ratio
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.8
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_ratio
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=3e-1
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```


```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.03
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
export clean=1
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-30ba3b_fp8_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
export clean=1
bash ./examples/CMDP/GRPO_qwen3_30ba3b_fp8_lag_mega_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_ratio
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0100
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```



```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1.5e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.3
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0043
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0033
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```


## 5.1

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_4b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export EXPERIMENT_NAME=v5_qwen3-4b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_4b_lag_v5.sh
```

## 4.30

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-4b_filter
bash ./examples/CMDP/GRPO_qwen3_4b_filter.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=math_1460
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

## 4.29

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=math_1460
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export EXPERIMENT_NAME=v5_qwen3-1p7b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=math_1460
export clean=1
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=math_1460
bash ./examples/CMDP/GRPO_qwen3_4b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export EXPERIMENT_NAME=v5_qwen3-4b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=math_1460
export clean=1
bash ./examples/CMDP/GRPO_qwen3_4b_lag_v5.sh
```


```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

## 4.28

```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export EXPERIMENT_NAME=v5_qwen3-1p7b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-1p7b_filter
bash ./examples/CMDP/GRPO_qwen3_1p7b_filter.sh
```

## 4.27

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```


```bash
export GRPO_LAGRANGIAN_C=0.0047
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0047
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export NORMALIZE_MODE=none
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```

## 4.26

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```

## 4.25

```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign0
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=0
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=0
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign0
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=0
export DATA_TYPE=dapo
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-1p7b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=0
export DATA_TYPE=dapo
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

## 4.24

```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign0
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign0
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo
export clean=1
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
export DATA_TYPE=dapo
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-1p7b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
export DATA_TYPE=dapo
bash ./examples/CMDP/GRPO_qwen3_1p7b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=samesign
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.01
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.02
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-1
export GRPO_LAGRANGIAN_LAMBDA_MAX=0.05
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.02
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-1
export GRPO_LAGRANGIAN_LAMBDA_MAX=0.05
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```

## 4.19

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-30ba3b_filter
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_mega_filter.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-14b-base_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
export clean=1
bash ./examples/CMDP/GRPO_qwen3_14b_base_lag_mega_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0135
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=5e-1
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_r1-1p5b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export clean=1
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-30ba3b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
export clean=1
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_mega_v5.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v4.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b-base_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v5_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v5.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_r1-1p5b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

## 4.18

```bash
export GRPO_LAGRANGIAN_C=0.0037
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-14b-base_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_14b_base_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.013
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.013
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.01
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_logp_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=abs_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e0
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

## 4.17

```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v4.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=absneg_0.0_none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v4.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v4_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v4.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DATA_TYPE=dapo_2080
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```


```bash
export GRPO_LAGRANGIAN_C=0.0030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_filter
bash ./examples/CMDP/GRPO_qwen3_8b_filter.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b-base_filter
bash ./examples/CMDP/GRPO_qwen3_8b_base_filter.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

# 4.14

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=2e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v3.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.00009
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_prob_diff
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v3.sh
```

复现之前1.5B的最好结果：
grpo_r1_1p5b_lag_c-0.00040_z1_seq_geo_no_exp_dual_init0.5_lr1e1_min0.0_max1.0_bufmean_normsigmoid
https://wandb.ai/anitaleung/CMDP/runs/w9fdeafx
```bash
export GRPO_LAGRANGIAN_C=0.00040
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

# 4.13

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=clip_negative
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

# 4.12

```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=clip_negative
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v3.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=clip_negative
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.3
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-1
export GRPO_LAGRANGIAN_RATIO=seq_prob_diff_sum
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=clip_negative
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0035
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=token_prob_diff
export GRPO_LAGRANGIAN_PENALTY_LEVEL=token
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.3
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-1
export GRPO_LAGRANGIAN_RATIO=seq_prob_diff_sum
export GRPO_LAGRANGIAN_BUFFER_MODE=gt0_mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```


```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-30ba3b_base_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-30ba3b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_base_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```


```bash
export GRPO_LAGRANGIAN_C=-0.00040
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=clip_negative
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=-0.00040
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_r1-1p5b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_r1-1p5b_fp16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=5e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_fp16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

# 4.11

```bash
export GRPO_LAGRANGIAN_C=-0.00040
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=-0.00040
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_r1-1p5b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_r1-1p5b_fp16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-30ba3b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_fsdp_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-30ba3b_fp16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export NNODES=4
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_30ba3b_lag_fsdp_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.00030
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=5e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

# 4.10

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=5e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=5e1
export GRPO_LAGRANGIAN_LAMBDA_MAX=1e2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e3
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=none
export GRPO_LAGRANGIAN_APPLY_ON=adv
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=none
export EXPERIMENT_NAME=v2_qwen3-8b_fp16
export SAVE_ALL_ROLLOUT_DATA=true
export SAVE_CKPT=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_r1-1p5b_bf16_log
export SAVE_ALL_ROLLOUT_DATA=true
export COMPUTE_PROB_SHIFT=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_r1-1p5b_fp16_log
export SAVE_ALL_ROLLOUT_DATA=true
export COMPUTE_PROB_SHIFT=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_qwen3-8b_bf16_log
export SAVE_ALL_ROLLOUT_DATA=true
export COMPUTE_PROB_SHIFT=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_qwen3-8b_fp16_log
export SAVE_ALL_ROLLOUT_DATA=true
export COMPUTE_PROB_SHIFT=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```

## 4.9

```bash
export GRPO_LAGRANGIAN_C=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.05
export GRPO_LAGRANGIAN_LAMBDA_LR=2e0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_qwen3-8b_bf16_log
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_r1-1p5b_bf16_log
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

## 4.8

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_r1-1p5b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_r1_1p5b_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.00030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_qwen3-8b_bf16
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_qwen3-8b_fp16
export SAVE_ALL_ROLLOUT_DATA=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_8b_lag_v2.sh
```


```bash
export GRPO_LAGRANGIAN_C=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.00030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=5e0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export EXPERIMENT_NAME=v2_qwen3-8b-base_bf16
export SAVE_ALL_ROLLOUT_DATA=true
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export SAVE_ALL_ROLLOUT_DATA=true
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_8b_base_lag_v2.sh
```

```bash
export GRPO_LAGRANGIAN_C=-0.00030
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p85
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p85
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```


```bash
export GRPO_LAGRANGIAN_C=-0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_C=-0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```


## 4.5

```bash
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_ENABLE2=true
export GRPO_LAGRANGIAN_C2=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT2=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR2=0.0
export NORMALIZE_MODE=sigmoid
export NNODES=4
export DTYPE=float16
bash ./examples/CMDP/GRPO_qwen3_30ba3b_base_lag_mega.sh
```
```bash
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export GRPO_LAGRANGIAN_C=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR=0.0
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_ENABLE2=true
export GRPO_LAGRANGIAN_C2=0.0
export GRPO_LAGRANGIAN_LAMBDA_INIT2=0.0
export GRPO_LAGRANGIAN_LAMBDA_LR2=0.0
export NORMALIZE_MODE=sigmoid
export NNODES=4
export DTYPE=bfloat16
bash ./examples/CMDP/GRPO_qwen3_30ba3b_base_lag_mega.sh
```


双lambda
```bash
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export GRPO_LAGRANGIAN_C=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_ENABLE2=true
export GRPO_LAGRANGIAN_C2=0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT2=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR2=1e1
export NORMALIZE_MODE=sigmoid
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export GRPO_LAGRANGIAN_C=0.00000
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_ENABLE2=true
export GRPO_LAGRANGIAN_C2=0.00000
export GRPO_LAGRANGIAN_LAMBDA_INIT2=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR2=1e1
export NORMALIZE_MODE=sigmoid
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```

单lambda

```bash
export GRPO_LAGRANGIAN_C=-0.00000
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p80
export NORMALIZE_MODE=clip_negative_sigmoid
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_C=-0.00000
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=p85
export NORMALIZE_MODE=clip_negative_sigmoid
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```
```bash
export GRPO_LAGRANGIAN_C=-0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export DATA_TYPE=qwen3-8b
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```

## xxx

Qwen3-4B 效果比较好的实验：`grpo_qwen3_4b_lag_c-0.00050_z1_seq_geo_no_exp_dual_init0.4_lr1e1_min0.0_max1.0_bufmean_normsigmoid`

```bash
export GRPO_LAGRANGIAN_C=-0.00050
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.4
export GRPO_LAGRANGIAN_LAMBDA_LR=1e1
export GRPO_LAGRANGIAN_RATIO=seq_geo_no_exp
export GRPO_LAGRANGIAN_BUFFER_MODE=mean
export NORMALIZE_MODE=sigmoid
export NNODES=2
bash ./examples/CMDP/GRPO_qwen3_4b_lag.sh
```

## 3.27

- 1.5b调参

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-2
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-3
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-2
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.1
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-3
export GRPO_LAGRANGIAN_BUFFER_MODE=p60
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.5
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-2
export GRPO_LAGRANGIAN_LAMBDA_MIN=0.25
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

## 3.26


- 1.5b调参

```bash
export GRPO_LAGRANGIAN_C=1.001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-2
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-2
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

- qwen3反向penalty

```bash
export REWARD_RATIO_PENALTY_MAX_RATIO=-1
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_qwen3_8b.sh
```

```bash
export REWARD_RATIO_PENALTY_MAX_RATIO=-1
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_qwen3_4b.sh
```

## 3.25

constraint_budget_c=1.001，lambda_init=1.0， GRPO LAGRANGIAN LAMBDA MAX=10

目前的优先级是这样
（1）constraint_budget_c=1.001，lambda_init=1.0 
（2） constraint_budget_c=1.001，lambda_init=1.0， GRPO LAGRANGIAN LAMBDA MAX=10
（3）constraint_budget_c=1.001，lambda_init=0.8 
（4）constraint_budget_c=1.001，lambda_init=0.8， GRPO LAGRANGIAN LAMBDA MAX=10

```bash
export GRPO_LAGRANGIAN_C=1.001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.8
export GRPO_LAGRANGIAN_LAMBDA_MAX=10
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.001
export GRPO_LAGRANGIAN_LAMBDA_INIT=1.0
export GRPO_LAGRANGIAN_LAMBDA_MAX=10
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0005
export GRPO_LAGRANGIAN_LAMBDA_INIT=1.0
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

```bash
export GRPO_LAGRANGIAN_C=1.0001
export GRPO_LAGRANGIAN_LAMBDA_INIT=1.0
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

## 3.22

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.0002
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="cubed"
export REWARD_RATIO_PENALTY_PENALTY_TRUNCATED=0.109
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.0002
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="tanh"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.004
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="tanh"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="tanh"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


## 3.20

1. 这个实验wandb有记录到10步就可以停了，只需要看一下一个新增指标的早期表现

```bash
bash ./examples/CMDP/GRPO_r1_1p5b_step380.sh
```

2. 需要正常训练的实验（一共九个）


- rational mode

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.0002
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="rational"
export REWARD_RATIO_PENALTY_PENALTY_TRUNCATED=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.004
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="rational"
export REWARD_RATIO_PENALTY_PENALTY_TRUNCATED=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="rational"
export REWARD_RATIO_PENALTY_PENALTY_TRUNCATED=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- sigmoid mode

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.0002
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="sigmoid"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.004
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="sigmoid"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="sigmoid"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


- cubed mode

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.0002
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="cubed"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.004
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="cubed"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export REWARD_RATIO_PENALTY_NORMALIZE_MODE="cubed"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

## 3.19 v2

```bash
export TEMPERATURE=0.7
bash ./examples/CMDP/GRPO_r1_1p5b_temperature.sh
```
```bash
export TEMPERATURE=1.0
bash ./examples/CMDP/GRPO_r1_1p5b_temperature.sh
```
```bash
export TEMPERATURE=0.3
bash ./examples/CMDP/GRPO_r1_1p5b_temperature.sh
```

- 这个模型换成之前存的collapse的那个ckpt
```bash
bash ./examples/CMDP/GRPO_r1_1p5b_step380.sh
```

## 3.19

- 以90分位为clip

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.0002
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export CLIP_PENALTY_LOW=0.9998
export CLIP_PENALTY_HIGH=1.0002
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- 以中位数为clip

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export CLIP_PENALTY_LOW=0.996
export CLIP_PENALTY_HIGH=1.004
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=1.004
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- jeffery + weights (优先级最低)

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_jeffreys"
export REWARD_RATIO_PENALTY_MAX_RATIO=0.7
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```
- 其它

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo_inverse"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

## 3.17

1.1 clip实验，仅reward cost

- jeffery

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=135
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- jeffery + weights

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export REWARD_RATIO_PENALTY_MAX_RATIO=0.7
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- log-ppl-abs
```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export CLIP_PENALTY_LOW=0
export CLIP_PENALTY_HIGH=0.02323
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- kl

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_kl"
export CLIP_PENALTY_HIGH=130
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- kl-k3

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_k3_kl"
export CLIP_PENALTY_HIGH=120
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- forward kl

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_forward_kl"
export CLIP_PENALTY_HIGH=3.95
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- diff_seq_geo_abs

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="diff_seq_geo_abs"
export CLIP_PENALTY_HIGH=0.01395
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

1.2 无clip，仅reward cost

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_kl"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_forward_kl"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="diff_seq_geo_abs"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_k3_kl"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

1.3 normalized adv cost实验，

- with std
```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_jeffreys"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- wo std
```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export ADV_COST_NORM_ADV_BY_STD=false
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_jeffreys"
export ADV_COST_NORM_ADV_BY_STD=false
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


## 历史实验

## 3.13

1. 1.5b实验
1.1 需要隔10步存ckpt（跑完根据曲线留一个即可）

```bash
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

1.2 基于 Jeffreys divergence (seq) 的朴素 adv_cost 和 r_cost

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_jeffreys"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_jeffreys"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

1.3 基于 log_ppl_abs_diff 的朴素 adv_cost 和 r_cost

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="log_ppl_abs_diff"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="log_ppl_abs_diff"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

1.4 基于 diff_seq_sum_add（概率差之和）的朴素 adv_cost 和 r_cost

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="diff_seq_sum_add"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="diff_seq_sum_add"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

1.5 基于 seq_kl 的朴素 adv_cost 和 r_cost

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_kl"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_kl"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

2. qwen3 1.7b实验
补充1.7b-base 的bf16和fp16

```bash
bash ./examples/CMDP/GRPO_qwen3_1p7b_base.sh
```

```bash
export DTYPE="float16"
bash ./examples/CMDP/GRPO_qwen3_1p7b_base.sh
```

## FOR 1.5B

1. 分段 
https://wandb.ai/anitaleung/CMDP/runs/2yzkmdjt
baseline实验mismatch指标（k3、ratio_seq_geo_mean）出现异常在115-120步开始明显比先前的值要更大/更小，慢慢偏离越来越明显（训练集分数在360步左右到最高点然后开始掉）
因此这里分段以出现细微异常前的时间点作为分段点——101步作为节点

```bash
export 
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export PENALTY_COST_START_STEP=101
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export 
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq"
export PENALTY_COST_START_STEP=101
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export PENALTY_COST_START_STEP=101
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq"
export PENALTY_COST_START_STEP=101
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

2. 令adv=adv_reward-adv_cost*0.1

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export ADV_COST_COEF=0.1
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq"
export ADV_COST_NORM_ADV_BY_STD=true
export ADV_COST_COEF=0.1
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

3. reverse penalty

```bash
export REWARD_RATIO_PENALTY_MAX_RATIO=-1
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_MAX_RATIO=-1
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```
（有卡的时候可以补一个REWARD_RATIO_PENALTY_MAX_RATIO=-0.1的，但优先级没有其他的实验高）

```bash
export REWARD_RATIO_PENALTY_MAX_RATIO=-0.1
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_MAX_RATIO=-0.1
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

4. penalty 的clip (mismatch < C penalty为0)

https://wandb.ai/anitaleung/CMDP/runs/2yzkmdjt
参考训练critic/score/mean的最高点360步左右的ratio分布选取c
CLIP_PENALTY即为C
在360步前观察到rollout_corr/ratio_seq_p90或rollout_corr/ratio_seq_geo_p90呈现先上升后下降的趋势，在100附近有最高点。(p90即90百分位)
和1.中提到的seq_geo_mean趋势类似，正常波动后在100步出现异常，300-500步在明显严重过小的数值，最后又回升到正常数值（崩了之后收敛）
这里CLIP_PENALTY_HIGH先采取360步平滑后的p90，0.654 for ratio_seq，0.99983 for ratio_seq_geo
但seq_geo_mean的值长期在1以下，collapse的过程中也是一直下降的，所以我觉得双向c的实验是可以考虑的


添加了新指标，需要重新跑一遍baseline所以

```bash
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

双向clip
c < mismatch < 1/c

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export ADV_COST_NORM_ADV_BY_STD=true
export CLIP_PENALTY_LOW=0.9999
export CLIP_PENALTY_HIGH=1.0001
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
export CLIP_PENALTY_LOW=0.9999
export CLIP_PENALTY_HIGH=1.0001
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq"
export ADV_COST_NORM_ADV_BY_STD=true
export CLIP_PENALTY_LOW=0.654
export CLIP_PENALTY_HIGH=1.53
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq"
export CLIP_PENALTY_LOW=0.654
export CLIP_PENALTY_HIGH=1.53
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```


---
## 历史实验
```bash
export
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export DTYPE="float16"
bash ./examples/CMDP/GRPO_qwen3_4b_base.sh
```

```bash
bash ./examples/CMDP/GRPO_qwen3_4b_base.sh
```

```bash
export DTYPE="float16"
bash ./examples/CMDP/GRPO_qwen3_8b_base.sh
```

```bash
bash ./examples/CMDP/GRPO_qwen3_8b_base.sh
```

```bash
export DTYPE="float16"
bash ./examples/CMDP/GRPO_qwen3_1p7b.sh
```

```bash
bash ./examples/CMDP/GRPO_qwen3_1p7b.sh
```



```bash
export DTYPE="float16"
bash ./examples/CMDP/GRPO_qwen3_4b.sh
```

```bash
bash ./examples/CMDP/GRPO_qwen3_4b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_qwen3_4b.sh
```

```bash
export DTYPE="float16"
bash ./examples/CMDP/GRPO_qwen3_8b.sh
```

```bash
bash ./examples/CMDP/GRPO_qwen3_8b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_qwen3_8b.sh
```


- 实验1:bf16 baseline
```bash
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- 实验2:bf16 baseline + adv_cost seq-geo-level
```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

- 实验3:bf16 baseline + adv_cost seq-level
```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq"
export ADV_COST_NORM_ADV_BY_STD=true
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

```bash
export REWARD_RATIO_PENALTY_ENABLE=true
export REWARD_RATIO_PENALTY_RATIO_TYPE="seq_geo"
bash ./examples/CMDP/GRPO_r1_1p5b.sh
```

# drgrpo

- 实验1:bf16 baseline + adv_cost seq-geo-level
```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq_geo"
export ADV_COST_NORM_ADV_BY_STD=false
bash ./examples/CMDP/drgrpo_r1_1p5b.sh
```

- 实验2:bf16 baseline + adv_cost seq-level
```bash
export ADV_COST_ENABLE=true
export ADV_COST_TYPE="seq"
export ADV_COST_NORM_ADV_BY_STD=false
bash ./examples/CMDP/drgrpo_r1_1p5b.sh
```


