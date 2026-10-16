# GRPO + CMDP 拉格朗日乘子法实现说明

本文解释这个仓库里 `GRPO_r1_1p5b_lag.sh` 对应的算法实现。目标读者是强化学习初学者，所以会先讲“它想解决什么问题”，再讲“代码里到底改了哪里”。

如果只看一句话版：

- 普通 GRPO 只想让模型拿更高的 reward。
- 这个版本在普通 GRPO 的 reward 外，又加了一个“别偏离约束太多”的 penalty。
- penalty 前面的系数不是写死的，而是一个会自动调节的拉格朗日乘子 `lambda`。
- 代码上，GRPO 本身的 advantage 公式没有改；改的是它看到的 reward，从 `R` 变成了 `R - lambda * p`。

## 1. 先看普通 GRPO 在做什么

这里先不讲 PPO 的全部细节，只讲这份实现真正用到的部分。

在这个仓库里，GRPO 的核心思路是：

1. 对同一个 prompt 采样出多条回答。
2. 用 reward model 给每条回答打一个分数。
3. 在同一个 prompt 的回答组内，比较谁比组平均更好、谁更差。
4. 提高高分回答的概率，降低低分回答的概率。

这一步在代码里的核心函数是：

- `verl/trainer/ppo/core_algos.py` 里的 `compute_grpo_outcome_advantage`

它做的事情很直接：

- 先把一条回答所有 token 的 reward 求和，得到这条回答的总分 `score`
- 再按同一个 prompt 的多条回答分组
- 组内做减均值，默认还会再除以标准差
- 得到 advantage 之后，把这个 advantage 广播回该回答的每个有效 token

可以把它写成：

```text
score_i = sum(token_level_rewards_i)
adv_i = (score_i - group_mean) / group_std
```

这就是普通 GRPO 的基本训练信号。

## 2. 为什么这里要加 CMDP

普通强化学习只有一个目标：

```text
尽量把 reward 做大
```

但很多时候我们不只想“高分”，还想满足一个额外约束，比如：

- 不要让训练策略和 rollout 策略偏差太大
- 不要让某种代价超过预算
- 不要让某种风险长期失控

这类问题通常写成 CMDP（Constrained Markov Decision Process，带约束的马尔可夫决策过程）：

```text
max   E[R]
s.t.  E[p] <= c
```

其中：

- `R` 是任务 reward
- `p` 是 penalty 或 cost
- `c` 是你允许的预算上限

这份实现里最重要的一点是：

> 这里的 `p` 不是“答案是否违规”这种语义层面的 cost，而是根据 `old_log_probs` 和 `rollout_log_probs` 算出来的策略偏移惩罚。

也就是说，它约束的是：

```text
训练时使用的策略分布，不要和 rollout 时的策略分布偏得太离谱
```

## 3. 拉格朗日乘子法的直觉

CMDP 难点在于：你既想把 reward 做高，又想满足约束。

一个经典做法是把约束“揉进”目标函数里：

```text
L(theta, lambda) = E[R - lambda * (p - c)]
```

展开后是：

```text
L(theta, lambda) = E[R - lambda * p] + lambda * c
```

对策略参数 `theta` 来说，`lambda * c` 是常数，所以真正影响策略更新的是：

```text
R_tilde = R - lambda * p
```

这就是为什么代码里最终做的是：

```text
把原始 reward 改成 shaped reward = reward - lambda * penalty
```

而 `lambda` 自己怎么更新？

直觉上：

- 如果最近 penalty 太大，说明违反约束了，就把 `lambda` 调大
- 如果最近 penalty 不大，说明约束很松，就把 `lambda` 调小

最常见的 dual ascent 写法是：

```text
lambda <- clip(lambda + lr * (g_bar - c), lambda_min, lambda_max)
```

其中：

- `g_bar` 是最近一段时间 penalty 的平均值
- `c` 是预算
- `lr` 是更新 `lambda` 的学习率

这份仓库默认就用这个版本，`operator_type=dual`。

## 4. 这份仓库里的 penalty `p` 是怎么定义的

### 4.1 它来自哪里

训练循环里会同时拿到两套 log-prob：

- `rollout_log_probs`：生成回答时，rollout 引擎给出的 log-prob
- `old_log_probs`：训练端重新计算出来的旧策略 log-prob

对应代码：

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` 会把 `rollout_log_probs` 放进 batch
- `verl/trainer/ppo/ray_trainer.py` 会重新计算 `old_log_probs`

所以这里比较的是“训练侧旧策略”和“rollout 侧策略”之间的差异。

### 4.2 默认脚本使用的 penalty 形式

脚本 `examples/CMDP/GRPO_r1_1p5b_lag.sh` 默认设置：

```bash
GRPO_LAGRANGIAN_RATIO=${GRPO_LAGRANGIAN_RATIO:-seq_geo}
```

对应代码在 `verl/trainer/ppo/rollout_corr_helper.py` 的 `compute_normalized_penalty_per_sequence` 里。

当 `ratio_type=seq_geo` 时：

```text
log_ratio = old_log_probs - rollout_log_probs
raw_per_seq = exp(mean(log_ratio))
```

然后默认还会做一次归一化。脚本没有显式传 `normalize_mode`，所以会走配置默认值 `rational`：

```text
p = raw_per_seq / (1 + raw_per_seq)
```

因此在默认配置下：

- `p` 的范围在 `(0, 1)` 之间
- `p` 越大，说明当前定义下的“偏移惩罚”越大

### 4.3 一个非常容易误解的点

当 `ratio_type=seq_geo` 且 `normalize_mode=rational` 时：

```text
如果 old_log_probs 和 rollout_log_probs 完全一样
=> log_ratio = 0
=> raw_per_seq = exp(0) = 1
=> p = 1 / (1 + 1) = 0.5
```

也就是说：

- 在这个默认 penalty 定义下，“完全一致”并不会得到 `p = 0`
- 它更像是一个压缩后的 ratio 指标，不是“以 0 为理想点”的标准距离函数

这点对理解 `constraint_budget_c` 非常重要。

## 5. 代码到底改了普通 GRPO 的哪里

这一节最关键。

### 5.1 脚本层：把拉格朗日参数传进去

脚本是：

- `examples/CMDP/GRPO_r1_1p5b_lag.sh`

它把这些环境变量传给训练配置：

- `algorithm.grpo_lagrangian.enable`
- `algorithm.grpo_lagrangian.constraint_budget_c`
- `algorithm.grpo_lagrangian.operator_update_every_z`
- `algorithm.grpo_lagrangian.buffer_size`
- `algorithm.grpo_lagrangian.lambda_lr`
- `algorithm.grpo_lagrangian.lambda_max`
- `algorithm.grpo_lagrangian.lambda_init`
- `algorithm.grpo_lagrangian.lambda_min`
- `algorithm.grpo_lagrangian.ratio_type`
- `algorithm.grpo_lagrangian.operator_type`

同时它显式关掉了另外两条旧路径：

```text
algorithm.reward_ratio_penalty.enable=false
algorithm.adv_cost.enable=false
```

这说明当前脚本要走的是“动态 lambda 的拉格朗日路径”，而不是“固定 penalty 系数”或“直接从 advantage 扣 cost”那两种实现。

### 5.2 Trainer 初始化：创建拉格朗日操作器

在 `verl/trainer/ppo/ray_trainer.py` 里，trainer 初始化时会读取配置：

```text
if grpo_lagrangian.enable:
    self._grpo_lagrangian = GRPOLagrangianOperator(cfg)
```

也就是说，只要脚本把 `algorithm.grpo_lagrangian.enable=true` 打开，trainer 内部就会多维护一个对象，专门负责：

- 当前 `lambda` 的值
- 最近若干个 batch 的 penalty 均值缓存
- 定期更新 `lambda`

### 5.3 训练主循环：先算普通 reward，再减去 `lambda * p`

真正的关键逻辑在 `verl/trainer/ppo/ray_trainer.py`。

训练时顺序大致是：

1. reward model 先产生原始 reward
2. 如果没开 KL in reward，就直接：

```text
token_level_rewards = token_level_scores
```

3. 如果开启了 `grpo_lagrangian`，就额外计算每条回答的 penalty `p_seq`
4. 把这个 batch 的 `mean(p_seq)` 记进 buffer
5. 用当前 `lambda` 执行：

```text
token_level_rewards = token_level_rewards - lambda * p_seq
```

不过这里不是把 penalty 平铺到每个 token 上，而是只扣在“最后一个有效 token”上。这样做的好处是：

- 整条回答仍然只对应一个 outcome-level reward
- 不会把一个序列长度为 `T` 的 penalty 重复扣 `T` 次

这正好和 GRPO 的 outcome-style reward 形式一致。

### 5.4 GRPO advantage 本身没有改

这一点必须单独强调：

> 这份实现没有改 `compute_grpo_outcome_advantage` 的公式。

GRPO 还是照常做：

```text
score_i = sum(token_level_rewards_i)
adv_i = group_relative(score_i)
```

只是这里的 `token_level_rewards` 已经不是原始 reward，而是被改写后的：

```text
R_tilde = R - lambda * p
```

所以从算法视角看：

- 普通 GRPO：用 `R` 算 advantage
- 这里的 GRPO-Lagrangian：用 `R - lambda * p` 算 advantage

### 5.5 每隔 `z` 步更新一次 `lambda`

`lambda` 的更新逻辑在：

- `verl/trainer/ppo/grpo_lagrangian.py`

默认 `operator_type=dual` 时，逻辑就是：

1. 每个 batch 记下一个 `g_mean = mean(p_seq)`
2. 放入一个 FIFO buffer
3. 每隔 `z` 步，计算：

```text
g_bar = mean(buffer)
violation = g_bar - c
lambda = clip(lambda + lambda_lr * violation, lambda_min, lambda_max)
```

含义很直观：

- `g_bar > c`：约束超了，`lambda` 增大
- `g_bar < c`：约束没超，`lambda` 减小

### 5.6 Checkpoint 也会保存 `lambda`

这份实现还把拉格朗日状态保存进 checkpoint：

- 保存文件：`global_step_*/grpo_lagrangian.pt`

里面包括：

- 当前 `lambda`
- buffer 历史
- 如果是 `mlp` operator，还会保存 MLP 参数和优化器状态

所以断点续训不会把 `lambda` 重新清零。

## 6. 和普通强化学习 / 普通 GRPO 相比，改动点在哪里

可以把改动总结成四句话。

### 6.1 目标函数变了

普通 GRPO 优化的是：

```text
max E[R]
```

这里优化的是近似的拉格朗日目标：

```text
max E[R - lambda * p]
```

而 `lambda` 再根据是否超预算自动调整。

### 6.2 reward 被改写了，但 actor loss 没改

普通 GRPO 和这里的主要差别不在 PPO loss 公式本身，而在 reward 预处理：

- 普通 GRPO：直接拿 reward model 的分数
- 这里：先减掉 `lambda * p`

后面的 GRPO advantage 计算和 PPO 更新流程基本照旧。

### 6.3 多了一套“约束状态”

普通 GRPO 不需要维护这些东西：

- `lambda`
- buffer
- `constraint_budget_c`
- `operator_update_every_z`

这里都需要。

### 6.4 多了一项 penalty 计算依赖

普通 GRPO 只要 reward 就能训。

这里还必须能计算：

- `old_log_probs`
- `rollout_log_probs`
- `response_mask`

因为 penalty `p` 是从这些量算出来的。

## 7. 和仓库里另外两种相关做法的区别

这个仓库里和它容易混淆的还有两条路径。

### 7.1 和 `reward_ratio_penalty` 的区别

`reward_ratio_penalty` 也是直接从 reward 里扣 penalty，但它的系数是固定的，不会自动调。

可以粗略理解成：

- `reward_ratio_penalty`：固定税率
- `grpo_lagrangian`：税率 `lambda` 会根据最近是否超预算自动涨跌

### 7.2 和 `adv_cost` 的区别

`adv_cost` 不是在 reward 阶段扣，而是额外算一个 cost advantage，然后从 reward advantage 里减掉。

也就是说：

- `grpo_lagrangian`：改的是 reward
- `adv_cost`：改的是 advantage

当前脚本里这两条都关掉了，只走拉格朗日路径。

## 8. 用伪代码看整个训练流程

下面这段伪代码基本对应当前实现：

```python
for each training step:
    batch = rollout()
    reward = reward_model(batch)
    token_level_rewards = reward

    p_seq = penalty_from(old_log_probs, rollout_log_probs, response_mask)
    buffer.append(mean(p_seq))

    lam = current_lambda()
    token_level_rewards[last_token] -= lam * p_seq

    advantages = GRPO_advantage(token_level_rewards)
    update_policy_with_GRPO(advantages)

    if step % z == 0:
        g_bar = mean(buffer)
        lam = clip(lam + lambda_lr * (g_bar - c), lambda_min, lambda_max)
```

所以这份实现的核心思想其实非常简单：

```text
普通 GRPO + 一个会自动调节强度的 penalty
```

## 9. 你给的默认运行参数，在当前代码里意味着什么

你的运行方式是：

```bash
export GRPO_LAGRANGIAN_C=1.001
export GRPO_LAGRANGIAN_LAMBDA_INIT=0.2
export GRPO_LAGRANGIAN_LAMBDA_LR=1e-3
bash ./examples/CMDP/GRPO_r1_1p5b_lag.sh
```

结合脚本默认值，实际关键参数大致是：

```text
constraint_budget_c = 1.001
lambda_init = 0.2
lambda_lr = 1e-3
lambda_min = 0.0
lambda_max = 1.0
operator_update_every_z = 1
buffer_size = 512
ratio_type = seq_geo
operator_type = dual
```

它们的含义分别是：

- `constraint_budget_c=1.001`：希望最近一段时间的平均 penalty 不超过 `1.001`
- `lambda_init=0.2`：训练一开始先用 `0.2` 的强度扣 penalty
- `lambda_lr=1e-3`：每次更新 `lambda` 时只做很小一步
- `operator_update_every_z=1`：每一步训练后都尝试更新一次 `lambda`
- `buffer_size=512`：`g_bar` 最多看最近 512 个 batch 的 penalty 均值

### 9.1 这个默认命令的一个重要结论

在当前代码默认归一化下：

- `ratio_type=seq_geo`
- `normalize_mode=rational`
- `max_ratio` 没有设置

所以 `p` 的范围是 `(0, 1)`。

而你把预算设成了：

```text
c = 1.001
```

这意味着：

```text
g_bar - c 基本总是 <= 0
```

于是默认 dual 更新会倾向于：

```text
lambda 逐渐减小，直到靠近 lambda_min
```

举个非常直接的数值例子：

```text
假设当前 buffer 平均 penalty g_bar = 0.62
lambda = 0.2
lambda_lr = 1e-3
c = 1.001
```

那么更新后：

```text
lambda_new = 0.2 + 1e-3 * (0.62 - 1.001)
           = 0.199619
```

也就是 `lambda` 会往下走，而不是往上走。

### 9.2 这代表什么

这说明在当前实现和当前默认 penalty 定义下，你这组参数更像是：

- 训练初期先施加一个初始 penalty 强度 `0.2`
- 后面如果没有别的配置改动，`lambda` 大概率会慢慢衰减

换句话说，这组参数并不是一个“强约束”的设置，更像是一个“先轻微约束，随后放松”的设置。

如果你想让拉格朗日机制真正积极工作，一般需要：

- `c` 处在实际观测到的 `p` 范围里
- 而不是明显高于 `p` 的上界

但这里要注意，因为默认 `seq_geo + rational` 在“完全匹配”时就会给出 `p=0.5`，所以 `c` 的解释不能按“越接近 0 越好”去理解，必须结合当前 penalty 定义一起看。

## 10. 这个实现里最值得记住的三件事

### 10.1 它约束的是策略偏移，不是语义安全 cost

默认这条实现不是“防有害回答”的那种 cost，而是用 log-prob ratio 构造出来的分布偏移惩罚。

### 10.2 它没有改 GRPO 的 advantage 公式

真正改的是 reward：

```text
R -> R - lambda * p
```

GRPO 只是拿这个新 reward 去算 advantage。

### 10.3 `lambda` 是自动调节的

这就是它和固定 penalty 最大的区别：

- 超预算就加大惩罚
- 低于预算就减小惩罚

这也是“拉格朗日乘子法”在代码里最本质的体现。

## 11. 关键代码位置

如果你想直接顺着源码读，建议按这个顺序：

1. `examples/CMDP/GRPO_r1_1p5b_lag.sh`
   - 看脚本把哪些参数传给 `algorithm.grpo_lagrangian.*`
2. `verl/trainer/ppo/ray_trainer.py`
   - 看训练循环里什么时候算 `p_seq`
   - 看什么时候把 reward 改成 `R - lambda * p`
   - 看什么时候更新 `lambda`
3. `verl/trainer/ppo/grpo_lagrangian.py`
   - 看 `GRPOLagrangianOperator`
   - 看 `apply_lagrangian_penalty_to_rewards`
4. `verl/trainer/ppo/rollout_corr_helper.py`
   - 看 `compute_normalized_penalty_per_sequence`
   - 这里定义了 `p` 到底怎么算
5. `verl/trainer/ppo/core_algos.py`
   - 看 `compute_grpo_outcome_advantage`
   - 这里能确认 GRPO 本身没有被改写

---

如果把整件事压缩成一句最容易记的话：

```text
这份实现 = 普通 GRPO 先照常算 reward，再额外扣一个动态权重 lambda 控制的 penalty，
然后用这个改写后的 reward 继续走原本的 GRPO 流程。
```
