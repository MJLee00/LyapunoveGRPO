# GRPO + Lagrangian (constrained penalty)

## Objective

Per sampled sequence (GRPO groups by prompt):

- Task reward \(R\) from the reward model (on the last token for outcome RL).
- Normalized penalty \(p \in [0,1]\) from the same **local** ratio modes as `reward_ratio_penalty` (e.g. `seq_forward_kl`).

**Shaped reward:** \(\tilde{R} = R - \lambda \, p\).

**Constraint:** keep the running average of batch-mean \(p\) near budget \(c\):

- **Dual operator (default):** every \(z\) steps,
  \(\lambda \leftarrow \mathrm{clip}(\lambda + \eta \, (\bar{g} - c),\, 0,\, \lambda_{\max})\),
  where \(\bar{g}\) is the mean of all values in the FIFO buffer (each step appends the current batch’s \(\mathrm{mean}(p)\)).

- **MLP operator:** a small MLP maps a fixed window of recent batch means to \(\lambda \ge 0\), updated every \(z\) steps with a subgradient-style step on \(-\lambda(\bar{g}-c)\) (see `grpo_lagrangian.py`).

## Config (`algorithm.grpo_lagrangian`)

| Key | Meaning |
|-----|---------|
| `enable` | Turn on Lagrangian shaping (disables fixed `reward_ratio_penalty` path when both would apply). |
| `constraint_budget_c` | Target mean normalized penalty \(c\). |
| `lambda_init`, `lambda_lr`, `lambda_max` | Dual multiplier init, step size, cap. |
| `operator_update_every_z` | Update \(\lambda\) every \(z\) training steps. |
| `buffer_size` | Max history length for batch-mean \(p\). |
| `operator_type` | `dual` or `mlp`. |
| `ratio_type`, `max_ratio` | Same as ratio penalty (local modes only; not `seq_global`). |
| `mlp_*` | MLP width, window, LR, L2 on \(\lambda\) when `operator_type=mlp`. |

Checkpoint: `global_step_*/grpo_lagrangian.pt` stores \(\lambda\), buffer, and optional MLP weights.

## Script

See `GRPO_r1_1p5b_lagrangian.sh` (minimal; copy rollout/model paths from `GRPO_r1_1p5b.sh` for your cluster).
