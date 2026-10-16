# arXiv:2606.08779 reproduction review

## Review conclusion

The repository implements the paper's DC-GRPO path: rollout probabilities from vLLM are compared with training-backend probabilities on the same generated tokens, `token_prob_diff` is normalized by `absneg_0.0_none`, and the absolute mismatch is applied to the GRPO advantage and projected Lagrange update.

The bounded Claude Scholar review found two blocking reproducibility issues in the upstream fork:

1. The README launcher had machine-specific `/mnt/ws-jfs/...` paths and embedded a W&B credential.
2. The launcher assumed eight GPUs and could not be submitted unchanged to a one-GH200 Roihu job.

The reproduction launcher now takes paths, Python, offload, batch, and secret settings from the environment. It never prints or stores `HF_TOKEN` or `WANDB_API_KEY`.

## Fixed provenance

| Component | Version or source |
|---|---|
| Repository | `AnitaLeungxx/verl`, commit `3f6ba816bedd9e03129d1e5c07234fe8041a5338` |
| verl | `0.7.0.dev` |
| PyTorch | `2.8.0+cu129` |
| vLLM | `0.11.0+cu129` |
| Transformers | `4.57.3` |
| Ray | `2.53.0` |
| sglang | `0.5.2` |
| math-verify | `0.8.0` |
| Dataset | repository `data/train/dapo_train.parquet`, 17,398 rows |

## Validation

- Local shell syntax, Python compilation, dataset schema/row count, DC-GRPO formula, and `git diff --check`: passed.
- Remote container import/version check: passed.
- Roihu GPU smoke job `2344496`: `COMPLETED`, exit code `0:0`; one GH200 visible and CUDA available.
- Roihu GPU smoke job `2345363`: `COMPLETED`, exit code `0:0`; one GH200 visible, version/data/import/config checks passed after the complete source sync.
- Full reproduction job `2345365`: `FAILED`, exit code `1:0`, after 2:06. The full 17,398-row dataset loaded and actor initialization began, but Qwen3's default `flash_attention_2` path raised an ImportError because `flash_attn` is unavailable in the ARM64 container; no training step ran. Logs are preserved locally under `logs/csc/roihu/2345365/`.
- Retry smoke `2345512` is running on one GH200 in `gputest`; full-data retry `2345513` is queued in `gpumedium` with `afterok:2345512`.
- Final observed retry state: smoke `2345512` completed in 1:03 with exit `0:0`, including the SDPA Hydra config check; full-data job `2345513` entered `RUNNING` on `rg1124`, started Ray/TaskRunner, and passed the former FlashAttention actor-initialization failure point. Local smoke logs and a full-job snapshot are under `logs/csc/roihu/2345512/` and `logs/csc/roihu/2345513/`.
- Full reproduction job `2344498`: failed after 15 seconds at import time because the first sync filter accidentally excluded the source directory `verl/verl/models`; no training step ran. Its logs are preserved locally.
- After syncing that missing source directory, retry job `2344533` was submitted to `gpumedium` and was pending on scheduler priority at the last check.
- Full job `2346379` then failed before training because the Ray Unix socket path exceeded the 107-byte limit. The launcher was repaired to use the short per-job path `/workspace/r/<job-id>` for Ray/TMPDIR while keeping model caches and outputs isolated by job under scratch.
- Latest smoke `2346845`: `COMPLETED`, exit code `0:0`; compiler, local reproduction, import, GPU/config, SDPA, and `use_remove_padding=false` checks passed. Full job `2346846` is `RUNNING` on `rg2118`; it started Ray, processed all 17,398 rows, loaded Qwen3-8B, completed both vLLM CUDA-graph capture phases, and entered the 301-step training loop without the earlier `flash_attn` import failure. Snapshot logs are archived under `logs/csc/roihu/2346845/` and `logs/csc/roihu/2346846/`.
- Full job `2346846` subsequently failed after 6:02 with CUDA OOM during the first actor `compute_log_prob`; the failing allocation was the dense BF16 `log_softmax` temporary. The complete traceback is preserved in the local Slurm stderr log.
- Smoke `2347302` after the OOM adaptation completed with exit code `0:0`; all version, import, data, GPU, SDPA, and launcher checks passed. Full retry `2347383` was submitted to `gpumedium` with `afterok:2347302` and is currently pending due to scheduler priority.
- The current comparison target is a matched AIME24 pilot: DAPO-Math-17K remains the training data, `data/eval/aime24_32.parquet` (960 examples) is the only validation set, and Qwen3-8B plus all pinned framework versions are shared. DC-GRPO uses `GRPO_LAGRANGIAN_ENABLE=true`; vanilla GRPO uses `false`. Jobs are serialized to avoid using two GH200s simultaneously: DC-GRPO `2347511`, then GRPO `2347512`.
- To verify the OOM repair without waiting for the long `gpumedium` queue, one-step 15-minute runtime gates were submitted on `gputest`: DC-GRPO `2347545`, then GRPO `2347546`. They use the same AIME24 config and are only a runtime gate, not the effect-comparison result.
- Gate `2347545` failed before the first training step because `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with vLLM 0.11.0's memory pool on Roihu. The allocator override was removed from all reproduction scripts; the official framework and pinned versions were unchanged.
- Gate `2347681` passed model load and CUDA-graph setup, entered `Training Progress`, and then failed in the first dense-logit entropy calculation. Gate `2347938` passed the rollout/log-prob phase but failed while FSDP initialized the actor optimizer. Gate `2348041` used shorter response and batch limits but still failed at optimizer initialization with only a few MiB free. These failures identify the single-GH200 memory boundary rather than a DC-GRPO code-path error.
- The current repair uses the same Qwen3-8B and DAPO/AIME24 data but sets the Roihu pilot to `GPU_MEMORY_UTILIZATION=0.20`, `ENFORCE_EAGER=true`, `MAX_RESPONSE_LENGTH=2048`, rollout `n=2`, prompt batch `16`, mini-batch `4`, and 4096-token vLLM/actor scheduling caps. Gate `2348129` was submitted with three real training steps and a 15-minute `gputest` limit to verify that this configuration can remain in training for at least ten minutes. The fixed AIME24 comparison is submitted only after this gate passes.
- Gate `2348129` reached the first real training step but failed at 2:57 while Adam initialized `exp_avg_sq`; only 19.75 MiB of GPU memory remained. The container's PyTorch documents `CPUOffloadPolicy` as offloading parameters, gradients, and optimizer states, so the next repair enables the existing FSDP2 `actor.fsdp_config.offload_policy=true` path and disables the legacy actor offload flags. No core `verl/` source, framework version, or algorithm was changed.

## Submitted resources

The full job requests account `project_2008161`, partition `gpumedium`, one `gh200`, 72 CPUs, 217086 MB Slurm memory, and the 36-hour partition maximum. It uses full DAPO-Math-17K data, 301 training steps, the paper's `c=0.0037`, `lambda_init=0.1`, `lambda_lr=1.0`, and a unique run directory under the project scratch path. Actor offload is enabled as the necessary one-GH200 hardware adaptation; this is not an exact eight-GPU numerical rerun.

## Files

- `verl/scripts/repro/roihu_gpu_smoke.sbatch`
- `verl/scripts/repro/roihu_dcmdp_dapo_qwen3_8b.sbatch`
- `verl/scripts/repro/validate_dcmdp_setup.py`
- `verl/scripts/repro/check_versions.py`
- `logs/csc/roihu/2344496/`
- `logs/csc/roihu/2344498/`
- `logs/csc/roihu/2346469/`
- `logs/csc/roihu/2346470/`
- `logs/csc/roihu/2346845/`
- `logs/csc/roihu/2346846/`
- `logs/csc/roihu/2347302/`
- `logs/csc/roihu/2347511/`
- `logs/csc/roihu/2347512/`
- `logs/csc/roihu/2347545/`
- `logs/csc/roihu/2347546/`

The requested `lumi-minimize-gpu-hours` skill path was not present on Roihu. Resource selection therefore follows the current CSC Roihu partition documentation and the GPU-training skill's minimum-resource guidance.

## Roihu ARM64 attention adaptation

The fork's pinned install helper targets `flash-attn==2.8.1`, but Roihu's ARM64 container does not ship that module and PyPI does not provide a matching ARM64 wheel. The launcher now accepts `ATTN_IMPLEMENTATION`; supported hosts keep the original `flash_attention_2` default, while the Roihu sbatch script explicitly sets `sdpa`. This changes only the Hugging Face attention backend required by the hardware/container and leaves the DC-GRPO algorithm and pinned repository/dependency versions unchanged.

The same Roihu container also lacks the `flash_attn` module needed by the fork's remove-padding helper. Therefore the Roihu sbatch script sets `actor_rollout_ref.model.use_remove_padding=false`; the official launcher default and package source remain unchanged. The Hydra configuration is validated in the GPU smoke job before the full run.

The one-GH200 adaptation also has a dense-logits memory limit during `compute_log_prob` and an FSDP optimizer-initialization peak. The official token-budget defaults remain intact for supported hosts; the Roihu pilot overrides response length, rollout count, batch sizes, eager execution, vLLM/actor scheduling caps, and FSDP2 CPU offload to leave memory headroom. This is a resource-scheduling adaptation only; it does not change the model, reward, DC-GRPO equations, optimizer, dataset, or dependency versions. It is therefore a hardware-adapted pilot, not an exact numerical reproduction of the paper's eight-GPU run.
