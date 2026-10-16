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
- Gate `2348202` validated that repair: it completed two real actor-update steps, ran for 15:21, and kept GPU usage around 23 GiB without an OOM. Slurm then stopped it at the `gputest` wall-time limit before step 3, so it is recorded as `TIMEOUT` rather than a false success. The ten-minute runtime criterion passed, and its logs are preserved under `logs/csc/roihu/2348202/`.
- With the repaired configuration, matched AIME24 pilot jobs were submitted: DC-GRPO `2348506` and vanilla GRPO `2348507` with `afterok:2348506`. Both use the full DAPO-Math-17K train parquet, only `aime24_32.parquet` for validation, Qwen3-8B, the same pinned framework versions, and 20 training steps; only `GRPO_LAGRANGIAN_ENABLE` differs. At the last status check, both were pending in `gpumedium` by scheduler priority/dependency.
- A later audit of the pre-existing full jobs found a separate storage failure: `2347383` was still marked running but Ray repeatedly aborted with `Disk quota exceeded` while opening worker logs under `/workspace/r`. The scratch filesystem is globally 96% full even though this project uses only about 156 MB. The active reproduction scripts now put only Ray/TMP session files under the already-bound `/envroot/ray/<job-id>` on `/projappl`; source code, datasets, checkpoints, and result directories remain under the required scratch project path. The pending comparison jobs were updated before execution.
- Because Slurm stores the batch script at submission time, the earlier `2348506/2348507` pair cannot inherit that storage fix. They were left untouched as required. The authoritative resubmission is DC-GRPO `2348712`, followed by GRPO `2348713` with `afterok:2348712`; both were submitted after the `/envroot/ray/<job-id>` fix.
- The first `/envroot/ray` resubmission still failed before the script opened its own log (`2348712`, `RaisedSignal:53`, no stdout/stderr). A direct inode check showed the scratch filesystem had `IUse=100%` and `IFree=0`; writing even a test file returned `Disk quota exceeded`. The final storage repair moves the Slurm `--output/--error` paths and all runtime-created files to `/projappl/project_2015042/mingyl/LyapunoveGRPO/runtime` (12 GB free), while keeping scratch as the read-only code/data source. The authoritative pair is now DC-GRPO `2348852`, followed by GRPO `2348853`.
- Fixed4 DC-GRPO `2348852` reached the container and passed the GPU/import checks, but Hydra still used its default relative `outputs/<date>/<time>` directory under `/workspace/verl`. Scratch inode exhaustion caused `OSError: [Errno 122] Disk quota exceeded` before training. The launcher now accepts `HYDRA_RUN_DIR` and passes `hydra.run.dir=...`; both the generic DC script and matched AIME24 comparison script place Hydra output under the per-job `/projappl` runtime directory. The next authoritative pair must be submitted from this corrected snapshot.
- Corrected pair submitted after local validation and remote launcher hash verification: DC-GRPO `2348912`, followed by vanilla GRPO `2348913` with `afterok:2348912`. Both use the full DAPO-Math-17K training parquet, the repository AIME24 validation parquet, Qwen3-8B, and one GH200 on `gpumedium`; the only method difference is the Lagrangian switch. At submission they were pending without a failure state.
- DC-GRPO `2348912` subsequently completed with `ExitCode=0:0`, `AIME24_COMPARE_COMPLETED=1`, visible GH200, and `global_step_20`. Its final AIME24 metrics were `avg@32=0.2072916667`, `pass@32=0.5`, `std@32=0.0422742164`, `best@32=0.4812333333`, and `maj@32=0.1844`. Vanilla GRPO `2348913` is now running after the dependency; no method comparison is declared until it also completes successfully.

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
- `logs/csc/roihu/2348202/`

The requested `lumi-minimize-gpu-hours` skill path was not present on Roihu. Resource selection therefore follows the current CSC Roihu partition documentation and the GPU-training skill's minimum-resource guidance.

## Roihu ARM64 attention adaptation

The fork's pinned install helper targets `flash-attn==2.8.1`, but Roihu's ARM64 container does not ship that module and PyPI does not provide a matching ARM64 wheel. The launcher now accepts `ATTN_IMPLEMENTATION`; supported hosts keep the original `flash_attention_2` default, while the Roihu sbatch script explicitly sets `sdpa`. This changes only the Hugging Face attention backend required by the hardware/container and leaves the DC-GRPO algorithm and pinned repository/dependency versions unchanged.

The same Roihu container also lacks the `flash_attn` module needed by the fork's remove-padding helper. Therefore the Roihu sbatch script sets `actor_rollout_ref.model.use_remove_padding=false`; the official launcher default and package source remain unchanged. The Hydra configuration is validated in the GPU smoke job before the full run.

The one-GH200 adaptation also has a dense-logits memory limit during `compute_log_prob` and an FSDP optimizer-initialization peak. The official token-budget defaults remain intact for supported hosts; the Roihu pilot overrides response length, rollout count, batch sizes, eager execution, vLLM/actor scheduling caps, and FSDP2 CPU offload to leave memory headroom. This is a resource-scheduling adaptation only; it does not change the model, reward, DC-GRPO equations, optimizer, dataset, or dependency versions. It is therefore a hardware-adapted pilot, not an exact numerical reproduction of the paper's eight-GPU run.

The Roihu pilot additionally disables the launcher's optional entropy `torch.compile` path via `USE_TORCH_COMPILE=false`. This was prompted by an observed ARM64 first-step bottleneck: TorchInductor workers consumed CPU while the GH200 remained idle during the initial dynamic compilation. The upstream launcher default is still `true`; the override is isolated to the one-GH200 Roihu adapter and does not alter the DC-GRPO method.

## Follow-up alignment audit (2026-10-16)

The current corrected pair is not a paper-equivalent result. Jobs `2348912` and `2348913` use one GH200, 20 training steps, response length 2048, two rollouts per prompt, prompt batch 16, mini-batch 4, SDPA, and FSDP2 CPU offload. These settings are required to keep Qwen3-8B inside one Roihu GH200, but they differ from the paper/README target of eight GPUs, response length 8192, eight rollouts per prompt, train batch 64, PPO mini-batch 16, and the full training schedule. The DC-GRPO metric from `2348912` is therefore a pilot metric, not evidence for the paper's 63.2 result.

The current CSC Roihu documentation confirms that eight GPUs require two four-GH200 nodes in `gpularge`, and that `gpularge` access requires a scalability test. The next exact-alignment submission must first verify that access and implement a multi-node Ray launch; it must not silently reuse the one-GH200 pilot script. Until then, only a clearly labelled hardware-adapted gate may be submitted.

The latest local gates still pass (`validate_dcmdp_setup.py`, all reproduction shell scripts with `bash -n`, and `git diff --check`). No new job was submitted during this audit because `roihu-gpu.csc.fi` currently rejects the connection (`Connection refused`), while `roihu-cpu.csc.fi` still rejects the available SSH certificate. Existing jobs and outputs were not cancelled or modified.

## Claude Scholar code-review follow-up (2026-10-16)

- **Fixed, medium severity:** `GRPOLagrangianOperator.apply_to_advantages()` recorded token-level constraint statistics twice: once after flattening the token tensor and again through the normal path. This duplicated observations in buffers larger than one and could bias the dual update toward repeated batches. The redundant call was removed; the paper configuration uses `buffer_size=1`, so the already-completed pilot metric is unchanged.
- Local verification after the fix passed: `validate_dcmdp_setup.py`, Python compilation of the changed module, `bash -n` for all reproduction launchers, and `git diff --check`. The local environment does not provide `pytest`, so the focused pytest suite remains unexecuted locally and is not claimed as passed.

## Paper-aligned 8-GPU submission design (2026-10-16)

- The README's Qwen3-8B BF16 main configuration uses `DATA_TYPE=dapo_2080`, which resolves to `data/train/dapo_train_qwen3-8b_20_80_6170.parquet`; the earlier pilot used the full 17,398-row parquet and is therefore also data-mismatched relative to the operational README command.
- Added `scripts/repro/roihu_aime24_paper_8gpu.sbatch` for the documented two-node Roihu allocation (`gpularge`, 4 GH200 GPUs and 288 CPUs per node). It keeps the paper/README knobs: BF16, response length 8192, prompt batch 64, 8 rollouts, PPO mini-batch 16, 301 steps, `c=0.0037`, `lambda_init=0.1`, `lambda_lr=1.0`, `token_prob_diff`, `absneg_0.0_none`, and `abs_mean`.
- The script starts a two-node Ray cluster and the launcher now accepts `RAY_ADDRESS` through `+ray_kwargs.ray_init.address=...`. Roihu-specific SDPA and `use_remove_padding=false` remain explicitly documented hardware adaptations. The script has not been submitted because gpularge/scalability access and SSH authentication are not currently verifiable.
- Local validation now additionally checks the 6,170-row README dataset and the 8-GPU script; `validate_dcmdp_setup.py`, Python compilation, `bash -n`, and `git diff --check` pass.
- Follow-up configuration review found that the paper script relied on the launcher's implicit BF16 default. The script now passes `DTYPE=bfloat16` explicitly, and the local validator asserts this setting so an environment override cannot silently change the Table 1 regime.

## Official CMDP entrypoint and Triton portability audit (2026-10-16)

- The authoritative training entrypoint is `examples/CMDP/GRPO_qwen3_8b_lag_v5.sh`. Its default `NNODES=1` and `NGPUS_PER_NODE=8` match a single Triton H200 node for the paper's eight-GPU Qwen3-8B run; the launcher passes the training, rollout, optimizer, validation, and DC-GRPO arguments into `verl.trainer.main_ppo`.
- `scripts/repro/roihu_aime24_paper_8gpu.sbatch` is not a Triton submission script. It binds Roihu `/scratch/project_2015042` and `/projappl/project_2015042` paths, uses a Roihu ARM64 Apptainer image, and applies the Roihu SDPA/remove-padding adaptation. It must not be submitted on Triton.
- A Triton wrapper must be finalized only after login-time checks identify the available Python/vLLM environment, model-cache location, dataset path, Slurm account, and H200 partition. Until those checks are available, the existing Roihu wrapper is deliberately left untouched and no Triton job is submitted.
- Added `scripts/repro/triton_aime24_smoke.sbatch` as the first Triton gate. Triton's live partition inventory shows that `gpu-debug` has A100/V100 nodes but no H200, so the gate requests one H200 from `gpu-h200-141g-short` for 15 minutes. It loads `scicomp-llm-env`, adds the project-local `runtime/python` directory for the repository's pinned FlashAttention wheel, requires `vllm==0.11.0` and importable FlashAttention, uses the shared read-only Qwen3-8B snapshot `9c925d64d72725edaf899c6cb9c377fd0709d9c5`, invokes the official CMDP launcher with a bounded one-GPU smoke configuration, and writes a unique `COMPLETED` marker. It fails on environment mismatch or a missing shared model instead of silently downloading or changing the paper regime; the `aalto_users` account and H200 partition remain subject to login-time verification.
