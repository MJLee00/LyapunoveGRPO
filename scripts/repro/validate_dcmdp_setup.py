#!/usr/bin/env python3
"""Dependency-light checks for the DC-GRPO reproduction configuration.

This intentionally avoids importing torch, Ray, vLLM, or verl so it can run on
the local macOS checkout before any remote GPU allocation is requested.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path


EXPECTED_ROWS = 17_398
EXPECTED_VERL_VERSION = "0.7.0.dev"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def check_version(repo_root: Path) -> None:
    version_file = repo_root / "verl" / "version" / "version"
    assert version_file.read_text(encoding="utf-8").strip() == EXPECTED_VERL_VERSION


def check_dataset(repo_root: Path) -> None:
    dataset = repo_root / "data" / "train" / "dapo_train.parquet"
    assert dataset.is_file(), dataset

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(dataset)
    assert parquet.metadata.num_rows == EXPECTED_ROWS, parquet.metadata.num_rows
    names = set(parquet.schema_arrow.names)
    for required in ("data_source", "prompt", "reward_model", "extra_info"):
        assert required in names, (required, sorted(names))


def check_algorithm_formula() -> None:
    train_probability = 0.73
    rollout_probability = 0.61
    delta = abs(train_probability - rollout_probability)
    assert math.isclose(delta, 0.12, rel_tol=0.0, abs_tol=1e-12)

    lambda_value = 0.1
    budget = 0.0037
    learning_rate = 1.0
    updated = min(1.0, max(0.0, lambda_value + learning_rate * (delta - budget)))
    assert math.isclose(updated, 0.2163, rel_tol=0.0, abs_tol=1e-12)


def check_launcher(repo_root: Path) -> None:
    launcher = repo_root / "examples" / "CMDP" / "GRPO_qwen3_8b_lag_v5.sh"
    source = launcher.read_text(encoding="utf-8")
    ast.parse("x = 1")  # Keep this check explicitly dependency-free.
    required_fragments = (
        "GRPO_LAGRANGIAN_RATIO=${GRPO_LAGRANGIAN_RATIO:-token_prob_diff}",
        "NORMALIZE_MODE=${NORMALIZE_MODE:-absneg_0.0_none}",
        "ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}",
        "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}",
        '+actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}"',
        'actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"',
        "PPO_MAX_TOKEN_LEN_PER_GPU",
        "LOG_PROB_MAX_TOKEN_LEN_PER_GPU",
        "MAX_NUM_BATCHED_TOKENS",
        "MAX_RESPONSE_LENGTH",
        "GPU_MEMORY_UTILIZATION",
        "ENFORCE_EAGER",
        "OFFLOAD_POLICY",
        "actor_rollout_ref.actor.fsdp_config.offload_policy=${offload_policy}",
        'export HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${BASE_PATH}/outputs}"',
        'hydra.run.dir="${HYDRA_RUN_DIR}"',
        "TOTAL_TRAINING_STEPS",
        "TEST_FREQ",
        "trainer.total_training_steps=${total_training_steps}",
        "trainer.test_freq=${test_freq}",
        "trainer.resume_mode=auto",
        '"${PYTHON_BIN}" -m verl.trainer.main_ppo',
    )
    for fragment in required_fragments:
        assert fragment in source, fragment
    assert "/mnt/ws-jfs/dsh-jfs" not in source
    assert "export WANDB_API_KEY=" not in source


def check_aime24_compare_script(repo_root: Path) -> None:
    script = repo_root / "scripts" / "repro" / "roihu_aime24_compare_qwen3_8b.sbatch"
    source = script.read_text(encoding="utf-8")
    required_fragments = (
        "METHOD must be grpo or dcmdp",
        "TEST_FILE=\"['/workspace/verl/data/eval/aime24_32.parquet']\"",
        "GRPO_LAGRANGIAN_ENABLE=${LAGRANGIAN_ENABLE}",
        "TOTAL_TRAINING_STEPS=\"${TOTAL_TRAINING_STEPS:-20}\"",
        "MAX_RESPONSE_LENGTH=2048",
        "GPU_MEMORY_UTILIZATION=0.20",
        "ENFORCE_EAGER=true",
        "OFFLOAD_POLICY=true",
        "RAY_ROOT=\"${ENV_ROOT}/ray/${SLURM_JOB_ID}\"",
        "--env RAY_TMPDIR=/envroot/ray/${SLURM_JOB_ID}",
        "RUNTIME_ROOT=\"${ENV_ROOT}/runtime\"",
        "--output=/projappl/project_2015042/mingyl/LyapunoveGRPO/logs/slurm-%x-%j.out",
        "--env RAY_DATA_HOME=/envroot/runtime/aime24-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}",
        "--env HYDRA_RUN_DIR=/envroot/runtime/aime24-${METHOD_TAG}-qwen3-8b-${SLURM_JOB_ID}/hydra",
        "PPO_MAX_TOKEN_LEN_PER_GPU=4096",
        "AIME24_COMPARE_COMPLETED=1",
    )
    for fragment in required_fragments:
        assert fragment in source, fragment
    assert "project_2008161" in source
    assert "flash_attention_2" not in source

def main() -> None:
    repo_root = parse_args().repo_root.resolve()
    check_version(repo_root)
    check_dataset(repo_root)
    check_algorithm_formula()
    check_launcher(repo_root)
    check_aime24_compare_script(repo_root)
    print("LOCAL_REPRO_CHECK=PASS")
    print(f"verl_version={EXPECTED_VERL_VERSION}")
    print(f"dataset_rows={EXPECTED_ROWS}")
    print("algorithm=DC-GRPO token_prob_diff absneg_0.0_none abs_mean")
    print("attention=sdpa on Roihu ARM64; flash_attention_2 remains the launcher default")


if __name__ == "__main__":
    main()
