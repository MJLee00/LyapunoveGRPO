#!/usr/bin/env python3
"""Print the exact runtime versions used by the Roihu reproduction."""

import importlib.metadata as metadata


PACKAGES = (
    "torch",
    "vllm",
    "transformers",
    "ray",
    "sglang",
    "tensordict",
    "torchdata",
    "flashinfer-python",
    "math-verify",
    "datasets",
    "hydra-core",
    "megatron-core",
    "wandb",
    "pyarrow",
    "pandas",
)


for package in PACKAGES:
    print(f"{package}={metadata.version(package)}")

import torch

print(f"python_cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")

import verl

print(f"verl={verl.__version__}")
print(f"verl_path={verl.__file__}")
