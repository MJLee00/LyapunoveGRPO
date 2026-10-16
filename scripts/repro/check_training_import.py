#!/usr/bin/env python3
"""Import the exact training entry point without allocating a Ray cluster."""

from verl.trainer.main_ppo import main

print("TRAINING_IMPORT=PASS")
print(f"main_callable={main.__module__}.{main.__name__}")
