#!/usr/bin/env python3
"""Verify the compiler environment used by Triton and TorchInductor."""

from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path


def main() -> None:
    cc = os.environ.get("CC")
    cxx = os.environ.get("CXX")
    gcc = shutil.which("gcc")
    gxx = shutil.which("g++")
    assert cc == "/usr/bin/gcc", cc
    assert cxx == "/usr/bin/g++", cxx
    assert gcc == "/usr/bin/gcc", gcc
    assert gxx == "/usr/bin/g++", gxx
    assert Path(cc).is_file()
    assert Path(cxx).is_file()
    print(f"compiler_cc={cc}")
    print(f"compiler_cxx={cxx}")
    print(f"sysconfig_cc={sysconfig.get_config_var('CC')}")
    print(f"sysconfig_cxx={sysconfig.get_config_var('CXX')}")
    print("COMPILER_ENV=PASS")


if __name__ == "__main__":
    main()
