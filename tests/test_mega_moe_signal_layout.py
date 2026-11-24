"""Compile the actual host/device workspace accessors without using a GPU."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_capacity_sized_signal_layout(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent
    native = source.parent
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    compiler = shutil.which("c++")
    cutlass = native / "third-party/cutlass/include"
    if compiler is None or not (cuda / "include/cuda_runtime.h").is_file() or not cutlass.is_dir():
        pytest.skip("C++ compiler, CUDA headers and initialized CUTLASS submodule are required")
    binary = tmp_path / "signal-layout"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fno-strict-aliasing",
            "-Wno-deprecated-declarations",
            "-I" + str(native / "deep_gemm/include"),
            "-I" + str(cutlass),
            "-I" + str(cuda / "include"),
            "-I" + str(cuda / "include/cccl"),
            str(source / "test_mega_moe_signal_layout.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=15)
    assert result.stdout == "54 workspace layouts passed\n"
