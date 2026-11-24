"""Prepare build dependencies, then install from source like ExecuTorch's installer."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def install(*, editable: bool = False) -> None:
    env = os.environ.copy()
    requirements = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
    index = []
    if platform.system() == "Linux":
        toolkit = Path(env.get("CUDA_HOME", "/usr/local/cuda-13.2"))
        env["PATH"] = str(toolkit / "bin") + os.pathsep + env["PATH"]
        result = subprocess.run(["nvcc", "--version"], env=env, capture_output=True, text=True, check=True)
        if "release 13.2," not in result.stdout:
            raise RuntimeError("Source installation requires the CUDA 13.2 toolkit.")
        requirements = [item + "+cu132" if item.startswith("torch==") else item for item in requirements]
        index = ["--extra-index-url", "https://download.pytorch.org/whl/test/cu132"]
    subprocess.run([sys.executable, "-m", "pip", "install", *index, *requirements], env=env, check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation",
         *(["--editable"] if editable else []), str(ROOT)],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editable", action="store_true", help="Install using pip install -e .")
    install(editable=parser.parse_args().editable)
