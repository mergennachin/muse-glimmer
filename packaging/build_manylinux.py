"""Build a CUDA wheel offline in a preloaded manylinux container and audit it."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "quay.io/pypa/manylinux_2_28_x86_64"
TAG = "manylinux_2_28_x86_64"


def prepare(wheelhouse: Path) -> None:
    requirements = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
    requirements = ["torch==2.15.0+cu132" if item == "torch==2.15.0" else item for item in requirements]
    wheelhouse.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "download", "--only-binary=:all:",
        "--implementation", "cp", "--python-version", "3.11", "--abi", "cp311",
        "--extra-index-url", "https://download.pytorch.org/whl/cu132",
        "--dest", str(wheelhouse),
    ]
    for minor in range(28, 16, -1):
        command.extend(["--platform", f"manylinux_2_{minor}_x86_64"])
    command.extend(["--platform", "manylinux2014_x86_64", "build>=1.2,<2", *requirements])
    subprocess.run(command, check=True)


def build(image: str, wheelhouse: Path, output: Path) -> None:
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("The manylinux CUDA builder requires Linux x86_64.")
    subprocess.run(["podman", "image", "exists", image], check=True)
    if not list(wheelhouse.glob("torch-2.15.0+cu132-*.whl")):
        raise RuntimeError("Prepare the CPython 3.11 CUDA 13.2 wheelhouse first.")
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        raise RuntimeError(f"Wheel output must be empty: {output}")
    raw = ROOT / "build/manylinux/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if list(raw.glob("*.whl")):
        raise RuntimeError(f"Raw wheel output must be empty: {raw}")
    toolkit = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")).resolve()
    if not (toolkit / "bin/nvcc").is_file():
        raise RuntimeError(f"CUDA toolkit not found: {toolkit}")
    mounts = {ROOT.parent: ROOT.parent}
    # Preserve local Git object alternates and submodule URLs in this workspace.
    aliases = subprocess.run(
        ["git", "-C", str(ROOT), "config", "--local", "--get-regexp", r"submodule\..*\.url"],
        capture_output=True, text=True,
    )
    for line in aliases.stdout.splitlines():
        value = line.split(" ", 1)[1]
        if value.startswith("/"):
            alias = Path(value).parent
            if alias.is_dir() and alias.resolve() == ROOT.parent:
                mounts[alias] = ROOT.parent
    command = [
        "podman", "run", "--rm", "--pull=never", "--network=none",
        "--userns=keep-id", "--security-opt", "label=disable",
        "--workdir", str(ROOT),
        "--volume", f"{toolkit}:/usr/local/cuda-13.2:ro",
    ]
    for destination, source in mounts.items():
        command.extend(["--volume", f"{source}:{destination}:rw"])
    cache = ROOT / "build/manylinux/native-cache"
    env = {
        "CUDA_HOME": "/usr/local/cuda-13.2",
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0 8.6 8.9 9.0 10.0 12.0+PTX"),
        "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8"),
        "MUSE_GLIMMER_BUILD_CACHE": str(cache),
    }
    if os.environ.get("MUSE_GLIMMER_STB_INCLUDE_DIR"):
        env["MUSE_GLIMMER_STB_INCLUDE_DIR"] = os.environ["MUSE_GLIMMER_STB_INCLUDE_DIR"]
    for name, value in env.items():
        command.extend(["--env", f"{name}={value}"])
    quote = shlex.quote
    script = "\n".join([
        "set -eu",
        'export PATH="$CUDA_HOME/bin:$PATH"',
        "/opt/python/cp311-cp311/bin/python -m venv /tmp/muse-glimmer-build-env",
        f"/tmp/muse-glimmer-build-env/bin/python -m pip install --no-index --only-binary=:all: --find-links {quote(str(wheelhouse))} "
        "build setuptools wheel 'cmake<4' pyyaml 'torch==2.15.0+cu132' patchelf",
        'export PATH="/tmp/muse-glimmer-build-env/bin:$PATH"',
        f"python -m build --wheel --no-isolation --outdir {quote(str(raw))}",
        f"auditwheel repair --plat {TAG} --only-plat --wheel-dir {quote(str(output))} {quote(str(raw))}/*.whl",
    ])
    subprocess.run([*command, image, "/bin/bash", "-c", script], check=True)
    wheels = list(output.glob(f"*-{TAG}.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one audited manylinux wheel")
    subprocess.run([sys.executable, str(ROOT / "packaging/check_wheel.py"), str(wheels[0])], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "build/manylinux/wheelhouse")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/manylinux")
    parser.add_argument("--prepare-only", action="store_true", help="Download build wheels without starting the container")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.wheelhouse.resolve())
    else:
        build(args.image, args.wheelhouse.resolve(), args.output.resolve())
