"""Prepare build inputs online, then build the CUDA wheel in an offline container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "quay.io/pypa/manylinux_2_28_x86_64"
TAG = "manylinux_2_28_x86_64"


def prepare_stb(recipe: Path, dest: Path) -> None:
    override = os.environ.get("MUSE_GLIMMER_STB_INCLUDE_DIR")
    if override:
        source = Path(override).resolve()
    else:
        # Follow the same immutable FetchContent revision as the pinned ExecuTorch recipe.
        match = re.search(
            r"FetchContent_Declare\(\s*stb\s+GIT_REPOSITORY\s+(\S+)\s+GIT_TAG\s+([a-f0-9]{40})\s*\)",
            recipe.read_text(),
        )
        if not match:
            raise RuntimeError(f"Could not identify the pinned STB source in {recipe}")
        url, revision = match.groups()
        source = dest.parent / "stb-checkout"
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "--no-tags", url, revision], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", revision], check=True)

    # CMake accepts both upstream's deprecated/ layout and a flat include directory.
    resize = source / "deprecated/stb_image_resize.h"
    if not resize.is_file():
        resize = source / "stb_image_resize.h"
    dest.mkdir()
    for path in (source / "stb_image.h", resize, source / "LICENSE"):
        shutil.copyfile(path, dest / path.name)


def package_source_id(root: Path, wheelhouse: Path) -> str:
    """Identify Muse sources, including working-tree edits used by setuptools."""
    digest = hashlib.sha256()
    if (root / ".git").exists():
        digest.update(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root))
        listing = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
             ".", ":(exclude)third-party/executorch"], cwd=root,
        )
        files = {root / os.fsdecode(name) for name in listing.split(b"\0") if name}
    else:
        # An extracted sdist carries setuptools' manifest, including package data.
        manifest = root / "python/src/muse_glimmer.egg-info/SOURCES.txt"
        files = {root / name for name in manifest.read_text().splitlines()}
        # setuptools discovers new modules too when rebuilding from a modified sdist.
        for name in ("python/src/muse_glimmer", "build_backend", "packaging", "tokenizer", "tests"):
            files.update((root / name).rglob("*"))
    for file in sorted(files):
        relative = file.relative_to(root)
        if file.is_relative_to(wheelhouse):
            cached = file.relative_to(wheelhouse)
            if file.suffix == ".whl" or (cached.parts and re.fullmatch("[0-9a-f]{40}", cached.parts[0])):
                continue
        if relative.parts[0] == "third-party" or not file.is_file():
            continue
        if any(p.endswith(".egg-info") or p in ("__pycache__", "_native", "_vendor") for p in relative.parts):
            continue
        digest.update(os.fsencode(relative) + b"\0")
        digest.update(b"x" if file.stat().st_mode & 0o111 else b"-")
        link = os.readlink(file) if file.is_symlink() else ""
        digest.update(os.fsencode(link) + b"\0")
        with file.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def archive_record(archive: Path, source_id: str) -> dict[str, str]:
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"muse_glimmer_source_id": source_id, "archive_sha256": checksum}


def source_archive(wheelhouse: Path, source_id: str) -> Path:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    revision = runpy.run_path(str(ROOT / "packaging/source.py"))["source_metadata"]()["revision"]
    return wheelhouse / revision / source_id / f"muse_glimmer-{version}.tar.gz"


def prepare(wheelhouse: Path) -> None:
    requirements = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
    wheelhouse.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "download", "--only-binary=:all:",
        "--implementation", "cp", "--python-version", "3.11", "--abi", "cp311",
        "--dest", str(wheelhouse),
    ]
    for minor in range(28, 16, -1):
        command.extend(["--platform", f"manylinux_2_{minor}_x86_64"])
    for legacy in ("manylinux2014", "manylinux2010", "manylinux1"):
        command.extend(["--platform", f"{legacy}_x86_64"])
    subprocess.run([*command, "build>=1.2,<2", *requirements], check=True)

    source_id = package_source_id(ROOT, wheelhouse)
    archive = source_archive(wheelhouse, source_id)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent) as tmp:
        stage = Path(tmp)
        source = runpy.run_path(str(ROOT / "packaging/source.py"))["source_checkout"]()
        prepare_stb(source / "examples/models/muse-glimmer/CMakeLists.txt", stage / "stb")
        env = stage / "env"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        python = str(env / "bin/python")
        subprocess.run([python, "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse),
                        "setuptools>=77", "wheel>=0.45"], check=True)
        subprocess.run([python, "setup.py", "sdist", "--cuda-stb-dir", str(stage / "stb"),
                        "--dist-dir", str(stage)], cwd=ROOT, check=True)
        if source_archive(wheelhouse, package_source_id(ROOT, wheelhouse)) != archive:
            raise RuntimeError("Muse-Glimmer sources changed during preparation. Rerun --prepare-only.")
        record = stage / "source.json"
        record.write_text(json.dumps(archive_record(stage / archive.name, source_id)) + "\n")
        (stage / archive.name).replace(archive)
        record.replace(archive.parent / "source.json")


def build(image: str, wheelhouse: Path, output: Path) -> None:
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("The manylinux CUDA builder requires Linux x86_64.")
    source_id = package_source_id(ROOT, wheelhouse)
    archive = source_archive(wheelhouse, source_id)
    record = archive.parent / "source.json"
    if not archive.is_file() or not record.is_file() or not list(wheelhouse.glob("torch-*.whl")):
        raise RuntimeError("Run --prepare-only for this Muse-Glimmer checkout to stage its sources and dependencies.")
    if json.loads(record.read_text()) != archive_record(archive, source_id):
        raise RuntimeError("Prepared archive differs from this checkout or its checksum. Rerun --prepare-only.")
    subprocess.run(["podman", "image", "exists", image], check=True)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        raise RuntimeError(f"Wheel output must be empty: {output}")
    toolkit = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")).resolve()
    if not (toolkit / "bin/nvcc").is_file():
        raise RuntimeError(f"CUDA toolkit not found: {toolkit}")
    cache = ROOT / "build/manylinux/native-cache"
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        "podman", "run", "--rm", "--pull=never", "--network=none",
        "--userns=keep-id", "--security-opt", "label=disable", "--workdir", "/tmp",
        "--volume", f"{toolkit}:/usr/local/cuda-13.2:ro",
        "--volume", f"{wheelhouse}:/inputs/wheels:ro",
        "--volume", f"{archive}:/inputs/{archive.name}:ro",
        "--volume", f"{cache}:/native-cache:rw",
        "--volume", f"{output}:/output:rw",
        "--env", "CUDA_HOME=/usr/local/cuda-13.2",
        "--env", "MUSE_GLIMMER_STB_INCLUDE_DIR=/tmp/source/third-party/stb",
        "--env", "MUSE_GLIMMER_BUILD_CACHE=/native-cache",
        "--env", "TORCH_CUDA_ARCH_LIST=" + os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0 8.6 8.9 9.0 10.0 12.0+PTX"),
        "--env", "CMAKE_BUILD_PARALLEL_LEVEL=" + os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8"),
    ]
    runtime = os.environ.get("MUSE_GLIMMER_CXX_RUNTIME_DIR")
    if runtime:
        command.extend(["--volume", f"{Path(runtime).resolve()}:/inputs/cxx-runtime:ro",
                        "--env", "MUSE_GLIMMER_CXX_RUNTIME_DIR=/inputs/cxx-runtime"])
    script = """set -eu
export PATH="$CUDA_HOME/bin:$PATH"
/opt/python/cp311-cp311/bin/python -m venv /tmp/build-env
export PATH="/tmp/build-env/bin:$PATH"
mkdir -p /tmp/source
tar -xzf "$1" --strip-components=1 -C /tmp/source
python -m pip wheel --no-index --find-links /inputs/wheels --no-deps --wheel-dir /tmp/raw /tmp/source
auditwheel repair --plat "$2" --only-plat --wheel-dir /output /tmp/raw/*.whl
"""
    subprocess.run([*command, image, "/bin/bash", "-c", script, "build", f"/inputs/{archive.name}", TAG], check=True)
    wheels = list(output.glob(f"*-{TAG}.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one audited manylinux wheel")
    subprocess.run([sys.executable, str(ROOT / "packaging/check_wheel.py"), str(wheels[0])], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "build/manylinux/wheelhouse")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/manylinux")
    parser.add_argument("--prepare-only", action="store_true", help="Fetch Python wheels, ExecuTorch sources and STB before the offline build")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.wheelhouse.resolve())
    else:
        build(args.image, args.wheelhouse.resolve(), args.output.resolve())
