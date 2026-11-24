"""Check the wheel's native payload and install it without build tools."""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import subprocess
import tempfile
import tomllib
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_wheel(wheel: Path) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    minimum = project["tool"]["muse-glimmer"]["macos-deployment-target"]
    version = project["project"]["version"]
    target = tuple(map(int, minimum.split(".")))
    cuda_tags = {"py3-none-linux_x86_64", "py3-none-manylinux_2_28_x86_64"}
    candidate_tag = wheel.name.removeprefix(f"muse_glimmer-{version}-").removesuffix(".whl")
    cuda = candidate_tag in cuda_tags
    tag = candidate_tag if cuda else f"py3-none-macosx_{target[0]}_0_arm64"
    if wheel.name != f"muse_glimmer-{version}-{tag}.whl":
        raise ValueError(f"Expected muse-glimmer {version} ({tag}), got {wheel.name}")
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as directory:
        # Setuptools can place the package at the root or under .data/purelib.
        worker_name = next(
            name
            for name in archive.namelist()
            if name.endswith("muse_glimmer/_native/muse_glimmer_batching_worker")
        )
        package = worker_name.rsplit("/_native/", 1)[0]
        for name in (
            "_native/muse_glimmer_batching_worker",
            *(() if cuda else ("_native/mlx.metallib",)),
            "_native/THIRD_PARTY_NOTICES",
            "_vendor/LICENSE",
            "_vendor/muse/serve_batching.py",
            "_vendor/llm_server/server.py",
            "default_config.toml",
            "models.yaml",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/chat_template.jinja",
        ):
            if not archive.getinfo(f"{package}/{name}").file_size:
                raise ValueError(f"Empty wheel resource: {name}")
        provenance = json.loads(archive.read(f"{package}/_vendor/provenance.json"))
        if (
            not re.fullmatch(r"[0-9a-f]{40}", provenance["revision"])
            or "local_changes" in provenance
        ):
            raise ValueError("Release wheels require a clean, pinned ExecuTorch revision")
        info = archive.getinfo(worker_name)
        if not (info.external_attr >> 16) & 0o111:
            raise ValueError("Packaged worker is not executable")
        worker = Path(directory) / "worker"
        worker.write_bytes(archive.read(info))
        if cuda:
            native = Path(directory) / "native"
            native.mkdir()
            for name in archive.namelist():
                if name.startswith(f"{package}/_native/") and not name.endswith("/"):
                    filename = name.rsplit("/", 1)[1]
                    (native / filename).write_bytes(archive.read(name))
            linux = runpy.run_path(str(ROOT / "packaging/linux.py"))
            linux["check_elf"](native / "muse_glimmer_batching_worker", bundled=True)
            for binary in native.glob("*.so*"):
                linux["check_elf"](binary, bundled=True)
            if provenance.get("backend") != "cuda" or provenance.get("cuda_version") != "13.2":
                raise ValueError("Linux wheels require CUDA 13.2 provenance")
        else:
            architecture = subprocess.check_output(
                ["lipo", "-archs", str(worker)], text=True
            ).strip()
            if architecture != "arm64":
                raise ValueError(f"Packaged worker has the wrong architecture: {architecture}")
            load_commands = subprocess.check_output(["otool", "-l", str(worker)], text=True)
            versions = re.findall(
                r"LC_BUILD_VERSION.*?minos (\d+)\.(\d+)", load_commands, re.DOTALL
            )
            metal = archive.read(f"{package}/_native/mlx.metallib")
            metal_versions = re.findall(rb"air64[\w.]*-apple-macosx(\d+)\.(\d+)", metal)
            for name, values in (("worker", versions), ("Metal library", metal_versions)):
                if {tuple(map(int, value)) for value in values} != {target}:
                    raise ValueError(
                        f"{name} must be compiled for the macOS {minimum} deployment target"
                    )
            links = subprocess.check_output(["otool", "-L", str(worker)], text=True)
            for line in links.splitlines()[1:]:
                dependency = line.strip().split(" (", 1)[0]
                if not dependency.startswith(("/System/Library/", "/usr/lib/")):
                    raise ValueError(f"Wheel depends on an unbundled library: {dependency}")
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        if f"Tag: {tag}" not in archive.read(metadata).decode():
            raise ValueError("Wheel metadata and filename tags disagree")
        metadata = metadata.removesuffix("WHEEL") + "METADATA"
        distribution = Parser().parsestr(archive.read(metadata).decode())
        if distribution["Name"] != "muse-glimmer" or distribution["Version"] != version:
            raise ValueError("Wheel metadata and release version disagree")
    print(f"Verified {wheel.name}: native payload and bundled dependencies.")


def test_install(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="muse-glimmer-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin/python"
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        env.update(PATH=str(environment / "bin"), DEVELOPER_DIR=str(root / "no-xcode"))
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                str(wheel.resolve()),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                """
import importlib.util
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from muse_glimmer.config import load_settings
from muse_glimmer.resources import resource_path
from muse_glimmer.runtime import native_dir
from transformers import AutoTokenizer

assert all(shutil.which(tool) is None for tool in ('git', 'cmake', 'clang', 'xcrun', 'make'))
assert importlib.util.find_spec('torch') is None
assert native_dir().resolve().is_relative_to(Path(sys.prefix).resolve())
assert resource_path('default_config.toml').resolve().is_relative_to(Path(sys.prefix).resolve())
load_settings({})
tokenizer = AutoTokenizer.from_pretrained(resource_path('tokenizer'), local_files_only=True)
assert tokenizer.decode(tokenizer.encode('Hello.', add_special_tokens=False)) == 'Hello.'
worker = subprocess.run([str(native_dir() / 'muse_glimmer_batching_worker'), '--help'], capture_output=True, text=True)
assert worker.returncode in (0, 1) and 'prefix_checkpoint_prompt_end' in worker.stdout + worker.stderr, worker.stderr
assert subprocess.check_output(['muse-glimmer', '--version'], text=True).strip() == version('muse-glimmer')
subprocess.run(['muse-glimmer', 'serve', '--help'], check=True)
print('Wheel installed and native worker started without source/build tools or PyTorch.')
""",
            ],
            cwd=root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--install", action="store_true", help="Test a fresh binary-only installation"
    )
    args = parser.parse_args()
    check_wheel(args.wheel)
    if args.install:
        test_install(args.wheel)
