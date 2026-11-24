"""Package the serving sources and artifacts produced by the ExecuTorch build."""

from __future__ import annotations

import ast
import json
import os
import platform
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/src/muse_glimmer"
REWRITES = {
    "executorch.examples.llm_server.python": "muse_glimmer._vendor.llm_server",
    "executorch.examples.models.muse_glimmer.serving": "muse_glimmer._vendor.muse",
}


def gcc_runtime_notices(openmp: bool, share: Path = Path("/usr/share")) -> list[str]:
    components = "libstdc++, libgcc" + (", libgomp" if openmp else "")
    # GCC's runtime libraries share GPLv3 with the GCC Runtime Library Exception.
    # RPMs can install the complete license pair with libgcc instead of libgomp.
    files = [ROOT / "packaging/licenses" / name for name in ("GCC-COPYING3", "GCC-RUNTIME-EXCEPTION")]
    if openmp:
        for package in ("libgomp", "libgcc", "gcc"):
            directory = share / "licenses" / package
            pair = [directory / name for name in ("COPYING3", "COPYING.RUNTIME")]
            if all(path.is_file() for path in pair):
                files = pair
                break
        debian_notice = share / "doc/libgomp1/copyright"
        if debian_notice.is_file():
            files.append(debian_notice)
    return [f"--- GNU runtime ({components})/{path.name} ---\n\n{path.read_text()}\n" for path in files]


def stage(source: Path, worker_dir: Path, *, editable: bool = False) -> None:
    # Validate the entire replacement before touching an existing editable install.
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=PACKAGE) as temporary:
        staged = Path(temporary)
        revision = _stage(source, worker_dir, staged, editable=editable)
        for name in ("_vendor", "_native"):
            destination = PACKAGE / name
            if destination.exists():
                shutil.rmtree(destination)
            (staged / name).rename(destination)
    print(f"Packaged {platform.system()} runtime from ExecuTorch {revision}")


def _stage(source: Path, worker_dir: Path, package: Path, *, editable: bool) -> str:
    acquisition = runpy.run_path(str(ROOT / "packaging/source.py"))
    metadata = acquisition["source_metadata"]()
    revision = metadata["revision"]
    acquisition["validate_checkout"](source, revision, editable=editable)
    if editable:
        local_changes = acquisition["local_source_hash"](source)
        if local_changes:
            metadata["local_changes"] = local_changes
    worker = worker_dir / "muse_glimmer_batching_worker"
    cuda = platform.system() == "Linux"
    if cuda:
        metadata["backend"] = "cuda"
        metadata["cuda_version"] = "13.2"
        metadata["cuda_architectures"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "native")
    artifacts = [worker] if cuda else [worker, worker_dir / "mlx.metallib"]
    for path in artifacts:
        if not path.is_file():
            raise ValueError(f"Missing native artifact: {path}")
    if not cuda:
        architecture = subprocess.check_output(["lipo", "-archs", str(worker)], text=True).strip()
        if architecture != "arm64":
            raise ValueError(f"Expected an arm64 worker, found {architecture}")
        links = subprocess.check_output(["otool", "-L", str(worker)], text=True)
        for line in links.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            if not dependency.startswith(("/System/Library/", "/usr/lib/")):
                raise ValueError(
                    f"Worker depends on an unbundled library: {dependency}. Build a self-contained worker first."
                )
    vendor = package / "_vendor"
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").write_text(
        '"""Generated from the pinned ExecuTorch source; do not edit here."""\n'
    )
    for original, destination in [
        ("examples/llm_server/python", "llm_server"),
        ("examples/models/muse-glimmer/serving", "muse"),
    ]:
        for path in sorted((source / original).rglob("*")):
            relative = path.relative_to(source / original)
            if "tests" in relative.parts or path.suffix not in (".py", ".jinja"):
                continue
            output = vendor / destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text()
            for old, new in REWRITES.items():
                text = text.replace(old, new)
            if path.suffix == ".py":
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    names = (
                        [node.module or ""]
                        if isinstance(node, ast.ImportFrom)
                        else [a.name for a in node.names]
                        if isinstance(node, ast.Import)
                        else []
                    )
                    if any(
                        name == "executorch" or name.startswith("executorch.") for name in names
                    ):
                        raise ValueError(f"Unpackaged upstream dependency in {path}")
            output.write_text(text)
    shutil.copy2(source / "LICENSE", vendor / "LICENSE")
    (vendor / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    native = package / "_native"
    native.mkdir()
    if cuda:
        linux = runpy.run_path(str(ROOT / "packaging/linux.py"))
        linux["bundle"](worker, native)
    else:
        for path in artifacts:
            shutil.copy2(path, native / path.name)
    notices = []
    license_roots = [
        *(acquisition["required_submodules"]() if cuda else acquisition["MLX_SUBMODULES"]),
        "extension/llm/tokenizers/third-party/abseil-cpp",
        "extension/llm/tokenizers/third-party/sentencepiece",
        "extension/llm/tokenizers/third-party/re2",
        "extension/llm/tokenizers/third-party/pcre2",
    ]
    build = worker_dir.parents[2]
    stb = Path(os.environ.get("MUSE_GLIMMER_STB_INCLUDE_DIR", build / "_deps/stb-src"))
    cache = build / "CMakeCache.txt"
    if "MUSE_GLIMMER_STB_INCLUDE_DIR" not in os.environ and cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("MUSE_GLIMMER_STB_INCLUDE_DIR:"):
                value = line.split("=", 1)[1]
                if value:
                    stb = Path(value)
                break
    fmt = build / "executorch/backends/mlx/mlx/_deps/fmt-src"
    fetched = [stb] if cuda else [stb, fmt]
    for directory in [*(source / relative for relative in license_roots), *fetched]:
        relative = (
            str(directory.relative_to(source))
            if directory.is_relative_to(source)
            else directory.name.removesuffix("-src")
        )
        paths = (
            sorted(
                p
                for p in directory.iterdir()
                if p.is_file()
                and p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE"))
            )
            if directory.is_dir()
            else []
        )
        if not paths:
            raise ValueError(f"Missing native dependency license: {directory}")
        for path in paths:
            notices.append(f"--- {relative}/{path.name} ---\n\n{path.read_text()}\n")
    if cuda:
        toolkit = Path(shutil.which("nvcc")).resolve().parents[1]
        eula = toolkit / "EULA.txt"
        if not eula.is_file():
            raise ValueError(f"Missing CUDA redistribution license: {eula}")
        notices.append(f"--- CUDA Toolkit EULA ---\n\n{eula.read_text()}\n")
        notices.extend(gcc_runtime_notices((native / "libgomp.so.1").is_file()))
    (native / "THIRD_PARTY_NOTICES").write_text("\n".join(notices))
    return revision
