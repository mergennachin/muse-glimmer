"""Package the serving sources and artifacts produced by the ExecuTorch build."""

from __future__ import annotations

import ast
import json
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
    print(f"Packaged Mac runtime from ExecuTorch {revision}")


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
    metallib = worker_dir / "mlx.metallib"
    for path in [worker, metallib]:
        if not path.is_file():
            raise ValueError(f"Missing native artifact: {path}")
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
    for path in [worker, metallib]:
        shutil.copy2(path, native / path.name)
    notices = []
    license_roots = [
        *acquisition["MLX_SUBMODULES"],
        "extension/llm/tokenizers/third-party/abseil-cpp",
        "extension/llm/tokenizers/third-party/sentencepiece",
        "extension/llm/tokenizers/third-party/re2",
        "extension/llm/tokenizers/third-party/pcre2",
    ]
    stb = worker_dir.parents[2] / "_deps/stb-src"
    for directory in [*(source / relative for relative in license_roots), stb]:
        relative = str(directory.relative_to(source)) if directory.is_relative_to(source) else "stb"
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
    (native / "THIRD_PARTY_NOTICES").write_text("\n".join(notices))
    return revision
