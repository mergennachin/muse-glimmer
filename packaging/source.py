"""Initialize and validate the ExecuTorch submodule; never edit its sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = "third-party/executorch"
MLX_SUBMODULES = [
    "backends/mlx/third-party/mlx",
    "backends/xnnpack/third-party/FP16",
    "backends/xnnpack/third-party/FXdiv",
    "backends/xnnpack/third-party/cpuinfo",
    "backends/xnnpack/third-party/pthreadpool",
    "extension/llm/tokenizers",
    "kernels/optimized/third-party/eigen",
    "third-party/flatbuffers",
    "third-party/flatcc",
    "third-party/gflags",
    "third-party/json",
    "third-party/pocketfft",
]


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True)


def source_metadata() -> dict[str, str]:
    if (ROOT / ".git").exists():
        entry = git(ROOT, "ls-files", "--stage", "--", SUBMODULE).split()
        if len(entry) != 4 or entry[0] != "160000" or entry[2] != "0":
            raise RuntimeError(f"{SUBMODULE} must have a resolved submodule gitlink")
        repository = git(
            ROOT, "config", "--file", ".gitmodules", "--get", f"submodule.{SUBMODULE}.url"
        ).strip()
        return {"repository": repository, "revision": entry[1]}
    path = ROOT / "third-party/executorch-source.json"
    if not path.is_file():
        raise RuntimeError("ExecuTorch pin is missing; use a Git checkout or source distribution")
    metadata = json.loads(path.read_text())
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("repository"), str)
        or not metadata["repository"]
        or not re.fullmatch(r"[a-f0-9]{40}", str(metadata.get("revision", "")))
    ):
        raise RuntimeError(f"Invalid ExecuTorch source metadata: {path}")
    return metadata


def validate_checkout(source: Path, revision: str, *, editable: bool = False) -> None:
    if not (source / "CMakePresets.json").is_file():
        raise RuntimeError(f"ExecuTorch source not found at {source}")
    if not (source / ".git").exists():
        manifest = source.parent / "executorch-source.json"
        if not manifest.is_file() or json.loads(manifest.read_text())["revision"] != revision:
            raise RuntimeError("ExecuTorch source archive is missing its pinned revision metadata")
        return
    if git(source, "rev-parse", "HEAD").strip() != revision:
        raise RuntimeError(
            f"ExecuTorch at {source} differs from the submodule pin {revision}; checkout preserved"
        )
    if not editable and git(source, "status", "--porcelain", "--untracked-files=all").strip():
        raise RuntimeError(f"ExecuTorch at {source} must be clean; local changes were preserved")
    if any(
        line.startswith(("+", "U"))
        for line in git(source, "submodule", "status", "--recursive").splitlines()
    ):
        raise RuntimeError(f"ExecuTorch submodules at {source} differ from their pinned revisions")


def initialize_submodules(source: Path) -> None:
    if not (source / ".gitmodules").is_file():
        return
    configured = git(
        source, "config", "--file", ".gitmodules", "--get-regexp", r"submodule\..*\.path"
    )
    registered = {line.split(" ", 1)[1] for line in configured.splitlines()}
    selected = [path for path in MLX_SUBMODULES if path in registered]
    if not selected:
        return
    for line in git(source, "submodule", "status", "--recursive", "--", *selected).splitlines():
        if not line.startswith("-"):
            continue
        missing = source / line.split()[1]
        parent = missing.parent
        while parent != source and not (parent / ".git").exists():
            parent = parent.parent
        print(f"Initializing {missing.relative_to(source)}", flush=True)
        git(
            parent,
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--depth",
            "1",
            "--",
            str(missing.relative_to(parent)),
        )


def source_checkout(*, editable: bool = False) -> Path:
    revision = source_metadata()["revision"]
    source = ROOT / SUBMODULE
    if (ROOT / ".git").exists() and not (source / ".git").exists():
        if source.exists() and any(source.iterdir()):
            raise RuntimeError(
                f"{source} is not an initialized submodule; existing files were preserved"
            )
        print(f"Initializing {SUBMODULE}", flush=True)
        git(ROOT, "submodule", "update", "--init", "--depth", "1", "--", SUBMODULE)
    validate_checkout(source, revision, editable=editable)
    if (source / ".git").exists():
        initialize_submodules(source)
        validate_checkout(source, revision, editable=editable)
    return source


def repositories(source: Path) -> list[Path]:
    return [source] + [
        source / line.split()[1]
        for line in git(source, "submodule", "status", "--recursive").splitlines()
        if not line.startswith("-")
    ]


def untracked_sources(source: Path) -> list[Path]:
    return [
        repository / name
        for repository in repositories(source)
        for name in git(repository, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
        if name
    ]


def local_source_hash(source: Path) -> str:
    """Identify working-tree contents, including new files and nested dependencies."""
    if not (source / ".git").exists():
        return ""
    changed = set(untracked_sources(source))
    for repository in repositories(source):
        changed.update(
            repository / name
            for name in git(
                repository, "diff", "HEAD", "--name-only", "--ignore-submodules=all", "-z"
            ).split("\0")
            if name
        )
    digest = hashlib.sha256()
    for path in sorted(changed):
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        if path.is_symlink():
            digest.update(b"link\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(str(path.stat().st_mode).encode() + b"\0")
            with path.open("rb") as handle:
                digest.update(hashlib.file_digest(handle, "sha256").digest())
        else:
            digest.update(b"deleted")
    return digest.hexdigest() if changed else ""


def copy_sources(source: Path, destination: Path, *, editable: bool = False) -> None:
    """Copy tracked sources and initialized dependencies, excluding Git metadata."""
    if (source / ".git").exists():
        paths = [
            source / name
            for name in git(source, "ls-files", "--recurse-submodules", "-z").split("\0")
            if name
        ]
        if editable:
            paths = sorted(set(paths) | set(untracked_sources(source)))
    else:
        paths = list(source.rglob("*"))
    for path in paths:
        relative = path.relative_to(source)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        target = destination / relative
        if path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(path))
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
