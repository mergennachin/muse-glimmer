from __future__ import annotations

import platform
import re
import tempfile
from pathlib import Path, PurePosixPath

import httpx
from huggingface_hub import HfApi, hf_hub_download
from platformdirs import user_data_path


def default_model_file() -> Path:
    return user_data_path("muse-glimmer", appauthor=False) / "default-model"


def default_model() -> Path | None:
    try:
        value = default_model_file().read_text().strip()
    except FileNotFoundError:
        return None
    return Path(value) if value else None


def _supports_variant(directory: str) -> bool:
    name = PurePosixPath(directory).name
    if name.endswith("-metal"):
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    # CUDA exports are excluded until the runtime supports them.
    return re.search(r"-sm\d+(?:\+ptx)?$", name) is None


def _export_files(model: str, files: dict[str, int | None]) -> list[str]:
    positions = str(PurePosixPath(model).with_name("pos_embed.bin"))
    return [model, positions] if positions in files else [model]


def _download_size(names: list[str], files: dict[str, int | None]) -> str:
    sizes = [files[name] for name in names]
    if any(size is None for size in sizes):
        return "size unknown"
    total = sum(sizes)
    for unit, scale in [("GB", 10**9), ("MB", 10**6), ("KB", 10**3)]:
        if total >= scale:
            return f"{total / scale:.1f} {unit}"
    return f"{total} bytes"


def _select_model(files: dict[str, int | None], variant: str | None) -> str:
    exports: dict[str, list[str]] = {}
    for name in sorted(files):
        if name.endswith(".pte"):
            exports.setdefault(str(PurePosixPath(name).parent), []).append(name)
    if not exports:
        raise ValueError("No .pte files found in this repository.")

    if variant is not None:
        variant = str(PurePosixPath(variant))
        if variant not in exports:
            raise ValueError(f"No export directory named {variant!r} in this repository.")
        if not _supports_variant(variant):
            raise ValueError(f"Variant {variant!r} is not supported on this platform.")
    else:
        candidates = [name for name in exports if _supports_variant(name)]
        if not candidates:
            raise ValueError(
                "No exports available for this platform. Only Apple silicon is supported."
            )
        if len(candidates) > 1:
            choices = []
            for name in candidates:
                models = exports[name]
                detail = (
                    _download_size(_export_files(models[0], files), files)
                    if len(models) == 1
                    else "multiple .pte files"
                )
                choices.append(f"  {name} ({detail})")
            raise ValueError(
                "Multiple exports found. Select a directory with --variant:\n" + "\n".join(choices)
            )
        variant = candidates[0]

    models = exports[variant]
    if len(models) != 1:
        raise ValueError(
            f"Export directory {variant!r} contains multiple .pte files. "
            "Each variant must contain exactly one .pte file. "
            "For a local copy, use serve --model with the explicit .pte path."
        )
    return models[0]


def pull(repo_id: str, variant: str | None = None) -> Path:
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        files = {file.rfilename: file.size for file in info.siblings}
        filename = _select_model(files, variant)
        required = _export_files(filename, files)

        print(f"Downloading {repo_id}/{filename} ({_download_size(required, files)})", flush=True)
        model = Path(hf_hub_download(repo_id, filename, revision=info.sha))
        for companion in required[1:]:
            hf_hub_download(repo_id, companion, revision=info.sha)
    except httpx.HTTPError as error:
        raise ValueError(f"Could not download from {repo_id}: {error}") from error

    # Keep the snapshot path: resolving its symlink would lose neighboring files.
    destination = default_model_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(str(model.absolute()) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return model
