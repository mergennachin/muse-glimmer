from __future__ import annotations

import platform
import tempfile
from pathlib import Path, PurePosixPath

import httpx
import yaml
from huggingface_hub import HfApi, hf_hub_download
from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, TypeAdapter

from .resources import resource_path

# Hub 2 uses httpx2; Hub 1 uses httpx.
try:
    from httpx2 import HTTPError as HubHTTPError
except ImportError:
    HubHTTPError = httpx.HTTPError


class ReleaseSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str
    directory: str


class ModelRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_variant: str
    variants: dict[str, dict[str, ReleaseSource]]


def _release_source(version: str, variant: str | None = None) -> ReleaseSource:
    try:
        contents = yaml.safe_load(resource_path("models.yaml").read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid model catalog: {error}") from error
    releases = TypeAdapter(dict[str, ModelRelease]).validate_python(contents)
    if version not in releases:
        raise ValueError(
            f"Unknown release {version!r}. Available releases: {', '.join(sorted(releases))}."
        )
    release = releases[version]
    if variant is None:
        variant = release.default_variant
    if variant not in release.variants:
        raise ValueError(
            f"Unknown variant {variant!r} for release {version!r}. "
            f"Available variants: {', '.join(sorted(release.variants))}."
        )
    sources = release.variants[variant]
    target = f"{platform.system().lower()}-{platform.machine().lower()}"
    if target not in sources:
        raise ValueError(
            f"Release {version!r} variant {variant!r} is not supported on this platform ({target}). "
            f"Supported platforms: {', '.join(sorted(sources))}."
        )
    return sources[target]


def default_model_file() -> Path:
    return user_data_path("muse-glimmer", appauthor=False) / "default-model"


def default_model() -> Path | None:
    try:
        value = default_model_file().read_text().strip()
    except FileNotFoundError:
        return None
    return Path(value) if value else None


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


def _select_model(files: dict[str, int | None], directory: str) -> str:
    directory = str(PurePosixPath(directory))
    models = sorted(
        name
        for name in files
        if name.endswith(".pte") and str(PurePosixPath(name).parent) == directory
    )
    if not models:
        raise ValueError(f"No .pte files found in export directory {directory!r}.")
    if len(models) != 1:
        raise ValueError(
            f"Export directory {directory!r} contains multiple .pte files. "
            "Each variant must contain exactly one .pte file. "
            "For a local copy, use serve --model with the explicit .pte path."
        )
    return models[0]


def pull(release: str, variant: str | None = None) -> Path:
    source = _release_source(release, variant)
    repo_id = source.repo_id
    try:
        info = HfApi().model_info(repo_id, revision=source.revision, files_metadata=True)
        files = {file.rfilename: file.size for file in info.siblings}
        filename = _select_model(files, source.directory)
        required = _export_files(filename, files)

        print(f"Downloading Muse Glimmer {release} ({_download_size(required, files)})", flush=True)
        model = Path(hf_hub_download(repo_id, filename, revision=info.sha))
        for companion in required[1:]:
            hf_hub_download(repo_id, companion, revision=info.sha)
    except (httpx.HTTPError, HubHTTPError) as error:
        raise ValueError(f"Could not download Muse Glimmer {release}: {error}") from error

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
