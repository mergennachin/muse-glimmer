from __future__ import annotations

import os
import platform
import sys
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

import httpx
import yaml
from huggingface_hub import CacheNotFound, HfApi, hf_hub_download, scan_cache_dir, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

from .resources import resource_path
from .terminal import print_path

# Hub 2 uses httpx2; Hub 1 uses httpx.
try:
    from httpx2 import HTTPError as HubHTTPError
except ImportError:
    HubHTTPError = httpx.HTTPError


class ModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str
    directory: str
    dflash: bool = False
    files: list[str] | None = None

    @field_validator("directory")
    @classmethod
    def relative_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("export directory must stay inside the repository")
        return value

    @field_validator("files")
    @classmethod
    def export_filenames(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (
            len(value) != len(set(value))
            or sum(name.endswith(".pte") for name in value) != 1
            or any(name != PurePosixPath(name).name or name in {"", ".", ".."} for name in value)
        ):
            raise ValueError("export files must be unique filenames including exactly one .pte")
        return value


class PulledModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    dflash: bool = False


class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_variant: str
    variants: dict[str, dict[str, ModelSource]]


def _catalog() -> dict[str, ModelVersion]:
    try:
        contents = yaml.safe_load(resource_path("models.yaml").read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid model catalog: {error}") from error
    return TypeAdapter(dict[str, ModelVersion]).validate_python(contents)


def _model_source(version: str, variant: str | None = None) -> ModelSource:
    versions = _catalog()
    if version not in versions:
        raise ValueError(
            f"Unknown model version {version!r}. Available model versions: {', '.join(sorted(versions))}."
        )
    model = versions[version]
    if variant is None:
        variant = model.default_variant
    if variant not in model.variants:
        raise ValueError(
            f"Unknown variant {variant!r} for model version {version!r}. "
            f"Available variants: {', '.join(sorted(model.variants))}."
        )
    sources = model.variants[variant]
    target = f"{platform.system().lower()}-{platform.machine().lower()}"
    if target not in sources:
        raise ValueError(
            f"Model version {version!r} variant {variant!r} is not supported on this platform ({target}). "
            f"Supported platforms: {', '.join(sorted(sources))}."
        )
    return sources[target]


def default_model_file() -> Path:
    return user_data_path("muse-glimmer", appauthor=False) / "default-model"


def default_model() -> PulledModel | None:
    try:
        value = default_model_file().read_text().strip()
    except FileNotFoundError:
        return None
    if not value:
        return None
    if value.startswith("{"):
        try:
            return PulledModel.model_validate_json(value)
        except ValueError as error:
            raise ValueError(
                "Saved model selection is invalid. Run muse-glimmer pull --remove to reset it."
            ) from error
    # Existing installations stored only the model path.
    return PulledModel(path=Path(value))


def _local_export(source: ModelSource) -> tuple[Path, list[Path]] | None:
    """Inspect an existing snapshot without contacting the Hub or downloading files."""
    try:
        snapshot = Path(snapshot_download(
            source.repo_id, revision=source.revision, local_files_only=True,
        ))
    except LocalEntryNotFoundError:
        return None
    directory = snapshot / source.directory
    if source.files is None:
        # Older/custom catalogs discover filenames online. Still expose their cached files.
        if not directory.is_dir():
            return None
        names = {str(p.relative_to(snapshot)): None for p in directory.iterdir()}
        try:
            filename = _select_model(names, source.directory)
        except ValueError:
            return None
        required = [snapshot / name for name in _export_files(filename, names)]
    else:
        names = {str(PurePosixPath(source.directory) / name): None for name in source.files}
        filename = _select_model(names, source.directory)
        required = [snapshot / name for name in names]
    return snapshot / filename, required


def _entry_path(path: Path) -> Path:
    # Normalize directory aliases without dereferencing the artifact's blob symlink.
    return path.parent.resolve() / path.name


def list_models(version: str | None = None) -> None:
    versions = _catalog()
    if version is not None:
        if version not in versions:
            raise ValueError(
                f"Unknown model version {version!r}. Available model versions: {', '.join(sorted(versions))}."
            )
        versions = {version: versions[version]}
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"

    def style(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    try:
        selected = default_model()
        selection_error = None
    except ValueError as error:
        selected, selection_error = None, str(error)
    print(style("SELECTED MODEL", "1;36"))
    print("==============")
    if selection_error:
        print(style(selection_error, "33"))
    elif selected is None:
        print("Artifact: none")
    else:
        path = selected.path.absolute()
        missing = " " + style("[missing]", "33") if not path.is_file() else ""
        print(f"Artifact: {style(path.name, '1')}{missing}")
        print_path(path)
    print("Selection applies to the next serve; a running server keeps its model.")
    print()
    target = f"{platform.system().lower()}-{platform.machine().lower()}"
    print(f"Available models for {target}:")
    available = False
    for version, model in sorted(versions.items()):
        variants = sorted(name for name, sources in model.variants.items() if target in sources)
        if variants:
            available = True
            print(f"  {version}")
            for name in variants:
                source = model.variants[name][target]
                local = _local_export(source)
                labels = ["recommended"] if name == model.default_variant else []
                is_selected = False
                if local is not None:
                    path, required = local
                    is_selected = selected is not None and _entry_path(path) == _entry_path(selected.path)
                    if any(p.is_file() or p.is_symlink() for p in required):
                        labels.append("downloaded" if all(p.is_file() for p in required) else "incomplete")
                        if is_selected:
                            labels.append("selected")
                print(f"    {name}" + "".join(f" [{label}]" for label in labels))
                if local is not None and not is_selected and any(p.is_file() or p.is_symlink() for p in local[1]):
                    print_path(local[0], prefix="      Location: ")
    if not available:
        print("  No models available for this platform.")


def remove_model(version: str | None = None, variant: str | None = None) -> Path | None:
    try:
        selected = default_model()
    except ValueError:
        if version is None:
            default_model_file().unlink()
            print("Cleared invalid selection. No model files were removed; use a model version to target a download.")
            return None
        selected = None
    if version is not None:
        source = _model_source(version, variant)
        local = _local_export(source)
        if local is None:
            print(f"Model {version} ({source.directory}) is not downloaded. Nothing to remove.")
            return None
        model, paths = local
    elif selected is not None:
        model = selected.path
        paths = [model, model.with_name("pos_embed.bin"), model.with_name("aoti_cuda_blob.ptd")]
    else:
        print("No model selected. Nothing to remove.")
        return None
    model = _entry_path(model)
    _delete_export(model, {_entry_path(p) for p in paths})
    if selected and _entry_path(selected.path) == model and default_model() == selected:
        default_model_file().unlink(missing_ok=True)
    return model


def _delete_export(model: Path, paths: set[Path]) -> None:
    snapshots = next(
        (p for p in model.parents if p.name == "snapshots" and p.parent.name.startswith("models--")),
        None,
    )
    if snapshots is None or model.suffix != ".pte":
        raise ValueError("Model is not in a Hugging Face model cache; no files were removed.")
    if not snapshots.parent.exists():
        return
    # Keep the first symlink hop: a repo blob may itself link to a shared payload.
    links = {p: os.readlink(p) for p in paths if p.is_symlink()}
    blobs_dir = (snapshots.parent / "blobs").resolve()
    for path, target in links.items():
        if _entry_path(path.parent / target).parent != blobs_dir:
            raise ValueError("Model cache points outside its blob directory; no files were removed.")
    # Missing payloads have no bytes to reclaim, but stale links must not block cache inspection.
    for path in links:
        if not path.exists():
            path.unlink()
    try:
        cache = scan_cache_dir(snapshots.parent.parent)
    except CacheNotFound as error:
        raise ValueError("Model cache disappeared during removal; try again.") from error
    repo = next((r for r in cache.repos if r.repo_path == snapshots.parent), None)
    if repo is None:
        raise ValueError("Cannot inspect this model's cache. Other damaged entries may need repair.")
    files = [f for revision in repo.revisions for f in revision.files if f.file_path in paths]
    for file in files:
        if file.blob_path != file.file_path and _entry_path(file.blob_path).parent != blobs_dir:
            raise ValueError("Model cache points outside its blob directory; no files were removed.")
    strategy = cache.delete_files(*files)
    retained = {
        _entry_path(f.blob_path) for revision in repo.revisions for f in revision.files
        if f.file_path not in paths
    }
    # HF compares paths lexically; normalize aliases before deciding a blob is unreferenced.
    strategy = replace(strategy, blobs=frozenset(
        p for p in strategy.blobs if _entry_path(p) not in retained
    ))
    try:
        strategy.execute()
        if any(p.exists() or p.is_symlink() for p in strategy.files | strategy.blobs):
            raise OSError("Could not remove all model cache files. Fix permissions and retry removal.")
    except BaseException:
        # Preserve references to payloads that survived a partial failure so retry can find them.
        for path, target in links.items():
            if not path.is_symlink() and not path.exists() and (path.parent / target).exists():
                path.symlink_to(target)
        raise
    if any(_entry_path(f.blob_path) in retained for f in files):
        print("Shared data still used by other downloads was kept.")


def _export_files(model: str, files: dict[str, int | None]) -> list[str]:
    companions = [str(PurePosixPath(model).with_name(name)) for name in ("pos_embed.bin", "aoti_cuda_blob.ptd")]
    return [model, *[name for name in companions if name in files]]


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


def pull(version: str, variant: str | None = None, force: bool = False) -> Path:
    source = _model_source(version, variant)
    if not force and source.files is not None:
        local = _local_export(source)
        if local is not None and all(p.is_file() for p in local[1]):
            model = local[0]
            _save_default(model, source.dflash)
            print(f"Using downloaded model: {model.name}", flush=True)
            return model
    repo_id = source.repo_id
    try:
        info = HfApi().model_info(repo_id, revision=source.revision, files_metadata=True)
        files = {file.rfilename: file.size for file in info.siblings}
        filename = _select_model(files, source.directory)
        required = _export_files(filename, files)
        if source.files is not None:
            declared = [str(PurePosixPath(source.directory) / name) for name in source.files]
            if not set(required).issubset(declared) or any(name not in files for name in declared):
                raise ValueError(f"Model catalog files do not match export {source.directory!r}.")
            required = [filename, *[name for name in declared if name != filename]]

        print(f"Downloading {source.directory} ({_download_size(required, files)})", flush=True)
        model = Path(hf_hub_download(repo_id, filename, revision=info.sha, force_download=force))
        for companion in required[1:]:
            hf_hub_download(repo_id, companion, revision=info.sha, force_download=force)
    except (httpx.HTTPError, HubHTTPError) as error:
        raise ValueError(f"Could not download Muse Glimmer {version}: {error}") from error
    except ValueError as error:
        # Hub wraps forced-download metadata failures in a generic ValueError.
        if not force or error.__cause__ is None:
            raise
        raise ValueError(f"Could not download Muse Glimmer {version}: {error.__cause__}") from error

    _save_default(model, source.dflash)
    return model


def _save_default(model: Path, dflash: bool) -> None:
    # Keep the snapshot path: resolving its symlink would lose neighboring files.
    destination = default_model_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(
                PulledModel(path=model.absolute(), dflash=dflash).model_dump_json() + "\n"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
