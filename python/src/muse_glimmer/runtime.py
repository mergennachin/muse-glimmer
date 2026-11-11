from __future__ import annotations

from pathlib import Path
from socket import socket

from .config import Settings
from .resources import resource_path

MODEL_ID = "muse-glimmer-30B"


def native_dir() -> Path:
    return Path(__file__).with_name("_native")


def resolve_model(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        models = sorted(file for file in path.iterdir() if file.is_file() and file.suffix == ".pte")
        if not models:
            raise ValueError(f"No .pte file found directly in {path}.")
        if len(models) > 1:
            raise ValueError(
                f"Multiple .pte files in {path}. Pass the explicit file with --model:\n"
                + "\n".join(f"  {file}" for file in models)
            )
        return models[0]
    if not path.is_file() or path.suffix != ".pte":
        raise ValueError(
            f"Expected an existing .pte file or export directory: {path}. Pull again or use --model."
        )
    return path


def server_options(model: Path, settings: Settings) -> list[str]:
    tokenizer = resource_path("tokenizer")
    options = [
        "--worker-bin",
        str(native_dir() / "muse_glimmer_batching_worker"),
        "--model-path",
        str(model),
        "--tokenizer-path",
        str(tokenizer / "tokenizer.json"),
        "--hf-tokenizer",
        str(tokenizer),
        "--backend",
        "mlx",
        "--model-id",
        MODEL_ID,
        "--max-context",
        str(settings.context),
        "--max-sessions",
        str(settings.concurrency),
        "--max-decode-sequences",
        str(settings.concurrency),
        "--max-inflight-requests",
        str(settings.concurrency),
        "--prefix-cache-entries",
        str(settings.prefix_cache_entries),
        "--tool-parser",
        "atem",
        "--host",
        settings.host,
        "--port",
        str(settings.port),
    ]
    pos_embed = model.with_name("pos_embed.bin")
    if pos_embed.is_file():
        options.extend(["--pos-embed-path", str(pos_embed)])
    if settings.dflash:
        options.extend(["--dflash", "--dflash-n-draft", str(settings.dflash_n_draft)])
    return options


def serve(
    model: Path,
    settings: Settings,
    listener: socket | None = None,
    *,
    api_dialect: str = "openai",
    evict_idle_sessions: bool = False,
) -> None:
    if not (native_dir() / "muse_glimmer_batching_worker").is_file():
        raise ValueError("Native runtime missing. Reinstall muse-glimmer to build it.")
    import uvicorn

    from ._vendor.muse import serve_batching as serving

    options = [*server_options(model, settings), "--api-dialect", api_dialect]
    if evict_idle_sessions:
        options.append("--evict-idle-sessions")
    args = serving._parse_args(options)
    app = serving.build_app_from_args(args)

    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
    uvicorn.Server(config).run(sockets=[listener] if listener is not None else None)
