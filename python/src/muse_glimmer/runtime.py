from __future__ import annotations

import asyncio
import errno
import platform
import socket
from contextlib import ExitStack
from pathlib import Path

from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings
from .resources import resource_path

MODEL_ID = "muse-glimmer-30B"
MINIMUM_MACOS = (26, 2)


def check_platform() -> None:
    # Wheel tags cannot express macOS minor versions after macOS 11.
    if platform.system() == "Darwin":
        version = platform.mac_ver()[0]
        if tuple(map(int, version.split(".")[:2])) < MINIMUM_MACOS:
            minimum = ".".join(map(str, MINIMUM_MACOS))
            raise ValueError(f"Serving requires macOS {minimum} or newer; found {version}.")


def bind_server(host: str, port: int, *, fixed_port: bool = False) -> socket.socket:
    """Reserve a listening socket before starting the model or client."""
    try:
        return socket.create_server((host, port), family=socket.AF_INET6 if ":" in host else socket.AF_INET)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        hint = "Stop the service using this port and try again."
        if not fixed_port:
            alternate = port + 1 if port < 65535 else 8000
            hint += f" Or choose another port with --port {alternate}."
        raise ValueError(f"Address already in use: {host}:{port}. {hint}") from None


class QueuedRequests:
    """Queue agent foreground and auxiliary requests across all their workers."""

    def __init__(self, app: ASGIApp, concurrency: int):
        self.app = app
        self.slots = asyncio.Semaphore(concurrency)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/v1/chat/completions"
        ):
            await self.app(scope, receive, send)
            return

        # Keep the slot through the response body and upstream worker cleanup.
        async with self.slots:
            await self.app(scope, receive, send)


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
            f"Expected an existing .pte file or export directory: {path}.\n"
            "Run muse-glimmer pull --list to choose a model to download, "
            "or use --model /path/to/model.pte for a local file."
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
        "cuda" if platform.system() == "Linux" else "mlx",
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
    if settings.prefix_cache_entries > 0 and settings.prefix_checkpoint_prompt_end == "every":
        options.extend(["--prefix-checkpoint-prompt-end", "every"])
    if platform.system() == "Linux":
        data = model.with_name("aoti_cuda_blob.ptd")
        if not data.is_file():
            raise ValueError(f"CUDA export data missing: {data}. Keep the .pte and .ptd together.")
        options.extend(["--data-path", str(data)])
    pos_embed = model.with_name("pos_embed.bin")
    if pos_embed.is_file():
        options.extend(["--pos-embed-path", str(pos_embed)])
    if settings.dflash:
        options.extend(["--dflash", "--dflash-n-draft", str(settings.dflash_n_draft)])
    return options


def serve(
    model: Path,
    settings: Settings,
    listener: socket.socket | None = None,
    *,
    api_dialect: str = "openai",
    evict_idle_sessions: bool = False,
    queue_requests: bool = False,
) -> None:
    check_platform()
    with ExitStack() as stack:
        listeners = [listener] if listener is not None else []
        if listener is None:
            addresses = socket.getaddrinfo(
                settings.host or None, settings.port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE,
            )
            # Match asyncio's hostname resolution, including both localhost loopbacks.
            for family, _, _, _, address in dict.fromkeys(addresses):
                host, _ = socket.getnameinfo(address, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)
                try:
                    listeners.append(stack.enter_context(bind_server(host, settings.port)))
                except OSError as error:
                    if error.errno not in {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}:
                        raise
            if not listeners:
                raise OSError(f"No supported listening addresses for {settings.host!r}")
        if not (native_dir() / "muse_glimmer_batching_worker").is_file():
            raise ValueError("Native runtime missing. Reinstall muse-glimmer.")
        import uvicorn

        from ._vendor.muse import serve_batching as serving

        options = [*server_options(model, settings), "--api-dialect", api_dialect]
        if evict_idle_sessions:
            options.append("--evict-idle-sessions")
        args = serving._parse_args(options)
        app = serving.build_app_from_args(args)
        if queue_requests:
            app = QueuedRequests(app, settings.concurrency)

        config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
        uvicorn.Server(config).run(sockets=listeners)
