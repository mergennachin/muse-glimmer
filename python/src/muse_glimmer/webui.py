from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx
from platformdirs import user_data_path

from .config import Settings
from .launch import _stop_server, _terminate
from .runtime import MODEL_ID, serve

WEBUI_PORT = 8080
STARTUP_TIMEOUT = 300


def _environment(data: Path, origin: str, webui_origin: str) -> dict[str, str]:
    env = os.environ.copy()
    # Keep the separate installation and its persistent key independent of the caller.
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "WEBUI_SECRET_KEY",
        "WEBUI_JWT_SECRET_KEY",
        "FRONTEND_BUILD_DIR",
    ):
        env.pop(key, None)
    env.update(
        DATA_DIR=str(data),
        DATABASE_URL=f"sqlite:///{data / 'webui.db'}",
        REDIS_URL="",
        WEBUI_URL=webui_origin,
        WEBUI_AUTH="false",
        ENABLE_PERSISTENT_CONFIG="false",
        ENABLE_OLLAMA_API="false",
        ENABLE_OPENAI_API="true",
        OPENAI_API_BASE_URL=f"{origin}/v1",
        OPENAI_API_BASE_URLS=f"{origin}/v1",
        OPENAI_API_KEY="",
        OPENAI_API_KEYS="",
        OPENAI_API_CONFIGS=json.dumps(
            {"0": {"auth_type": "none", "headers": {"x-session-affinity": "{{CHAT_ID}}"}}}
        ),
        DEFAULT_MODELS=MODEL_ID,
        DEFAULT_MODEL_METADATA=json.dumps(
            {
                "capabilities": {
                    "builtin_tools": False,
                    "vision": False,
                    "file_upload": False,
                    "usage": True,
                }
            }
        ),
        ENABLE_TITLE_GENERATION="false",
        ENABLE_TAGS_GENERATION="false",
        ENABLE_FOLLOW_UP_GENERATION="false",
        ENABLE_AUTOCOMPLETE_GENERATION="false",
        OFFLINE_MODE="true",
        RAG_EMBEDDING_MODEL="",
        ANONYMIZED_TELEMETRY="false",
        SCARF_NO_ANALYTICS="true",
        DO_NOT_TRACK="true",
    )
    return env


def _wait_ready(process, url: str, name: str, server=None) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    with httpx.Client(trust_env=False, timeout=0.5) as client:
        while True:
            if server is not None and server.poll() is not None:
                raise ValueError("Model server exited while Open WebUI was starting.")
            if process.poll() is not None:
                raise ValueError(f"{name} exited before it was ready.")
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise ValueError(f"Timed out waiting for {name} to start.")
            time.sleep(0.1)


def launch_open_webui(model: Path, settings: Settings) -> int:
    binary = shutil.which("open-webui") or shutil.which(str(Path.home() / ".local/bin/open-webui"))
    if binary is None:
        raise ValueError(
            "Open WebUI is not installed. Run: uv tool install --python 3.12 open-webui==0.13.3"
        )
    if settings.port == WEBUI_PORT:
        raise ValueError(f"Open WebUI uses port {WEBUI_PORT}; choose another model server --port.")

    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(settings.host, settings.host)
    origin = f"http://{'[' + host + ']' if ':' in host else host}:{settings.port}"
    webui_origin = f"http://127.0.0.1:{WEBUI_PORT}"
    data = user_data_path("muse-glimmer", appauthor=False) / "open-webui"
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = data / "webui.log"
    family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET

    with (data / ".launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Open WebUI is already running through muse-glimmer.") from None
        # Reserve both ports while the model loads. The WebUI CLI binds its own socket.
        with socket.create_server(("127.0.0.1", WEBUI_PORT)) as webui_listener:
            with socket.create_server((settings.host, settings.port), family=family) as listener:
                with log_path.open("w") as log:
                    print(
                        f"Open WebUI: {webui_origin}\nChats: {data}\nWebUI log: {log_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    server = webui = None
                    old_int = signal.getsignal(signal.SIGINT)
                    old_term = signal.signal(signal.SIGTERM, _terminate)
                    try:
                        # Inherit the terminal so the model server owns the live HUD.
                        server = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "muse_glimmer.webui",
                                str(model.absolute()),
                                settings.model_dump_json(),
                                str(listener.fileno()),
                            ],
                            pass_fds=(listener.fileno(),),
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                        )
                        listener.close()
                        _wait_ready(server, f"{origin}/health", "Model server")
                        webui_listener.close()
                        webui = subprocess.Popen(
                            [binary, "serve", "--host", "127.0.0.1", "--port", str(WEBUI_PORT)],
                            cwd=data,
                            env=_environment(data, origin, webui_origin),
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                        )
                        _wait_ready(webui, f"{webui_origin}/ready", "Open WebUI", server)
                        try:
                            webbrowser.open(webui_origin)
                        except (webbrowser.Error, OSError):
                            pass  # The printed URL remains usable without a desktop browser.
                        while True:
                            if server.poll() is not None:
                                raise ValueError(
                                    "Model server exited while Open WebUI was running."
                                )
                            if webui.poll() is not None:
                                raise ValueError("Open WebUI exited.")
                            time.sleep(0.1)
                    except (ValueError, OSError) as error:
                        raise ValueError(f"{error}\nOpen WebUI log: {log_path}") from error
                    finally:
                        signal.signal(signal.SIGINT, lambda *_: None)
                        signal.signal(signal.SIGTERM, signal.SIG_IGN)
                        try:
                            if webui is not None:
                                _stop_server(webui)
                            if server is not None:
                                _stop_server(server)
                        finally:
                            signal.signal(signal.SIGINT, old_int)
                            signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    with socket.socket(fileno=int(sys.argv[3])) as listener:
        serve(
            Path(sys.argv[1]),
            Settings.model_validate_json(sys.argv[2]),
            listener,
            evict_idle_sessions=True,
        )
