from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from .config import Settings
from .runtime import MODEL_ID, serve

STARTUP_TIMEOUT = 300


def _stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # Also reap a worker left behind if the server crashed or hung during shutdown.
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait()


def _stop_muse(muse: subprocess.Popen) -> None:
    if muse.poll() is None:
        muse.terminate()
        try:
            muse.wait(timeout=5)
        except subprocess.TimeoutExpired:
            muse.kill()
            muse.wait()


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def launch_muse(model: Path, settings: Settings) -> int:
    binary = shutil.which("muse") or shutil.which(str(Path.home() / ".local/bin/muse"))
    if binary is None:
        raise ValueError("Muse is not installed. Install Muse and make sure muse is on PATH.")

    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(settings.host, settings.host)
    host = f"[{host}]" if ":" in host else host
    origin = f"http://{host}:{settings.port}"
    family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET
    # Bind before starting the child so an existing server can never be mistaken for ours.
    with socket.create_server((settings.host, settings.port), family=family) as listener:
        with tempfile.TemporaryFile(mode="w+b") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "muse_glimmer.launch",
                    str(model),
                    settings.model_dump_json(),
                    str(listener.fileno()),
                ],
                pass_fds=(listener.fileno(),),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
            listener.close()
            muse = None
            old_int = signal.getsignal(signal.SIGINT)
            old_term = signal.signal(signal.SIGTERM, _terminate)
            try:
                print(f"Starting server at {origin}...", file=sys.stderr, flush=True)
                deadline = time.monotonic() + STARTUP_TIMEOUT
                with httpx.Client(trust_env=False, timeout=0.5) as client:
                    while True:
                        if server.poll() is not None:
                            raise ValueError("Server exited before it was ready.")
                        try:
                            if client.get(f"{origin}/health").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise ValueError("Timed out waiting for the server to start.")
                        time.sleep(0.1)

                env = dict(
                    os.environ,
                    OPENROUTER_API_KEY="muse-glimmer",
                    OPENROUTER_BASE_URL=f"{origin}/v1",
                )
                print(f"Launching Muse using {origin}/v1", file=sys.stderr, flush=True)
                # Muse owns Ctrl+C while interactive; keep its server alive for cancelled turns.
                signal.signal(signal.SIGINT, lambda *_: None)
                muse = subprocess.Popen(
                    [
                        binary,
                        "--provider",
                        "openrouter",
                        "--model",
                        MODEL_ID,
                        "--base-url",
                        f"{origin}/v1",
                    ],
                    env=env,
                )
                while True:
                    if server.poll() is not None:
                        raise ValueError("Server exited while Muse was running.")
                    try:
                        result = muse.wait(timeout=0.25)
                        return result if result >= 0 else 128 - result
                    except subprocess.TimeoutExpired:
                        pass
            except ValueError:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                print(log.read().decode(errors="replace"), file=sys.stderr, end="")
                raise
            finally:
                signal.signal(signal.SIGINT, lambda *_: None)
                try:
                    if muse is not None:
                        _stop_muse(muse)
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
            api_dialect="openrouter",
        )
