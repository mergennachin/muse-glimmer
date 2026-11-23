from __future__ import annotations

import json
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
from .runtime import MODEL_ID, bind_server, serve

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


def _stop_agent(agent: subprocess.Popen) -> None:
    if agent.poll() is None:
        agent.terminate()
        try:
            agent.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent.kill()
            agent.wait()


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def _register_pi_extension() -> Path:
    # Subagents start fresh Pi processes without the parent's --extension flag.
    directory = Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi/agent")
    extension = directory.expanduser().absolute() / "extensions/muse-glimmer.js"
    source = Path(__file__).with_name("pi.mjs").resolve()
    extension.parent.mkdir(parents=True, exist_ok=True)
    try:
        extension.symlink_to(source)
    except FileExistsError:
        if not extension.is_symlink():
            raise ValueError(f"Cannot register Pi provider: {extension} already exists.") from None
        previous = extension.readlink()
        if previous != source:
            if previous.name != "pi.mjs" or previous.parent.name != "muse_glimmer":
                raise ValueError(f"Cannot replace unrelated Pi extension: {extension}") from None
            # Switch installations atomically; other Pi processes may be discovering extensions.
            with tempfile.TemporaryDirectory(prefix=".muse-glimmer-", dir=extension.parent) as temp:
                replacement = Path(temp) / "provider"
                replacement.symlink_to(source)
                replacement.replace(extension)
    return extension


def launch_agent(model: Path, settings: Settings, name: str) -> int:
    label = {"muse": "Muse", "pi": "Pi", "omp": "Oh My Pi", "hermes": "Hermes Agent", "opencode": "OpenCode"}[name]
    binary = shutil.which(name) or shutil.which(str(Path.home() / ".local/bin" / name))
    if binary is None and name == "opencode":
        binary = shutil.which(str(Path.home() / ".opencode/bin/opencode"))
    if binary is None:
        install = {
            "muse": "Install from https://dev.meta.ai/products/muse-code and make sure muse is on PATH.",
            "pi": "npm install -g @earendil-works/pi-coding-agent",
            "omp": "Install from https://omp.sh and make sure omp is on PATH.",
            "hermes": "Install from https://hermes-agent.nousresearch.com and make sure hermes is on PATH.",
            "opencode": "Install OpenCode 2 from https://opencode.ai and make sure opencode is on PATH.",
        }[name]
        raise ValueError(f"{label} is not installed. {install}")
    pi_extension = _register_pi_extension() if name == "pi" else None

    origin = _origin(settings)
    env = dict(os.environ)
    if name == "muse":
        env.update(
            OPENROUTER_API_KEY="muse-glimmer", OPENROUTER_BASE_URL=f"{origin}/v1"
        )
        command = [
            binary,
            "--provider",
            "openrouter",
            "--model",
            MODEL_ID,
            "--base-url",
            f"{origin}/v1",
        ]
    elif name == "hermes":
        # Launch-only overrides preserve Hermes's saved provider. Its
        # classic CLI supports custom endpoint flags; the TUI does not.
        # Hermes's first-run check also requires credentials in the environment.
        env.update(OPENAI_API_KEY="muse-glimmer", OPENAI_BASE_URL=f"{origin}/v1")
        command = [
            binary,
            "chat",
            "--cli",
            "--provider",
            "custom",
            "--model",
            MODEL_ID,
            "--base-url",
            f"{origin}/v1",
            "--api-key",
            "muse-glimmer",
            "--reasoning",
            "low",
        ]
    elif name == "opencode":
        local_model = f"muse-glimmer/{MODEL_ID}#low"
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            {
                "model": local_model,
                "providers": {
                    "muse-glimmer": {
                        "name": "Muse Glimmer",
                        "package": "@opencode/ai/providers/openai-compatible",
                        "settings": {"baseURL": f"{origin}/v1", "apiKey": "muse-glimmer"},
                        "models": {
                            MODEL_ID: {
                                "name": "Muse Glimmer 30B",
                                "limit": {"context": settings.context, "output": min(32768, settings.context // 4)},
                                "capabilities": {
                                    "input": ["text"], "output": ["text"], "tools": True,
                                    "reasoning": {"supported": True, "effort": ["low", "medium", "high"]},
                                },
                                "compatibility": {"maxTokensField": "max_tokens", "supportsPromptCacheKey": False},
                                "settings": {"reasoningEffort": "low"},
                            }
                        },
                    }
                },
                "agents": {"title": {"model": local_model}},
                "plugins": [Path(__file__).with_name("opencode").resolve().as_uri()],
            }
        )
        # OpenCode's shared background service does not inherit this environment.
        command = [binary, "--standalone"]
    else:
        env[f"MUSE_GLIMMER_{name.upper()}_MODEL"] = json.dumps(
            {
                "baseUrl": f"{origin}/v1",
                "id": MODEL_ID,
                "contextWindow": settings.context,
            }
        )
        if name == "pi":
            command = [
                binary,
                "--extension",
                str(pi_extension),
                "--provider",
                "muse-glimmer",
                "--model",
                MODEL_ID,
            ]
        else:
            # OMP inherits extensions into subagents. Keep its model roles
            # and auxiliary inference on the local provider for this run.
            env["OMP_SKIP_SETUP"] = "1"
            local_model = f"muse-glimmer/{MODEL_ID}"
            command = [
                binary,
                "--extension",
                str(Path(__file__).with_name("omp.mjs").resolve()),
                "--model",
                local_model,
                "--smol",
                local_model,
                "--slow",
                local_model,
                "--plan",
                local_model,
                "--offline",
            ]
    return _launch(model, settings, name, label, origin, command, env)


def _origin(settings: Settings) -> str:
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(settings.host, settings.host)
    host = f"[{host}]" if ":" in host else host
    return f"http://{host}:{settings.port}"


def reusable_server(settings: Settings) -> bool:
    origin = _origin(settings)
    with httpx.Client(base_url=origin, trust_env=False, timeout=2) as client:
        try:
            models = client.get("/v1/models")
        except httpx.ConnectError:
            return False
        except httpx.RequestError as error:
            raise ValueError(f"Could not identify the server at {origin}: {error}") from error
        try:
            data = models.json()["data"] if models.is_success else []
            matches = isinstance(data, list) and any(
                isinstance(entry, dict) and entry.get("id") == MODEL_ID
                and entry.get("owned_by") == "executorch" for entry in data
            )
        except (ValueError, TypeError, KeyError):
            matches = False
        if not matches:
            raise ValueError(f"{origin} is occupied by a different service. Choose another --port.")
        try:
            if client.get("/health").status_code == 200:
                return True
        except httpx.RequestError as error:
            raise ValueError(f"Glimmer at {origin} failed its health check: {error}") from error
        raise ValueError(f"Glimmer at {origin} is not ready.")


def launch_script(model: Path | None, settings: Settings, script: Path, args: list[str]) -> int:
    origin = _origin(settings)
    env = dict(os.environ, MUSE_GLIMMER_BASE_URL=f"{origin}/v1")
    command = [sys.executable, str(script.absolute()), *args]
    if model is None:
        print(f"Reusing Glimmer at {origin}. Launching {script}", file=sys.stderr, flush=True)
        old_term = signal.signal(signal.SIGTERM, _terminate)
        try:
            with subprocess.Popen(command, env=env) as child:
                try:
                    result = child.wait()
                    return result if result >= 0 else 128 - result
                finally:
                    _stop_agent(child)
        finally:
            signal.signal(signal.SIGTERM, old_term)
    return _launch(model, settings, "script", str(script), origin, command, env)


def _launch(
    model: Path, settings: Settings, name: str, label: str, origin: str,
    command: list[str], env: dict[str, str],
) -> int:
    # Bind before starting the child so an existing server can never be mistaken for ours.
    with bind_server(settings.host, settings.port) as listener:
        with tempfile.TemporaryFile(mode="w+b") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "muse_glimmer.launch",
                    str(model),
                    settings.model_dump_json(),
                    str(listener.fileno()),
                    name,
                ],
                pass_fds=(listener.fileno(),),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
            listener.close()
            agent = None
            old_int = signal.getsignal(signal.SIGINT)
            old_term = signal.signal(signal.SIGTERM, _terminate)
            try:
                print(f"Loading {model.name}; starting server at {origin}...", file=sys.stderr, flush=True)
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

                print(f"Server ready. Launching {label} using {origin}/v1", file=sys.stderr, flush=True)
                # The agent owns Ctrl+C; keep its server alive for cancelled turns.
                signal.signal(signal.SIGINT, lambda *_: None)
                agent = subprocess.Popen(command, env=env)
                while True:
                    if server.poll() is not None:
                        raise ValueError(f"Server exited while {label} was running.")
                    try:
                        result = agent.wait(timeout=0.25)
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
                    if agent is not None:
                        _stop_agent(agent)
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
            api_dialect="openrouter" if sys.argv[4] == "muse" else "openai",
            # Agents may start new conversations without releasing old sessions.
            evict_idle_sessions=True,
            queue_requests=sys.argv[4] in {"omp", "hermes", "opencode"},
        )
