import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from muse_glimmer import launch
from muse_glimmer.config import load_settings


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("Timed out waiting for subprocess")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture
def run_launch(tmp_path):
    # Replace only model execution. Exercise the actual CLI, socket handoff,
    # HTTP readiness, environment, terminal signals and subprocess cleanup.
    (tmp_path / "sitecustomize.py").write_text("""
import json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import muse_glimmer.runtime

def serve(model, settings, listener=None, *, api_dialect="openai", evict_idle_sessions=False):
    root = Path(os.environ["LAUNCH_TEST_ROOT"])
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "server.json").write_text(json.dumps({
        "pid": os.getpid(), "worker": worker.pid,
        "model": str(model), "settings": settings.model_dump(), "api_dialect": api_dialect,
        "evict_idle_sessions": evict_idle_sessions,
    }))
    mode = os.environ["LAUNCH_TEST_MODE"]
    if mode == "startup-error":
        print("model could not be loaded", file=sys.stderr, flush=True)
        raise SystemExit(3)
    if mode == "timeout":
        time.sleep(60)
    time.sleep(0.15)
    print("Model HUD output", flush=True)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            if mode == "crash":
                threading.Timer(0.5, lambda: os._exit(3)).start()
        def log_message(self, *args): pass
    server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
    server.socket.close()
    server.socket = listener
    server.serve_forever()

muse_glimmer.runtime.serve = serve

if os.environ.get("LAUNCH_CLIENT") == "open-webui":
    import muse_glimmer.webui, webbrowser
    root = Path(os.environ["LAUNCH_TEST_ROOT"])
    muse_glimmer.webui.WEBUI_PORT = int(os.environ["LAUNCH_WEBUI_PORT"])
    muse_glimmer.webui.STARTUP_TIMEOUT = 2
    muse_glimmer.webui.user_data_path = lambda *a, **kw: root / "data"
    webbrowser.open = lambda url: (root / "browser-url").write_text(url)
""")
    binary = tmp_path / "muse"
    binary.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, signal, sys, time, urllib.request
from pathlib import Path
root = Path(os.environ["LAUNCH_TEST_ROOT"])
def interrupt(*args): (root / "interrupted").touch()
signal.signal(signal.SIGINT, interrupt)
url = os.environ["OPENROUTER_BASE_URL"]
with urllib.request.urlopen(url.removesuffix("/v1") + "/health", timeout=2) as response:
    assert response.status == 200
(root / "muse.json").write_text(json.dumps({
    "pid": os.getpid(), "args": sys.argv[1:], "cwd": os.getcwd(),
    "url": url, "key": os.environ["OPENROUTER_API_KEY"],
}))
if os.environ["LAUNCH_TEST_MODE"] == "exit":
    raise SystemExit(7)
while not (root / "release").exists(): time.sleep(0.02)
"""
    )
    binary.chmod(0o755)
    webui = tmp_path / "open-webui"
    webui.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
root = Path(os.environ["LAUNCH_TEST_ROOT"])
(root / "webui.json").write_text(json.dumps({
    "pid": os.getpid(), "args": sys.argv[1:], "cwd": os.getcwd(),
    "env": {k: v for k, v in os.environ.items() if k in (
        "DATA_DIR", "DATABASE_URL", "ENABLE_PERSISTENT_CONFIG", "WEBUI_AUTH",
        "OPENAI_API_CONFIGS", "OPENAI_API_BASE_URLS", "DEFAULT_MODELS",
        "RAG_EMBEDDING_MODEL", "ENABLE_FOLLOW_UP_GENERATION", "DEFAULT_MODEL_METADATA",
        "PYTHONPATH", "WEBUI_SECRET_KEY",
    )},
}))
print("WebUI diagnostic output", flush=True)
if os.environ["LAUNCH_TEST_MODE"] == "webui-error": raise SystemExit(3)
if os.environ["LAUNCH_TEST_MODE"] == "webui-timeout": time.sleep(60)
Path(".webui_secret_key").touch(exist_ok=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/ready" else 404)
        self.end_headers()
        self.wfile.write(b'{"status":true}')
    def log_message(self, *args): pass
HTTPServer(("127.0.0.1", int(sys.argv[-1])), Handler).serve_forever()
"""
    )
    webui.chmod(0o755)
    model = tmp_path / "model.pte"
    model.touch()
    config = tmp_path / "config.toml"
    config.write_text("context=4096\nconcurrency=2\n")
    muse_config = tmp_path / "muse-config/muse/settings.json"
    muse_config.parent.mkdir(parents=True)
    muse_config.write_text('{"model":"my-usual-model"}')
    source = str(Path(launch.__file__).parents[1])

    @contextmanager
    def run(mode="exit", host="127.0.0.1", port=None, client="muse", webui_port=None):
        if port is None:
            with socket.create_server(("127.0.0.1", 0)) as listener:
                port = listener.getsockname()[1]
        if webui_port is None:
            with socket.create_server(("127.0.0.1", 0)) as listener:
                webui_port = listener.getsockname()[1]
        env = dict(
            os.environ,
            PATH=f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
            PYTHONPATH=os.pathsep.join([str(tmp_path), source]),
            PYTHONDONTWRITEBYTECODE="1",
            LAUNCH_TEST_ROOT=str(tmp_path),
            LAUNCH_TEST_MODE=mode,
            LAUNCH_CLIENT=client,
            LAUNCH_WEBUI_PORT=str(webui_port),
            WEBUI_SECRET_KEY="unrelated-key",
            XDG_CONFIG_HOME=str(muse_config.parent.parent),
            OPENROUTER_API_KEY="existing-cloud-key",
            OPENROUTER_BASE_URL="https://example.invalid",
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from muse_glimmer import launch; launch.STARTUP_TIMEOUT=2; "
                "from muse_glimmer.cli import main; raise SystemExit(main())",
                "serve",
                "--launch",
                client,
                "--model",
                str(model),
                "--config",
                str(config),
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=tmp_path,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            yield proc, f"http://127.0.0.1:{port}/v1"
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            assert muse_config.read_text() == '{"model":"my-usual-model"}'
            assert env["OPENROUTER_API_KEY"] == "existing-cloud-key"
            state = tmp_path / "server.json"
            if state.exists():
                server = json.loads(state.read_text())
                try:
                    wait_for(lambda: not alive(server["pid"]) and not alive(server["worker"]))
                finally:
                    # Do not leave a test worker running if cleanup regresses.
                    if alive(server["pid"]) or alive(server["worker"]):
                        os.killpg(server["pid"], signal.SIGKILL)
            state = tmp_path / "muse.json"
            if state.exists():
                assert not alive(json.loads(state.read_text())["pid"])
            state = tmp_path / "webui.json"
            if state.exists():
                assert not alive(json.loads(state.read_text())["pid"])

    return run


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_launch_waits_for_server_and_preserves_muse_exit_code(run_launch, tmp_path, host):
    with run_launch(host=host) as (proc, url):
        _, err = proc.communicate(timeout=10)
        assert proc.returncode == 7, err
        muse = json.loads((tmp_path / "muse.json").read_text())
        assert muse["url"] == url
        assert muse["key"] == "muse-glimmer"
        assert muse["args"] == [
            "--provider",
            "openrouter",
            "--model",
            "muse-glimmer-30B",
            "--base-url",
            url,
        ]
        assert Path(muse["cwd"]).samefile(tmp_path)
        server = json.loads((tmp_path / "server.json").read_text())
        assert server["model"] == str(tmp_path / "model.pte")
        assert server["settings"]["context"] == 4096
        assert server["settings"]["concurrency"] == 2
        assert server["api_dialect"] == "openrouter"


@pytest.mark.parametrize(
    "mode,message",
    [
        ("startup-error", "model could not be loaded"),
        ("timeout", "Timed out waiting for the server"),
        ("crash", "Server exited while Muse was running"),
    ],
)
def test_server_failure_cleans_up_processes(run_launch, tmp_path, mode, message):
    with run_launch(mode=mode) as (proc, _):
        _, err = proc.communicate(timeout=10)
        assert proc.returncode == 1
        assert message in err
        assert (tmp_path / "muse.json").exists() == (mode == "crash")


def test_occupied_port_does_not_launch_muse_or_touch_existing_server(run_launch, tmp_path):
    with socket.create_server(("127.0.0.1", 0)) as listener:
        with run_launch(port=listener.getsockname()[1]) as (proc, _):
            _, err = proc.communicate(timeout=5)
            assert proc.returncode == 1
            assert "Address already in use" in err
            assert not (tmp_path / "muse.json").exists()
            assert not (tmp_path / "server.json").exists()
            assert listener.getsockname()[1] > 0


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_terminal_signals_and_cleanup(run_launch, tmp_path, sig):
    with run_launch(mode="wait") as (proc, _):
        wait_for(lambda: (tmp_path / "muse.json").exists())
        os.killpg(proc.pid, sig)
        if sig == signal.SIGINT:
            wait_for(lambda: (tmp_path / "interrupted").exists())
            assert proc.poll() is None
            server = json.loads((tmp_path / "server.json").read_text())
            assert alive(server["pid"]) and alive(server["worker"])
            (tmp_path / "release").touch()
        _, err = proc.communicate(timeout=10)
        assert proc.returncode == (0 if sig == signal.SIGINT else 143), err


def test_missing_muse_fails_before_server_start(monkeypatch):
    monkeypatch.setattr(launch.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="Muse is not installed"):
        launch.launch_muse(Path("unused.pte"), load_settings({}))


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_webui_launch_and_shutdown_preserves_hud(run_launch, tmp_path, sig):
    with run_launch(mode="wait", client="open-webui") as (proc, url):
        wait_for(lambda: (tmp_path / "browser-url").exists())
        webui = json.loads((tmp_path / "webui.json").read_text())
        env = webui["env"]
        assert env["OPENAI_API_BASE_URLS"] == url
        assert env["WEBUI_AUTH"] == env["ENABLE_PERSISTENT_CONFIG"] == "false"
        assert env["RAG_EMBEDDING_MODEL"] == ""
        assert env["ENABLE_FOLLOW_UP_GENERATION"] == "false"
        assert env["DEFAULT_MODELS"] == "muse-glimmer-30B"
        assert "PYTHONPATH" not in env and "WEBUI_SECRET_KEY" not in env
        assert json.loads(env["OPENAI_API_CONFIGS"])["0"]["headers"] == {
            "x-session-affinity": "{{CHAT_ID}}"
        }
        assert json.loads(env["DEFAULT_MODEL_METADATA"])["capabilities"]["builtin_tools"] is False
        data = tmp_path / "data/open-webui"
        assert Path(webui["cwd"]) == data
        assert Path(env["DATA_DIR"]) == data
        assert (data / ".webui_secret_key").exists()
        assert webui["args"][:4] == ["serve", "--host", "127.0.0.1", "--port"]
        assert (tmp_path / "browser-url").read_text() == f"http://127.0.0.1:{webui['args'][-1]}"
        server = json.loads((tmp_path / "server.json").read_text())
        assert server["evict_idle_sessions"] is True
        assert server["api_dialect"] == "openai"
        os.killpg(proc.pid, sig)
        out, err = proc.communicate(timeout=15)
        assert proc.returncode == (130 if sig == signal.SIGINT else 143), err
        assert "Model HUD output" in out
        assert "WebUI diagnostic output" not in out + err
        assert "WebUI diagnostic output" in (data / "webui.log").read_text()


@pytest.mark.parametrize(
    "mode, message",
    [
        ("startup-error", "Model server exited before it was ready"),
        ("webui-error", "Open WebUI exited before it was ready"),
        ("webui-timeout", "Timed out waiting for Open WebUI"),
        ("crash", "Model server exited"),
    ],
)
def test_webui_failures_clean_up_owned_children(run_launch, tmp_path, mode, message):
    with run_launch(mode=mode, client="open-webui") as (proc, _):
        _, err = proc.communicate(timeout=15)
        assert proc.returncode == 1
        assert message in err
        assert "webui.log" in err
        assert not (tmp_path / "browser-url").exists() or mode == "crash"


def test_webui_occupied_port_does_not_start_model(run_launch, tmp_path):
    with socket.create_server(("127.0.0.1", 0)) as listener:
        with run_launch(client="open-webui", webui_port=listener.getsockname()[1]) as (proc, _):
            _, err = proc.communicate(timeout=5)
            assert proc.returncode == 1
            assert "Address already in use" in err
            assert not (tmp_path / "server.json").exists()
            assert listener.getsockname()[1] > 0


def test_missing_webui_has_install_instructions(monkeypatch):
    from muse_glimmer import webui

    monkeypatch.setattr(webui.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="uv tool install --python 3.12 open-webui==0.13.3"):
        webui.launch_open_webui(Path("unused.pte"), load_settings({}))
