"""Run the live smoke test against the installed wheel, using an isolated server."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path, model: Path | None = None) -> None:
    import muse_glimmer
    from huggingface_hub.constants import HF_HUB_CACHE

    if not Path(muse_glimmer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Install the built wheel first; do not use an editable installation.")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    with tempfile.TemporaryDirectory(prefix="muse-glimmer-smoke-") as temporary:
        home = Path(temporary)
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        # Preserve cached weights, but never change the runner owner's selected model.
        env.update(HOME=str(home), HF_HUB_CACHE=HF_HUB_CACHE, NO_COLOR="1")
        command = [sys.executable, "-I", "-m", "muse_glimmer.cli"]
        if model is None:
            subprocess.run([*command, "pull", "1.0"], cwd=home, env=env, check=True, timeout=1200)
        else:
            model = model.expanduser().resolve()
        with socket.socket() as available:
            available.bind(("127.0.0.1", 0))
            port = available.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        log_path = output / "server.log"
        with log_path.open("w") as log:
            server = subprocess.Popen(
                [*command, "serve", *(["--model", str(model)] if model is not None else []), "--host", "127.0.0.1", "--port", str(port),
                 "--context", "4096", "--concurrency", "4", "--prefix-cache-entries", "8",
                 "--prefix-checkpoint-prompt-end", "every"],
                cwd=home, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                opener = build_opener(ProxyHandler({}))
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited during startup ({server.returncode})")
                    try:
                        with opener.open(url + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Server did not become ready within 180 seconds")
                    time.sleep(0.5)
                subprocess.run(
                    [sys.executable, "-I", str(ROOT / "scripts/smoke_test.py"),
                     "--base-url", url, "--requests", "8", "--concurrency", "4",
                     "--output", str(output / "smoke.json")],
                    cwd=home, env=env, check=True, timeout=600,
                )
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during the smoke test ({server.returncode})")
            except BaseException:
                print(log_path.read_text(errors="replace")[-16000:], file=sys.stderr)
                raise
            finally:
                # Include the native worker, including after a Python server crash.
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        os.killpg(server.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    server.wait()
        print(f"Live wheel smoke passed. Report and server log: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/live"))
    parser.add_argument("--model", type=Path, help="Use a local export instead of pulling a model")
    args = parser.parse_args()

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    run(args.output, args.model)
