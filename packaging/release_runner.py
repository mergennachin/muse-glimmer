"""Join one GitHub release on this machine, then remove the temporary runner.

Requires Python 3.11+, Git, and an authenticated gh CLI. Mac builds need Xcode
with Metal. Linux builds need the toolchain/GPU specified by their platform.
See .github/RELEASING.md for setup, release steps, and retries.

    python packaging/release_runner.py RUN_ID
    python packaging/release_runner.py RUN_ID --label cuda-13_2

Approve the pypi deployment in GitHub when prompted. Ctrl+C cancels the run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TransientAPIError(RuntimeError):
    """A GitHub read failed after retries; the runner may still be healthy."""


def command(*args: str, timeout: int = 120) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def api(path: str, method: str = "GET"):
    for attempt in range(5):
        try:
            output = command("gh", "api", "--method", method, path, timeout=30)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            # gh includes the HTTP status in its error text. Only retry safe
            # reads; replaying an ambiguous POST can repeat a mutation.
            status = re.search(r"\(HTTP (\d{3})\)", str(error))
            transient = (
                (int(status[1]) in {408, 429} or 500 <= int(status[1]) < 600)
                if status else isinstance(error, subprocess.TimeoutExpired) or re.search(
                    r"error connecting|connection (?:reset|refused)|timeout|timed out|"
                    r"temporary failure|no such host|unexpected eof|tls handshake|dial tcp",
                    str(error), re.IGNORECASE,
                )
            )
            if method != "GET" or not transient:
                raise
            if attempt == 4:
                raise TransientAPIError(f"GitHub read failed after 5 attempts: {error}") from error
            delay = 2 ** (attempt + 1)
            print(f"GitHub read failed; retrying in {delay}s: {error}", flush=True)
            time.sleep(delay)
        else:
            return json.loads(output) if output.strip() else None


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(run_id: int, labels: list[str] | None = None) -> int:
    host = (platform.system(), platform.machine().lower())
    archive_platform = {
        ("Darwin", "arm64"): "osx-arm64",
        ("Linux", "x86_64"): "linux-x64",
        ("Linux", "aarch64"): "linux-arm64",
    }.get(host)
    if archive_platform is None:
        raise RuntimeError("The release runner supports Apple Silicon and Linux x86_64/aarch64.")
    labels = labels or []
    if any(not re.fullmatch(r"[a-zA-Z0-9_.-]+", label) for label in labels):
        raise RuntimeError("Runner labels may contain only letters, numbers, dots, underscores and hyphens.")
    required_tools = ["gh", "git", *(["xcrun"] if host[0] == "Darwin" else [])]
    for tool in required_tools:
        if not shutil.which(tool):
            raise RuntimeError(f"Missing {tool}; see this script's setup instructions.")
    if host[0] == "Darwin":
        command("xcrun", "-sdk", "macosx", "metal", "--version")
    repository = command(
        "gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"
    ).strip()
    run_path = f"repos/{repository}/actions/runs/{run_id}"
    workflow = api(run_path)
    allowed_workflows = {".github/workflows/wheel.yml"}
    if host[0] == "Linux" and workflow["event"] == "workflow_dispatch":
        allowed_workflows.add(".github/workflows/wheel-linux-cuda.yml")
    if workflow["path"] not in allowed_workflows or workflow["event"] not in {
        "push",
        "workflow_dispatch",
    }:
        raise RuntimeError("Expected a shared wheel release or a manual Linux CUDA build.")
    if workflow["status"] == "completed":
        print(f"Run already completed: {workflow['conclusion']}")
        return 0 if workflow["conclusion"] == "success" else 1
    source_path = (
        ".github/workflows/linux_wheel.yml"
        if workflow["path"] == ".github/workflows/wheel-linux-cuda.yml"
        else workflow["path"]
    )
    source = api(f"repos/{repository}/contents/{source_path}?ref={workflow['head_sha']}")
    if "muse-glimmer-${{ github.run_id }}" not in base64.b64decode(source["content"]).decode():
        raise RuntimeError(
            "Start a run from the updated workflow; this run does not support the local runner helper."
        )

    print(workflow["html_url"], flush=True)
    name = f"muse-glimmer-{run_id}-{uuid.uuid4().hex[:8]}"
    runner_id = None
    process = None
    cancel_run = False
    with tempfile.TemporaryDirectory(prefix="muse-glimmer-runner-") as temporary:
        work = Path(temporary)
        runner = work / "runner"
        runner.mkdir()
        try:
            release = api("repos/actions/runner/releases/latest")
            filename = f"actions-runner-{archive_platform}-{release['tag_name'].removeprefix('v')}.tar.gz"
            asset = next(a for a in release["assets"] if a["name"] == filename)
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest") or ""):
                raise RuntimeError("GitHub did not provide a SHA-256 digest for the runner.")
            print(f"Downloading runner {release['tag_name']}...", flush=True)
            command(
                "gh",
                "release",
                "download",
                release["tag_name"],
                "--repo",
                "actions/runner",
                "--pattern",
                filename,
                "--dir",
                str(work),
            )
            archive = work / filename
            with archive.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != asset["digest"].removeprefix("sha256:"):
                raise RuntimeError("Runner checksum mismatch.")
            with tarfile.open(archive) as bundle:
                bundle.extractall(runner, filter="data")

            # Keep job tooling and native build caches inside the temporary directory.
            env = {
                key: os.environ[key]
                for key in (
                    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "DEVELOPER_DIR",
                    "CUDA_HOME", "CUDA_PATH", "CUDACXX",
                )
                if key in os.environ
            }
            runtime_dir = os.environ.get("MUSE_GLIMMER_CXX_RUNTIME_DIR")
            if host[0] == "Linux" and runtime_dir:
                env["MUSE_GLIMMER_CXX_RUNTIME_DIR"] = str(Path(runtime_dir).expanduser().resolve())
            env.update(
                PATH=os.pathsep.join(
                    dict.fromkeys(
                        [str(Path(shutil.which(tool)).parent) for tool in required_tools]
                        + [str(Path(shutil.which(tool)).parent) for tool in ("nvcc", "nvidia-smi") if shutil.which(tool)]
                        + ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
                    )
                ),
                UV_CACHE_DIR=str(work / "uv-cache"),
                UV_PYTHON_INSTALL_DIR=str(work / "python"),
                MUSE_GLIMMER_BUILD_CACHE=str(work / "build"),
            )
            token = api(f"repos/{repository}/actions/runners/registration-token", "POST")["token"]
            # Do not include the registration token in an exception's command line.
            try:
                configured = subprocess.run(
                    [
                        "./config.sh",
                        "--unattended",
                        "--url",
                        f"https://github.com/{repository}",
                        "--token",
                        token,
                        "--name",
                        name,
                        "--labels",
                        ",".join([f"muse-glimmer-{run_id}", *labels]),
                        "--work",
                        "_work",
                        "--disableupdate",
                    ],
                    cwd=runner,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("Runner configuration timed out.") from None
            if configured.returncode:
                raise RuntimeError(
                    (configured.stdout + configured.stderr).replace(token, "[redacted]")
                )
            runner_id = json.loads((runner / ".runner").read_text(encoding="utf-8-sig"))["agentId"]
            with (work / "runner.log").open("w") as log:
                process = subprocess.Popen(
                    ["./run.sh"],
                    cwd=runner,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                previous = None
                while True:
                    if process.poll() is not None:
                        cancel_run = True
                        raise RuntimeError(
                            "Runner exited early:\n" + (work / "runner.log").read_text()[-4000:]
                        )
                    try:
                        workflow = api(run_path)
                    except TransientAPIError as error:
                        # A status API outage is not evidence that the job failed.
                        # Keep serving jobs and try another bounded read after a pause.
                        print(
                            f"{error}\nKeeping the runner online; checking again in 30s. "
                            "Ctrl+C cancels the shared release.", flush=True,
                        )
                        time.sleep(30)
                        continue
                    status = workflow["status"]
                    if status != previous:
                        print(
                            "Approve the pypi deployment in GitHub."
                            if status == "waiting"
                            else f"Run: {status}",
                            flush=True,
                        )
                        previous = status
                    if status == "completed":
                        print(f"Result: {workflow['conclusion']}", flush=True)
                        return 0 if workflow["conclusion"] == "success" else 1
                    time.sleep(15)
        except KeyboardInterrupt:
            cancel_run = True
            raise
        finally:
            try:
                if process is not None and cancel_run:
                    api(run_path + "/cancel", "POST")
            finally:
                try:
                    if process is not None:
                        stop(process)
                finally:
                    # Registration can succeed even if config.sh fails afterward.
                    if runner_id is None:
                        runners = api(f"repos/{repository}/actions/runners?per_page=100")["runners"]
                        runner_id = next((r["id"] for r in runners if r["name"] == name), None)
                    if runner_id is not None:
                        api(f"repos/{repository}/actions/runners/{runner_id}", "DELETE")
                        print("Temporary runner removed.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_id", type=int, help="GitHub Actions run ID from the release page")
    parser.add_argument("--label", action="append", default=[], help="Additional runner capability label (repeatable)")
    args = parser.parse_args()

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        return run(args.run_id, args.label)
    except KeyboardInterrupt:
        print("Cancelled.")
        return 130
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"release runner: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
