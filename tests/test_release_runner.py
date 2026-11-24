"""Runner shutdown and registration cleanup, using a real local child process."""

import base64
import hashlib
import importlib.util
import io
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner_module():
    path = Path(__file__).resolve().parents[1] / "packaging/release_runner.py"
    spec = importlib.util.spec_from_file_location("release_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", [
    "success", "failure", "poll_error", "transient", "interrupt", "runner_exit",
])
@pytest.mark.parametrize("host,arch,asset,label", [
    ("Darwin", "arm64", "osx-arm64", None),
    ("Linux", "x86_64", "linux-x64", "cuda-13_2"),
])
def test_runner_cleanup(tmp_path, monkeypatch, runner_module, outcome, host, arch, asset, label):
    module = runner_module
    monkeypatch.setattr(module.platform, "system", lambda: host)
    monkeypatch.setattr(module.platform, "machine", lambda: arch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))

    archive = tmp_path / "runner.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, contents in {
            "config.sh": "#!/bin/sh\nprintf '\ufeff{\"agentId\":42}' > .runner\n",
            "run.sh": "#!/bin/sh\nexec sleep 60\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.mode = 0o755
            info.size = len(contents.encode())
            bundle.addfile(info, io.BytesIO(contents.encode()))

    requests = []
    polls = 0

    def api(endpoint, method="GET"):
        nonlocal polls
        requests.append((endpoint, method))
        if endpoint.endswith("/runs/123"):
            polls += 1
            if polls > 1 and outcome == "poll_error":
                raise RuntimeError("Cannot read run status")
            if polls == 2 and outcome == "interrupt":
                raise KeyboardInterrupt
            if polls == 2 and outcome in {"transient", "runner_exit"}:
                if outcome == "runner_exit":
                    children[-1].terminate()
                    children[-1].wait(timeout=5)
                raise module.TransientAPIError("GitHub is unavailable")
            return {
                "path": ".github/workflows/wheel.yml",
                "event": "push",
                "head_sha": "a" * 40,
                "html_url": "https://github.com/owner/repo/actions/runs/123",
                "status": "queued" if polls == 1 else "completed",
                "conclusion": "success" if outcome == "transient" else outcome,
            }
        if "/contents/" in endpoint:
            return {"content": base64.b64encode(b"muse-glimmer-${{ github.run_id }}").decode()}
        if endpoint == "repos/actions/runner/releases/latest":
            return {
                "tag_name": "v1",
                "assets": [
                    {
                        "name": f"actions-runner-{asset}-1.tar.gz",
                        "digest": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ],
            }
        if endpoint.endswith("registration-token"):
            return {"token": "private-registration-token"}
        if endpoint.endswith("/cancel") or method == "DELETE":
            return None
        raise AssertionError(endpoint)

    def command(*args):
        if args[0] == "xcrun":
            assert host == "Darwin"
            return "Metal compiler"
        if args[:3] == ("gh", "repo", "view"):
            return "owner/repo"
        assert args[:3] == ("gh", "release", "download")
        shutil.copy2(archive, Path(args[-1]) / f"actions-runner-{asset}-1.tar.gz")
        return ""

    children = []
    popen = subprocess.Popen
    monkeypatch.setenv("MUSE_GLIMMER_CXX_RUNTIME_DIR", str(tmp_path / "gcc-runtime"))

    def start(*args, **kwargs):
        if args[0] == ["./run.sh"] and host == "Linux":
            assert kwargs["env"]["MUSE_GLIMMER_CXX_RUNTIME_DIR"] == str(tmp_path / "gcc-runtime")
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module, "api", api)
    monkeypatch.setattr(module, "command", command)
    monkeypatch.setattr(module.subprocess, "Popen", start)
    waits = []

    def wait(delay):
        waits.append(delay)
        if outcome == "transient":
            assert children[-1].poll() is None
            assert not any(endpoint.endswith("/cancel") for endpoint, _ in requests)

    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=wait))
    if outcome == "poll_error":
        with pytest.raises(RuntimeError, match="Cannot read run status"):
            module.run(123, [label] if label else [])
    elif outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            module.run(123, [label] if label else [])
    elif outcome == "runner_exit":
        with pytest.raises(RuntimeError, match="Runner exited early"):
            module.run(123, [label] if label else [])
    else:
        assert module.run(123, [label] if label else []) == (1 if outcome == "failure" else 0)
    if outcome == "transient":
        assert waits == [30] and polls == 3
    assert children and all(child.poll() is not None for child in children)
    assert ("repos/owner/repo/actions/runners/42", "DELETE") in requests
    assert (("repos/owner/repo/actions/runs/123/cancel", "POST") in requests) == (
        outcome in {"interrupt", "runner_exit"}
    )
    assert not any(p.name.startswith("muse-glimmer-runner-") for p in tmp_path.iterdir())


@pytest.mark.parametrize("host,workflow,event,allowed", [
    ("Linux", "wheel-linux-cuda.yml", "workflow_dispatch", True),
    ("Linux", "wheel-linux-cuda.yml", "push", False),
    ("Darwin", "wheel-linux-cuda.yml", "workflow_dispatch", False),
    ("Linux", "wheel.yml", "push", True),
    ("Linux", "unrelated.yml", "workflow_dispatch", False),
])
def test_runner_only_joins_expected_platform_workflows(runner_module, monkeypatch, host, workflow, event, allowed):
    module = runner_module
    monkeypatch.setattr(module.platform, "system", lambda: host)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64" if host == "Linux" else "arm64")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(module, "command", Mock(return_value="owner/repo"))
    read = Mock(return_value={
        "path": f".github/workflows/{workflow}", "event": event,
        "status": "completed", "conclusion": "success",
    })
    monkeypatch.setattr(module, "api", read)
    if allowed:
        assert module.run(123) == 0
    else:
        with pytest.raises(RuntimeError, match="Expected a shared wheel release"):
            module.run(123)
    # No registration or other mutation happens during validation.
    read.assert_called_once_with("repos/owner/repo/actions/runs/123")


@pytest.mark.parametrize("error", [
    RuntimeError("gh: Service Unavailable (HTTP 503)"),
    RuntimeError("gh: Too Many Requests (HTTP 429)"),
    RuntimeError("error connecting to api.github.com"),
    subprocess.TimeoutExpired(["gh", "api"], 120),
])
def test_github_read_recovers(runner_module, monkeypatch, error):
    call = Mock(side_effect=[error, '{"status": "queued"}'])
    sleep = Mock()
    monkeypatch.setattr(runner_module, "command", call)
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(sleep=sleep))
    assert runner_module.api("repos/owner/repo/actions/runs/123") == {"status": "queued"}
    assert call.call_count == 2
    sleep.assert_called_once_with(2)


@pytest.mark.parametrize("method,status", [
    ("GET", 401), ("GET", 403), ("GET", 404), ("POST", 503), ("DELETE", 503),
])
def test_github_does_not_retry_permanent_errors_or_mutations(runner_module, monkeypatch, method, status):
    call = Mock(side_effect=RuntimeError(f"gh: request failed (HTTP {status})"))
    sleep = Mock()
    monkeypatch.setattr(runner_module, "command", call)
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        runner_module.api("repos/owner/repo/actions/runs/123", method)
    assert call.call_count == 1
    sleep.assert_not_called()


def test_github_read_retries_are_bounded(runner_module, monkeypatch):
    call = Mock(side_effect=RuntimeError("gh: Bad Gateway (HTTP 502)"))
    sleep = Mock()
    monkeypatch.setattr(runner_module, "command", call)
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(runner_module.TransientAPIError, match="after 5 attempts"):
        runner_module.api("repos/owner/repo/actions/runs/123")
    assert call.call_count == 5
    assert [args[0][0] for args in sleep.call_args_list] == [2, 4, 8, 16]
