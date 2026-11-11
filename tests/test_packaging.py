"""Build and install real wheels in isolation; native compilation is not exercised."""

import os
import runpy
import shutil
import subprocess
import sys
import sysconfig
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["wheel", "editable"])
def test_installed_package_uses_its_own_shared_resources(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in [
        "setup.py",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
        "default_config.toml",
        "models.yaml",
    ]:
        shutil.copy2(root / name, project / name)
    shutil.copytree(
        root / "python",
        project / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "_native", "_vendor"),
    )
    shutil.copytree(root / "tokenizer", project / "tokenizer")
    shared = project / "default_config.toml"
    shared.write_text(shared.read_text().replace("port = 8000", "port = 8082"))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def run(arguments, cwd=tmp_path):
        result = subprocess.run(arguments, cwd=cwd, env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    command = "bdist_wheel" if mode == "wheel" else "editable_wheel"
    arguments = [sys.executable, "setup.py", command, "--dist-dir", str(tmp_path / "dist")]
    if mode == "wheel":
        removed = project / "python/src/muse_glimmer/removed_module.py"
        removed.write_text("OLD = True\n")
        run(arguments, cwd=project)
        removed.unlink()
    run(arguments, cwd=project)

    # An existing editable installation must not satisfy imports for the test wheel.
    installed = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(installed)
    python = installed / "bin/python"
    wheel = next((tmp_path / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.endswith("/removed_module.py") for name in archive.namelist())
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        description = metadata.get_payload()
        assert "pip install --pre muse-glimmer" in description
        assert "pip install -e ." not in description
        source_url = f"https://github.com/mergennachin/muse-glimmer/tree/v{metadata['Version']}"
        assert f"Source, {source_url}" in metadata.get_all("Project-URL")
        assert f"/blob/v{metadata['Version']}/default_config.toml" in description
    assert "pip install -e ." in (project / "README.md").read_text()
    run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)])
    if mode == "wheel":
        shutil.rmtree(project)
        expected_root, expected_port = installed, "8082"
    else:
        shared.write_text(shared.read_text().replace("port = 8082", "port = 9091"))
        expected_root, expected_port = project, "9091"

    run(
        [
            str(python),
            "-I",
            "-c",
            """
import sys
from pathlib import Path
# Reuse dependencies without processing the parent environment's editable .pth files.
sys.path.extend(sys.argv[4:])
from muse_glimmer.config import load_settings
from muse_glimmer.resources import resource_path
assert resource_path('default_config.toml').resolve().is_relative_to(Path(sys.argv[1]))
assert load_settings({}).port == int(sys.argv[2])
assert resource_path('models.yaml').resolve().is_relative_to(Path(sys.argv[1]))
assert resource_path('models.yaml').read_bytes() == (Path(sys.argv[3]) / 'models.yaml').read_bytes()
for original in (Path(sys.argv[3]) / 'tokenizer').iterdir():
    assert (resource_path('tokenizer') / original.name).read_bytes() == original.read_bytes()
""",
            str(expected_root),
            expected_port,
            str(root),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        ]
    )


@pytest.mark.parametrize("failure", ["import", "license"])
def test_failed_staging_preserves_installed_runtime(tmp_path, monkeypatch, failure):
    root = Path(__file__).resolve().parents[1]
    stage = runpy.run_path(str(root / "packaging/stage.py"))["stage"]
    package = tmp_path / "package"
    for name in ("_vendor", "_native"):
        (package / name).mkdir(parents=True)
        (package / name / "previous").write_bytes(b"working install")
    monkeypatch.setitem(stage.__globals__, "PACKAGE", package)
    source = tmp_path / "source"
    sources = source / "examples/llm_server/python"
    sources.mkdir(parents=True)
    (sources / "server.py").write_text(
        "import executorch.unbundled_dependency\n" if failure == "import" else "READY = True\n"
    )
    (source / "LICENSE").write_text("fixture license\n")
    worker = tmp_path / "build/examples/models/worker"
    worker.mkdir(parents=True)
    for name in ("muse_glimmer_batching_worker", "mlx.metallib"):
        (worker / name).write_bytes(b"new artifact")
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda _: {
            "source_metadata": lambda: {"revision": "a" * 40},
            "validate_checkout": lambda *args, **kwargs: None,
            "MLX_SUBMODULES": [],
        },
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda args, **kwargs: "arm64\n" if args[0] == "lipo" else "worker:\n",
    )
    message = (
        "Unpackaged upstream dependency"
        if failure == "import"
        else "Missing native dependency license"
    )
    with pytest.raises(ValueError, match=message):
        stage(source, worker)
    for name in ("_vendor", "_native"):
        assert (package / name / "previous").read_bytes() == b"working install"
        assert len(list((package / name).iterdir())) == 1
    assert sorted(path.name for path in package.iterdir()) == ["_native", "_vendor"]


def test_missing_metal_toolchain_fails_before_native_build(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    check_metal = runpy.run_path(str(root / "packaging/build_native.py"))["check_metal"]
    calls = []

    def missing_toolchain(argv, **kwargs):
        calls.append(argv)
        # Xcode 26+ resolves `xcrun --find metal` but cannot run it without the component.
        return subprocess.CompletedProcess(
            argv,
            72,
            "",
            "error: error: cannot execute tool 'metal' due to missing Metal Toolchain; "
            "use: xcodebuild -downloadComponent MetalToolchain\n",
        )

    monkeypatch.setattr(subprocess, "run", missing_toolchain)
    with pytest.raises(RuntimeError, match="sudo xcodebuild -license accept") as error:
        check_metal()
    assert "missing Metal Toolchain" in str(error.value)
    assert calls == [["xcrun", "-sdk", "macosx", "metal", "--version"]]
