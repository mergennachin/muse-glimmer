"""Build and install real wheels in isolation; native compilation is not exercised."""

import os
import shutil
import subprocess
import sys
import sysconfig
import venv
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
    run(arguments, cwd=project)

    # An existing editable installation must not satisfy imports for the test wheel.
    installed = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(installed)
    python = installed / "bin/python"
    wheel = next((tmp_path / "dist").glob("*.whl"))
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
