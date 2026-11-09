"""Exercise real Git submodule initialization and clean source distributions."""

import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def commit(path):
    git(path, "add", ".")
    git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "muse_source", Path(__file__).resolve().parents[1] / "packaging/source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "package"
    root.mkdir()
    git(root, "init", "-q")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    (origin / "CMakePresets.json").write_text("{}\n")
    (origin / ".gitignore").write_text("*.pte\nbuild/\n")
    pin = commit(origin)
    (origin / "later.txt").write_text("not part of the pinned revision\n")
    commit(origin)
    git(root, "submodule", "add", str(origin), module.SUBMODULE)
    git(root / module.SUBMODULE, "checkout", "-q", "--detach", pin)
    git(root, "add", module.SUBMODULE)
    git(root, "submodule", "deinit", "-f", "--", module.SUBMODULE)
    return module, origin, pin


def test_initializes_pinned_submodule_and_reuses_without_fetch(acquisition):
    module, origin, pin = acquisition
    assert module.source_metadata() == {"repository": str(origin), "revision": pin}
    source = module.source_checkout()
    assert source == module.ROOT / module.SUBMODULE
    assert (source / ".git").is_file()
    assert git(source, "rev-parse", "HEAD") == pin
    assert not (source / "later.txt").exists()
    origin.rename(origin.with_name("unavailable"))
    assert module.source_checkout() == source
    assert not git(source, "status", "--porcelain")


def test_missing_gitlink_is_rejected(acquisition):
    module, _, _ = acquisition
    git(module.ROOT, "update-index", "--force-remove", module.SUBMODULE)
    with pytest.raises(RuntimeError, match="gitlink"):
        module.source_checkout()


def test_unresolved_gitlink_is_rejected(acquisition):
    module, _, pin = acquisition
    git(module.ROOT, "update-index", "--force-remove", module.SUBMODULE)
    subprocess.run(
        ["git", "-C", str(module.ROOT), "update-index", "--index-info"],
        input=f"160000 {pin} 1\t{module.SUBMODULE}\n",
        text=True,
        check=True,
    )
    with pytest.raises(RuntimeError, match="resolved submodule gitlink"):
        module.source_checkout()


def test_existing_files_are_preserved(acquisition):
    module, _, _ = acquisition
    source = module.ROOT / module.SUBMODULE
    source.mkdir(parents=True, exist_ok=True)
    (source / "notes.txt").write_text("keep me")
    with pytest.raises(RuntimeError, match="preserved"):
        module.source_checkout()
    assert (source / "notes.txt").read_text() == "keep me"


def test_dirty_checkout_is_preserved(acquisition):
    module, _, _ = acquisition
    source = module.source_checkout()
    (source / "CMakePresets.json").write_text('{"local": true}\n')
    with pytest.raises(RuntimeError, match="must be clean"):
        module.source_checkout()
    assert (source / "CMakePresets.json").read_text() == '{"local": true}\n'


@pytest.mark.parametrize("editable", [False, True])
def test_wrong_revision_is_preserved(acquisition, editable):
    module, _, _ = acquisition
    source = module.source_checkout()
    (source / "local.txt").write_text("local work")
    local_revision = commit(source)
    with pytest.raises(RuntimeError, match="differs from the submodule pin"):
        module.source_checkout(editable=editable)
    assert git(source, "rev-parse", "HEAD") == local_revision


def test_editable_snapshot_includes_local_changes_and_excludes_builds(acquisition, tmp_path):
    module, _, pin = acquisition
    source = module.source_checkout(editable=True)
    assert module.local_source_hash(source) == ""
    (source / "CMakePresets.json").write_text('{"local": true}\n')
    hud = source / "examples/hud.py"
    hud.parent.mkdir()
    hud.write_text("LOCAL_HUD = True\n")
    fingerprint = module.local_source_hash(source)
    (source / "model.pte").write_bytes(b"ignored weights")
    (source / "build").mkdir()
    (source / "build/worker").write_bytes(b"ignored executable")
    assert module.source_checkout(editable=True) == source
    assert module.local_source_hash(source) == fingerprint
    snapshot = tmp_path / "snapshot"
    module.copy_sources(source, snapshot, editable=True)
    assert (snapshot / "CMakePresets.json").read_text() == '{"local": true}\n'
    assert (snapshot / "examples/hud.py").read_text() == hud.read_text()
    assert not (snapshot / "model.pte").exists()
    assert not (snapshot / "build").exists()
    assert not (snapshot / ".git").exists()
    git(source, "add", "examples/hud.py")
    assert module.local_source_hash(source) == fingerprint
    hud.write_text("LOCAL_HUD = False\n")
    assert module.local_source_hash(source) != fingerprint
    fingerprint = module.local_source_hash(source)
    hud.unlink()
    assert module.local_source_hash(source) != fingerprint
    assert git(source, "rev-parse", "HEAD") == pin
    with pytest.raises(RuntimeError, match="must be clean"):
        module.source_checkout()


def test_sdist_roundtrip_without_git(acquisition, tmp_path):
    module, origin, pin = acquisition
    root = Path(__file__).resolve().parents[1]
    for name in [
        "setup.py",
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "default_config.toml",
        "models.yaml",
    ]:
        shutil.copy2(root / name, module.ROOT / name)
    for name in ["packaging", "build_backend", "python", "tokenizer"]:
        shutil.copytree(
            root / name,
            module.ROOT / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "_native", "_vendor"),
        )
    (module.source_checkout() / "model.pte").write_bytes(b"ignored weights")
    expected = {"repository": str(origin), "revision": pin}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    project = module.ROOT
    for iteration in range(2):
        output = tmp_path / f"dist-{iteration}"
        result = subprocess.run(
            [sys.executable, "setup.py", "sdist", "--dist-dir", str(output)],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        extracted = tmp_path / f"extracted-{iteration}"
        with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
            archive.extractall(extracted, filter="data")
        project = next(extracted.iterdir())
        assert not (project / ".git").exists()
        assert not (project / module.SUBMODULE / ".git").exists()
        assert not (project / module.SUBMODULE / "model.pte").exists()
        assert (project / "models.yaml").read_bytes() == (root / "models.yaml").read_bytes()
        assert json.loads((project / "third-party/executorch-source.json").read_text()) == expected
        acquisition_from_archive = runpy.run_path(str(project / "packaging/source.py"))
        assert acquisition_from_archive["source_checkout"]() == project / module.SUBMODULE
    (project / "third-party/executorch-source.json").unlink()
    with pytest.raises(RuntimeError, match="pin is missing"):
        acquisition_from_archive["source_checkout"]()


def test_only_required_dependencies_are_initialized(acquisition, monkeypatch, tmp_path):
    module, origin, _ = acquisition
    monkeypatch.setattr(module, "MLX_SUBMODULES", ["deps/needed"])
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    git(dependency, "init", "-q")
    (dependency / "file.txt").write_text("dependency")
    commit(dependency)
    for name in ["needed", "unused"]:
        git(origin, "submodule", "add", str(dependency), f"deps/{name}")
    pin = commit(origin)
    source = module.ROOT / module.SUBMODULE
    git(module.ROOT, "submodule", "update", "--init", "--", module.SUBMODULE)
    git(source, "fetch", "origin")
    git(source, "checkout", "-q", "--detach", pin)
    git(module.ROOT, "add", module.SUBMODULE)
    assert module.source_checkout() == source
    assert module.source_metadata()["revision"] == pin
    assert (source / "deps/needed/file.txt").read_text() == "dependency"
    assert not (source / "deps/unused/file.txt").exists()
    (source / "deps/needed/file.txt").write_text("local changes")
    with pytest.raises(RuntimeError, match="clean"):
        module.source_checkout()
    assert module.source_checkout(editable=True) == source
    fingerprint = module.local_source_hash(source)
    (source / "deps/needed/new.cpp").write_text("// new source\n")
    assert module.local_source_hash(source) != fingerprint
    snapshot = tmp_path / "snapshot"
    module.copy_sources(source, snapshot, editable=True)
    assert (snapshot / "deps/needed/file.txt").read_text() == "local changes"
    assert (snapshot / "deps/needed/new.cpp").read_text() == "// new source\n"
