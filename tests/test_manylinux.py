"""Check the native dependencies prepared for the offline CUDA builder."""

import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_stb_uses_the_recipe_commit(tmp_path, monkeypatch):
    # Use a local remote whose HEAD has moved beyond ExecuTorch's pin.
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "deprecated").mkdir()
    for name in ("stb_image.h", "deprecated/stb_image_resize.h", "LICENSE"):
        (repo / name).write_text(name)
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "add", "."], check=True)
    commit = [*git, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm"]
    subprocess.run([*commit, "pinned"], check=True)
    pin = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    (repo / "stb_image.h").write_text("later version")
    subprocess.run([*git, "add", "stb_image.h"], check=True)
    subprocess.run([*commit, "later"], check=True)
    recipe = tmp_path / "CMakeLists.txt"
    recipe.write_text(f"FetchContent_Declare(stb GIT_REPOSITORY {repo.as_uri()} GIT_TAG {pin})")
    monkeypatch.delenv("MUSE_GLIMMER_STB_INCLUDE_DIR", raising=False)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    dest = tmp_path / "input"
    runpy.run_path(str(ROOT / "packaging/build_manylinux.py"))["prepare_stb"](recipe, dest)
    assert (dest / "stb_image.h").read_text() == "stb_image.h"
    assert (dest / "stb_image_resize.h").read_text() == "deprecated/stb_image_resize.h"
    assert (dest / "LICENSE").read_text() == "LICENSE"


@pytest.mark.parametrize("layout", ["rpm", "debian", "vendored"])
def test_bundled_openmp_notices_include_license_and_exception(tmp_path, layout):
    if layout == "rpm":
        directory = tmp_path / "licenses/libgcc"
        directory.mkdir(parents=True)
        (directory / "COPYING3").write_text("installed GPLv3 license")
        (directory / "COPYING.RUNTIME").write_text("installed GCC runtime exception")
    elif layout == "debian":
        directory = tmp_path / "doc/libgomp1"
        directory.mkdir(parents=True)
        (directory / "copyright").write_text("packager's OpenMP notice")

    notices = "".join(runpy.run_path(str(ROOT / "packaging/stage.py"))["gcc_runtime_notices"](True, tmp_path))
    assert "libgomp" in notices
    if layout == "rpm":
        assert "installed GPLv3 license" in notices and "installed GCC runtime exception" in notices
    else:
        assert "GNU GENERAL PUBLIC LICENSE" in notices and "GCC RUNTIME LIBRARY EXCEPTION" in notices
    if layout == "debian":
        assert "packager's OpenMP notice" in notices
