"""Check Linux serving options and a relocated native payload without a GPU."""

import os
import platform
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
from muse_glimmer import runtime
from muse_glimmer.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_compiler_rejects_mixed_toolkits_and_accepts_aliases(tmp_path, monkeypatch):
    select = runpy.run_path(str(ROOT / "packaging/build_native.py"))["cuda_compiler"]
    toolkit = tmp_path / "cuda-13.2"
    nvcc = toolkit / "bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.2, V13.2.86'\n")
    nvcc.chmod(0o755)
    other = tmp_path / "cuda-12.8/bin/nvcc"
    other.parent.mkdir(parents=True)
    other.touch(mode=0o755)
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CUDACXX", str(other))
    with pytest.raises(RuntimeError, match="conflicts"):
        select()
    alias = tmp_path / "current-nvcc"
    alias.symlink_to(nvcc)
    monkeypatch.setenv("CUDACXX", f"{alias} --allow-unsupported-compiler")
    assert select()[:2] == (nvcc, "--allow-unsupported-compiler")
    monkeypatch.delenv("CUDA_HOME")
    assert select()[:2] == (nvcc, "--allow-unsupported-compiler")


def test_linux_solo_loads_the_cuda_data_file(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    model = tmp_path / "solo.pte"
    data = tmp_path / "aoti_cuda_blob.ptd"
    data.write_bytes(b"data")
    settings = load_settings({})
    assert settings.dflash is False
    options = runtime.server_options(model, settings)
    assert options[options.index("--backend") + 1] == "cuda"
    assert options[options.index("--data-path") + 1] == str(data)
    assert "--dflash" not in options
    data.unlink()
    with pytest.raises(ValueError, match="CUDA export data missing"):
        runtime.server_options(model, settings)


def test_macos_keeps_mlx_and_dflash_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    options = runtime.server_options(tmp_path / "model.pte", load_settings({}))
    assert options[options.index("--backend") + 1] == "mlx"
    assert "--data-path" not in options
    assert "--dflash" in options


@pytest.mark.skipif(platform.system() != "Linux", reason="ELF test")
def test_relocated_worker_finds_its_bundled_library(tmp_path, monkeypatch):
    for tool in ("gcc", "patchelf", "readelf", "ldd"):
        if not shutil.which(tool):
            pytest.skip(f"{tool} required for native packaging test")
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "library.c").write_text("int answer(void) { return 42; }\n")
    (sources / "worker.c").write_text(
        "int answer(void); int main(void) { return answer() != 42; }\n"
    )
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            str(sources / "library.c"),
            "-Wl,-soname,libfixture.so",
            "-o",
            str(sources / "libfixture.so"),
        ],
        check=True,
    )
    worker = sources / "worker"
    subprocess.run(
        [
            "gcc",
            str(sources / "worker.c"),
            f"-L{sources}",
            "-lfixture",
            f"-Wl,-rpath,{sources}",
            "-o",
            str(worker),
        ],
        check=True,
    )
    linux = runpy.run_path(str(ROOT / "packaging/linux.py"))
    native = tmp_path / "native"
    linux["bundle"](worker, native)
    assert (native / "libfixture.so").is_file()
    shutil.rmtree(sources)
    relocated = tmp_path / "installed/package/_native"
    relocated.parent.mkdir(parents=True)
    native.rename(relocated)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    linux["check_elf"](relocated / "worker", bundled=True)
    subprocess.run([str(relocated / "worker")], env=dict(os.environ, PATH=""), check=True)
    (relocated / "libfixture.so").unlink()
    with pytest.raises(ValueError, match="Unresolved native dependency"):
        linux["check_elf"](relocated / "worker", bundled=True)
