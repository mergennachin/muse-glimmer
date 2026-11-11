"""Build just the upstream Mac batching worker and its native dependencies."""

from __future__ import annotations

import hashlib
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str], cwd: Path) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def check_metal() -> None:
    # `xcrun --find metal` succeeds even when Xcode 26+ lacks its separately downloaded
    # Metal toolchain, so run the compiler to fail before the long native build starts.
    argv = ["xcrun", "-sdk", "macosx", "metal", "--version"]
    print("+ " + " ".join(argv), flush=True)
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"The Metal compiler is not usable: {detail}\n"
            "Accept the Xcode license with `sudo xcodebuild -license accept` and install "
            "the Metal toolchain with `xcodebuild -downloadComponent MetalToolchain`."
        )


def main(*, editable: bool = False) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Source installation currently supports Apple silicon macOS.")
    for tool in ("git", "cmake", "xcrun", "make"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"Missing build tool: {tool}. Install Xcode with Metal tools and the Python build requirements."
            )
    check_metal()
    acquisition = runpy.run_path(str(ROOT / "packaging/source.py"))
    source = acquisition["source_checkout"](editable=editable)
    revision = acquisition["source_metadata"]()["revision"]
    local_changes = acquisition["local_source_hash"](source) if editable else ""
    source_id = f"{revision}-{local_changes}" if local_changes else revision
    minimum = platform.mac_ver()[0].split(".")[0] + ".0"
    key = hashlib.sha256(
        f"clean-worker:{source}:{source_id}:{minimum}:{sys.version_info[:2]}".encode()
    ).hexdigest()[:16]
    build = Path.home() / "Library/Caches/muse-glimmer/build" / key
    build.mkdir(parents=True, exist_ok=True)
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1))
    if not jobs.isdigit() or int(jobs) < 1:
        raise RuntimeError("CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer")
    # Both pip and editable installs can share the incremental build cache.
    import fcntl

    with (build / ".muse-glimmer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Upstream's normal build adjusts MLX sources. Use a disposable snapshot
        # so the registered submodule stays clean before and after compilation.
        snapshot = build / "sources" / source_id / "executorch"
        if not snapshot.is_dir():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=snapshot.parent) as temporary:
                staged = Path(temporary) / "executorch"
                acquisition["copy_sources"](source, staged, editable=editable)
                staged.rename(snapshot)
        run(
            [
                "cmake",
                "-S",
                str(ROOT / "packaging/native"),
                "-B",
                str(build),
                f"-DEXECUTORCH_SOURCE_DIR={snapshot}",
                f"-DEXECUTORCH_BUILD_PRESET_FILE={snapshot / 'tools/cmake/preset/mlx.cmake'}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}",
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
                "-DEXECUTORCH_BUILD_PYBIND=OFF",
                "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
                "-DEXECUTORCH_BUILD_EXTENSION_ASR_RUNNER=OFF",
                "-DEXECUTORCH_BUILD_KERNELS_LLM=ON",
                "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
                "-DEXECUTORCH_ENABLE_LOGGING=ON",
                "-DET_MLX_ENABLE_OP_LOGGING=OFF",
                "-DET_MIN_LOG_LEVEL=Error",
                "-DBUILD_TESTING=OFF",
                "-DEXECUTORCH_BUILD_CUDA=OFF",
                "-DMUSE_GLIMMER_BUILD_BATCHING=ON",
                "-DMUSE_GLIMMER_BUILD_BATCHING_TESTS=OFF",
            ],
            snapshot,
        )
        run(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "muse_glimmer_batching_worker",
                "--parallel",
                jobs,
            ],
            snapshot,
        )
        worker = build / "examples/models/muse-glimmer-batching"
        # Upstream's copy helper only registers a copy if the Metal library
        # already exists at configure time. Fresh builds need this copy too.
        for line in (build / "CMakeCache.txt").read_text().splitlines():
            if line.startswith("MLX_METALLIB_PATH:INTERNAL="):
                shutil.copy2(Path(line.split("=", 1)[1]), worker / "mlx.metallib")
                break
        else:
            raise RuntimeError("The ExecuTorch build did not provide MLX_METALLIB_PATH")
        staging = runpy.run_path(str(ROOT / "packaging/stage.py"))
        staging["stage"](source, worker, editable=editable)


if __name__ == "__main__":
    main()
