"""Build the upstream batching worker and its native dependencies."""

from __future__ import annotations

import hashlib
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
import tomllib
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


def check_macos(minimum: str) -> None:
    target = tuple(map(int, minimum.split(".")))
    host = platform.mac_ver()[0]
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
        text=True,
    ).strip()
    for name, version in (("build host", host), ("SDK", sdk)):
        if tuple(map(int, version.split(".")[:2])) < target:
            raise RuntimeError(
                f"Building muse-glimmer requires macOS {minimum}+ {name}; found {version}."
            )


def main(*, editable: bool = False) -> None:
    target = (platform.system(), platform.machine())
    if target not in {("Darwin", "arm64"), ("Linux", "x86_64")}:
        raise RuntimeError(
            "Source installation supports Apple silicon macOS or Linux x86_64 with CUDA."
        )
    cuda = target[0] == "Linux"
    for tool in (
        ("git", "cmake", "nvcc", "patchelf", "readelf", "ldd", "make")
        if cuda
        else ("git", "cmake", "xcrun", "make")
    ):
        if not shutil.which(tool):
            raise RuntimeError(
                f"Missing build tool: {tool}. Install the platform toolchain and Python build requirements."
            )
    minimum = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["muse-glimmer"][
        "macos-deployment-target"
    ]
    if cuda:
        import torch

        if torch.version.cuda != "13.2":
            raise RuntimeError("Building the CUDA worker requires PyTorch built with CUDA 13.2.")
    else:
        check_macos(minimum)
        check_metal()
    acquisition = runpy.run_path(str(ROOT / "packaging/source.py"))
    source = acquisition["source_checkout"](editable=editable)
    revision = acquisition["source_metadata"]()["revision"]
    if cuda:
        expected_torch = runpy.run_path(str(source / "torch_pin.py"))["TORCH_VERSION"]
        if torch.__version__.split("+")[0] != expected_torch:
            raise RuntimeError(
                f"ExecuTorch requires torch {expected_torch}; found {torch.__version__}"
            )
        toolkit = subprocess.check_output(["nvcc", "--version"], text=True)
        if "release 13.2," not in toolkit:
            raise RuntimeError("The Linux wheel requires the CUDA 13.2 toolkit.")
    local_changes = acquisition["local_source_hash"](source) if editable else ""
    source_id = f"{revision}-{local_changes}" if local_changes else revision
    key = hashlib.sha256(
        f"clean-worker:{target}:{source}:{source_id}:{minimum}:{sys.version_info[:2]}:{torch.__version__ if cuda else 'mlx'}:{os.environ.get('TORCH_CUDA_ARCH_LIST', '')}".encode()
    ).hexdigest()[:16]
    # A temporary release runner supplies its own cache directory for cleanup.
    cache = os.environ.get(
        "MUSE_GLIMMER_BUILD_CACHE",
        Path.home()
        / (".cache/muse-glimmer/build" if cuda else "Library/Caches/muse-glimmer/build"),
    )
    build = Path(cache).expanduser() / key
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
        backend_options = (
            [
                f"-DEXECUTORCH_BUILD_PRESET_FILE={snapshot / 'tools/cmake/preset/linux.cmake'}",
                "-DEXECUTORCH_BUILD_CUDA=ON",
                "-DEXECUTORCH_BUILD_MLX=OFF",
                "-DEXECUTORCH_BUILD_XNNPACK=OFF",
                "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
                "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            ]
            if cuda
            else [
                f"-DEXECUTORCH_BUILD_PRESET_FILE={snapshot / 'tools/cmake/preset/mlx.cmake'}",
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}",
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
                "-DEXECUTORCH_BUILD_CUDA=OFF",
            ]
        )
        run(
            [
                "cmake",
                "-S",
                str(ROOT / "packaging/native"),
                "-B",
                str(build),
                f"-DEXECUTORCH_SOURCE_DIR={snapshot}",
                *backend_options,
                *(
                    [f"-DMUSE_GLIMMER_STB_INCLUDE_DIR={os.environ['MUSE_GLIMMER_STB_INCLUDE_DIR']}"]
                    if os.environ.get("MUSE_GLIMMER_STB_INCLUDE_DIR")
                    else []
                ),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                "-DEXECUTORCH_BUILD_PYBIND=OFF",
                "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
                "-DEXECUTORCH_BUILD_EXTENSION_ASR_RUNNER=OFF",
                "-DEXECUTORCH_BUILD_KERNELS_LLM=ON",
                "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
                "-DEXECUTORCH_ENABLE_LOGGING=ON",
                "-DET_MLX_ENABLE_OP_LOGGING=OFF",
                "-DET_MIN_LOG_LEVEL=Error",
                "-DBUILD_TESTING=OFF",
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
        if not cuda:
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
