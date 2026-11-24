"""Bundle the CUDA worker's ELF dependencies with paths relative to the wheel."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

SYSTEM_LIBRARIES = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "librt.so.1",
    "ld-linux-x86-64.so.2",
    "libcuda.so.1",
}
CXX_RUNTIME_LIBRARIES = {"libstdc++.so.6", "libgcc_s.so.1"}
CXX_RUNTIME_VERSION = "GLIBCXX_3.4.30"


def provides_cxx_runtime(library: Path) -> bool:
    versions = subprocess.check_output(["readelf", "--version-info", str(library)], text=True)
    return CXX_RUNTIME_VERSION in versions


def check_cxx_runtime(library: Path) -> None:
    if not provides_cxx_runtime(library):
        raise ValueError(
            f"CUDA AOTI models require {CXX_RUNTIME_VERSION}; {library} does not provide it. "
            "Set MUSE_GLIMMER_CXX_RUNTIME_DIR to a GCC 12+ runtime directory."
        )


def dependencies(binary: Path, *, isolated: bool = False) -> dict[str, Path]:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    if isolated:
        env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        ["ldd", str(binary)], env=env, capture_output=True, text=True, check=True
    )
    libraries = {}
    for line in result.stdout.splitlines():
        missing = re.match(r"\s*(\S+) => not found", line)
        if missing and missing[1] not in SYSTEM_LIBRARIES:
            raise ValueError(f"Unresolved native dependency: {missing[1]} ({binary})")
        match = re.match(r"\s*(\S+) => (/.+?) \(0x", line)
        if match and match[1] not in SYSTEM_LIBRARIES:
            libraries[match[1]] = Path(match[2])
    return libraries


def check_elf(binary: Path, *, bundled: bool = False) -> None:
    header = subprocess.check_output(["readelf", "-h", str(binary)], text=True)
    if "Advanced Micro Devices X86-64" not in header or "ELF64" not in header:
        raise ValueError(f"Expected an x86_64 ELF binary: {binary}")
    libraries = dependencies(binary, isolated=bundled)
    if bundled:
        for name, path in libraries.items():
            if path.resolve().parent != binary.resolve().parent:
                raise ValueError(f"Unbundled native dependency: {name} => {path}")
        dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
        paths = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[(.*?)\]", dynamic)
        if paths != ["$ORIGIN"]:
            raise ValueError(f"Native runtime must use only $ORIGIN: {binary} ({paths})")


def bundle(worker: Path, native: Path) -> list[Path]:
    check_elf(worker)
    native.mkdir(parents=True, exist_ok=True)
    pending = {worker.name: worker, **dependencies(worker)}
    runtime = os.environ.get("MUSE_GLIMMER_CXX_RUNTIME_DIR")
    runtime = Path(runtime).expanduser().resolve() if runtime else None
    copied = {}
    while pending:
        name, original = pending.popitem()
        if runtime and name in CXX_RUNTIME_LIBRARIES:
            original = runtime / name
            if not original.is_file():
                raise ValueError(f"Missing C++ runtime library: {original}")
        if name in copied:
            if copied[name].resolve() != original.resolve():
                raise ValueError(f"Conflicting native libraries named {name}")
            continue
        if name.startswith(("libtorch", "libc10", "libpython")):
            raise ValueError(f"Worker must run without PyTorch or Python shared libraries: {name}")
        if name == "libstdc++.so.6":
            check_cxx_runtime(original)
        copied[name] = original
        shutil.copy2(original, native / name)
        pending.update(dependencies(original))
    for name in copied:
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(native / name)], check=True)
    for name in copied:
        check_elf(native / name, bundled=True)
    return list(copied.values())
