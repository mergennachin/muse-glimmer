"""Export the 17G text-only solo model with the package's ExecuTorch revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import runpy
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = "meta-models/Muse-Glimmer-30B-GGUF"
SOURCE_REVISION = "b44f1072cf75e918a1b82a3a17b45c4389b076e7"
SOURCE_FILE = "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
SOURCE_BYTES = 16756683904
SOURCE_SHA256 = "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
VARIANT = "muse-glimmer-k-quant-17G-128K-text-solo-cuda"
MODULE = "executorch.examples.models.muse_glimmer.export.export_solo"


def file_info(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}


def prepare_sdk(source: Path, env: dict[str, str]) -> None:
    """Stage the SDK's generated Python resources outside the clean submodule."""
    python = ROOT / "build/export-python"
    egg = ROOT / "build/export-egg"
    python.mkdir(parents=True, exist_ok=True)
    egg.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(egg),
         "build_py", "--build-lib", str(python)],
        cwd=source, env={**env, "EXECUTORCH_BUILD_MINIMAL": "ON"}, check=True,
    )
    env["PYTHONPATH"] = str(python) + os.pathsep + str(source / "src")
    if not env.get("FLATC_EXECUTABLE") and not shutil.which("flatc", path=env["PATH"]):
        flatc = ROOT / "build/export-flatc"
        subprocess.run(
            ["cmake", "-S", str(source / "third-party/flatbuffers"), "-B", str(flatc),
             "-DCMAKE_BUILD_TYPE=Release", "-DFLATBUFFERS_BUILD_TESTS=OFF",
             "-DFLATBUFFERS_BUILD_FLATC=ON"], env=env, check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(flatc), "--target", "flatc", "--parallel", "8"],
            env=env, check=True,
        )
        env["FLATC_EXECUTABLE"] = str(flatc / "flatc")
    subprocess.run(
        [sys.executable, "-c",
         "import torch; from executorch.exir import to_edge; "
         "p = to_edge(torch.export.export(torch.nn.ReLU(), (torch.randn(2, 3),))).to_executorch(); "
         "assert len(p.buffer) > 8; print('ExecuTorch serialization preflight passed')"],
        env=env, check=True,
    )


def export(
    checkpoint: Path,
    output: Path,
    forward_sizes: str = "1,4",
    *,
    kv_layout: str = "sequence",
    max_kv_seqs: int = 16,
    max_cells: int = 131072,
) -> None:
    import torch

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Export requires Linux with a CUDA GPU.")
    acquisition = runpy.run_path(str(ROOT / "packaging/source.py"))
    source = acquisition["source_checkout"]()
    metadata = acquisition["source_metadata"]()
    expected = runpy.run_path(str(source / "torch_pin.py"))["TORCH_VERSION"]
    if torch.__version__.split("+")[0] != expected or torch.version.cuda != "13.2":
        raise RuntimeError(f"Export requires torch {expected}+cu132; found {torch.__version__}")
    checkpoint = checkpoint.expanduser().resolve()
    if (
        checkpoint.stat().st_size != SOURCE_BYTES
        or file_info(checkpoint)["sha256"] != SOURCE_SHA256
    ):
        raise ValueError(f"Checkpoint does not match the pinned 17G GGUF: {checkpoint}")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.pte")) or any(output.glob("*.ptd")):
        raise ValueError(f"Export output already contains artifacts: {output}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")
    env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
    prepare_sdk(source, env)
    command = [
        sys.executable,
        "-u",
        "-c",
        "import logging, runpy, sys; "
        "logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s'); "
        "logging.getLogger('executorch.backends.cuda.cuda_backend').setLevel(logging.INFO); "
        "sys.argv = sys.argv[1:]; runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)",
        MODULE,
        "--gguf",
        str(checkpoint),
        "--backend",
        "cuda",
        "--activation-dtype",
        "bfloat16",
        "--use-offgraph-kv-cache",
        "--max-seq-len",
        "131072",
        "--max-step-tokens",
        "512",
        "--max-cells",
        str(max_cells),
        "--kv-layout",
        kv_layout,
        "--max-kv-seqs",
        str(max_kv_seqs),
        "--forward-sizes",
        forward_sizes,
        "--output-dir",
        str(output),
    ]
    started = time.monotonic()
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)
    pte = output / "model.pte"
    data = output / "aoti_cuda_blob.ptd"
    if not pte.is_file() or not data.is_file():
        raise RuntimeError("Export did not produce both model.pte and aoti_cuda_blob.ptd")
    pte = pte.rename(output / f"{VARIANT}.pte")
    toolkit = subprocess.check_output(
        [str(Path(env["CUDA_HOME"]) / "bin/nvcc"), "--version"], text=True
    )
    provenance = {
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_files": {SOURCE_FILE: {"bytes": SOURCE_BYTES, "sha256": SOURCE_SHA256}},
        "executorch_repository": metadata["repository"],
        "executorch_revision": metadata["revision"],
        "export_module": MODULE,
        "backend": "cuda",
        "activation_dtype": "bfloat16",
        "offgraph_kv_cache": True,
        "batching": True,
        "dflash": False,
        "vision": False,
        "max_context_len": 131072,
        "max_step_tokens": 512,
        "kv_layout": kv_layout,
        "max_cells": max_cells if kv_layout == "cell" else None,
        "max_kv_seqs": max_kv_seqs if kv_layout == "sequence" else None,
        "forward_sizes": sorted({int(n) for n in forward_sizes.split(",")}),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_toolkit": toolkit.strip(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "export_seconds": time.monotonic() - started,
        "artifacts": [file_info(pte), file_info(data)],
        "export_command": command[1:],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/models" / VARIANT)
    parser.add_argument("--forward-sizes", default="1,4", help="Static batch sizes; default matches CI's smaller configuration")
    parser.add_argument("--kv-layout", choices=("sequence", "cell"), default="sequence",
                        help="Sequence layout uses private histories and sliding-window ring buffers")
    parser.add_argument("--max-kv-seqs", type=int, default=16,
                        help="Sequence slots: default fits four sessions, eight prefixes and four captures")
    parser.add_argument("--max-cells", type=int, default=131072,
                        help="Total shared token cells when --kv-layout=cell")
    args = parser.parse_args()
    if args.max_kv_seqs < 1 or args.max_cells < 2:
        parser.error("--max-kv-seqs must be positive and --max-cells must leave a padding cell")
    export(args.checkpoint, args.output_dir, args.forward_sizes,
           kv_layout=args.kv_layout, max_kv_seqs=args.max_kv_seqs, max_cells=args.max_cells)
