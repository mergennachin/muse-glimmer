# CUDA release handoff

CUDA 13.2 build and serving support and the Linux workflow are implemented.
Shared CUDA wheel publication remains disabled pending workflow validation and a
PyPI-accepted wheel tag. Model version `1.0` pins the published 17G CUDA text-solo
export for Linux x86_64. Use [RELEASING.md](RELEASING.md)
for the manual Linux validation workflow and the shared release pipeline. Keep
the top-level README limited to working user flows.

## Local validation status

The 17G text-solo CUDA export completed with static forward sizes `1,4` and a
dynamic method through 512 tokens. The installed-wheel live smoke suite passed
17 completed requests and intentional disconnect recovery with four concurrent
requests, including streaming, tools, and prefix reuse. The existing reversed
tool-result case was skipped for ExecuTorch issue 25299.

The rebuilt 26 MB wheel bundles `libstdc++` and `libgcc` and passed this live
smoke suite without PyTorch, build tools on PATH, or an `LD_LIBRARY_PATH`
override. This resolves the model-loading failure caused by the host's older
`libstdc++`. Fresh binary-only installation and a real editable source install
also passed. Source builds retain PyTorch 2.15.0+cu132 for code generation and
native headers; wheel installation and serving do not need PyTorch.

## Supported Linux baseline

Target modern Linux: glibc 2.34+, x86-64-v2, Python 3.11+, and an NVIDIA driver
supporting CUDA 13.2. Older-glibc support is outside the initial release scope;
rebuilding against manylinux_2_28 is not an acceptance requirement. The current
candidate uses the `linux_x86_64` tag and can be installed directly from a wheel
download. A PyPI release still needs an accepted platform tag and an ABI audit
against the chosen modern baseline; do not relabel the raw wheel without that
check. The CUDA 13.2 build and live model smoke have been validated on sm80.

## Reproducing the 17G text-solo export

In an export environment with the pinned ExecuTorch export dependencies, run
the following from the package checkout.

```sh
hf download meta-models/Muse-Glimmer-30B-GGUF \
  Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --revision b44f1072cf75e918a1b82a3a17b45c4389b076e7 --local-dir models
python packaging/export_cuda.py \
  --checkpoint models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
```

The exporter verifies the source checksum and writes the PTE, its companion
PTD, and `provenance.json`. For the cell-layout export with 131072 shared KV
cells, four live sessions, eight cached prefixes, and four capture slots leave
8191 tokens per slot after reserving one padding cell.

## Offline manylinux builder

Run `python packaging/build_manylinux.py --prepare-only` on a connected Linux
x86-64 host. It downloads the CPython 3.11 build wheels, initializes the pinned
ExecuTorch CUDA submodules, and packages their sources with the STB image and
resize headers and license from ExecuTorch's CMake pin. To supply a trusted
local copy of STB, set `MUSE_GLIMMER_STB_INCLUDE_DIR` during preparation.
Preparation records the Muse-Glimmer source fingerprint and archive checksum.
Repeat preparation after switching commits or editing the package sources.

With the manylinux image preloaded and CUDA 13.2 installed, run
`python packaging/build_manylinux.py`. The container builds from the prepared
archive with networking disabled. Use the same `--wheelhouse` path for both
commands if you override the default.

## What is already handled

- One version/tag and source commit for every platform.
- `wheel.yml` coordinates reusable platform workflows, gates publication on all
  tests, and rejects incomplete or mixed artifact sets.
- `packaging/release.py` records/checks metadata, verifies each builder's GitHub
  attestation, publishes via the existing PyPI Trusted Publisher, handles partial
  upload retries, and creates one GitHub Release with all downloads/checksums.
- Collection/publication can inspect foreign-platform wheels without installing
  or executing them. Published installation checks run on their own platform.
- `packaging/release_runner.py` can register Linux x86_64/aarch64 runners against
  the same Actions run as the Mac. `--label cuda-13_2` advertises a capability.
- The Mac build/test workflow remains independent in `mac_wheel.yml`.

## CUDA agent responsibilities

1. **Choose and verify the support baseline.** Confirm CPU architecture, actual
   CUDA toolkit version, minimum NVIDIA driver, supported GPU architectures and
   glibc baseline. `linux-x86_64-cuda-13_2` is the validated CUDA baseline's
   artifact identifier. Update the coordinator's Linux condition
   and runner labels if the final identifier changes.
2. **Implement the native/package path.** Extend `packaging/build_native.py`,
   `packaging/source.py`, `packaging/stage.py`, `packaging/native/CMakeLists.txt`,
   `setup.py` and `packaging/check_wheel.py` as needed for CUDA. Preserve the Mac
   path and the shared Python package/version. Do not ship a Metal-only payload
   under a Linux filename. Audit ELF dependencies, RPATHs and licenses; decide
   which CUDA libraries are bundled or installed as wheel dependencies. The
   NVIDIA driver remains a host requirement.
3. **Implement runtime/model selection.** Update `python/src/muse_glimmer/runtime.py`
   and `models.yaml` to select CUDA and a compatible, revision-pinned export.
   Platform keys in `models.yaml` follow Python's platform naming (for example
   `linux-x86_64`), whereas Actions artifact IDs can include the CUDA baseline.
   Preserve batching, session behavior, DFlash defaults where supported, and
   useful errors for unsupported drivers/GPUs. The model recipe lives in the
   pinned ExecuTorch submodule; coordinate any pin update for both platforms.
4. **Validate `linux_wheel.yml`.** Use a reproducible modern Linux build
   image/toolchain and the existing installed-wheel/live-model test contract.
   Audit the actual ABI and use an accepted wheel tag for PyPI publication;
   changing a filename alone does not establish manylinux compatibility.
5. **Test on real hardware.** Build from clean sources, install the wheel without
   a compiler or CUDA toolkit, start its native worker, run the full applicable
   suite on Python 3.11/3.13 and run `packaging/smoke_wheel.py` against the actual
   GPU/export. Adapt platform-specific checks in existing tests; do not remove
   shared coverage or replace live inference with mocks. Verify tool calls,
   parallel sessions, prefix reuse and disconnect recovery. After publication,
   `check_wheel.py --install` must work on the Linux runner too.
6. **Enable the platform only after these pass.** Add its entry to
   `release-platforms.json`, then run the complete non-publishing workflow with
   both machines. Mac success alone is insufficient to enable CUDA releases.

## Required workflow outputs

Within `linux_wheel.yml`, build exactly one wheel and record it:

```sh
python -m build --wheel
python packaging/release.py record --platform linux-x86_64-cuda-13_2 dist
```

Attest `dist/*.whl` with the same pinned `actions/attest-build-provenance` action
as the Mac workflow. Upload `dist/*.whl`, `dist/release.json` and `dist/SHA256SUMS`
as **`muse-glimmer-linux-x86_64-cuda-13_2`**. Upload test reports separately under
`test-results-linux-x86_64-cuda-13_2-<python>`. Do not publish to PyPI from this
workflow. `record` validates the embedded `_vendor/provenance.json` against the
tagged ExecuTorch gitlink; wheels built from locally modified sources fail.

When enabling, the platform entry needs these fields (values below illustrate
structure; validate the Linux/glibc/driver requirements before filling them in):

```json
{
  "runner": ["self-hosted", "Linux", "X64", "cuda-13_2"],
  "workflow": "linux_wheel.yml",
  "wheel_tag": "py3-none-linux_x86_64",
  "description": "NVIDIA CUDA 13.2",
  "requirements": "glibc 2.34+, x86-64-v2 CPU, Python 3.11+, NVIDIA driver supporting CUDA 13.2"
}
```

The tag above describes the direct-download candidate; replace it with the
validated, PyPI-accepted tag before enabling shared registry publication.
pip understands OS/architecture/Python tags,
**not CUDA versions**. Start with one CUDA baseline per Linux architecture;
two CUDA variants must not produce the same wheel filename or compatibility tag.
If `auditwheel` emits multiple platform tags, put the full dot-separated tag in
`wheel_tag`; collection checks it against the expanded `Tag` headers in the wheel.

## Validation and coordination

Use a review branch containing all CUDA changes. Dispatch **one** `wheel.yml`
run on that branch. Join its run ID from both machines; each downloads its own
runner archive and receives only matching jobs. The helper preserves CUDA_HOME,
CUDA_PATH and CUDACXX and adds discoverable `nvcc`/`nvidia-smi` directories to PATH.
On Linux it also forwards `MUSE_GLIMMER_CXX_RUNTIME_DIR` when the build needs
a GNU C++ runtime outside the compiler's default library directory.
Any additional build setup belongs explicitly in `linux_wheel.yml`.

The Mac must remain available for coordination and its own tests. The Linux
machine must remain available for build, tests and post-publication installation
verification. No source/version changes or manual cross-machine wheel copying
are needed during a run. Both builders use the same checked-out commit.

Do not bump/publish a new package version merely to test CUDA. First pass the
branch workflow, then coordinate one new version/tag for both platforms.
