# CUDA release handoff

CUDA 13.2 build and serving support and the Linux workflow are implemented.
Shared CUDA publication remains disabled pending real-model acceptance checks
and a revision-pinned CUDA model catalog entry. Use [RELEASING.md](RELEASING.md)
for the manual Linux validation workflow and the shared release pipeline. Keep
the top-level README limited to working user flows.

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
   glibc baseline. `linux-x86_64-cuda-13_2` is a proposed artifact identifier,
   not a promise of CUDA 13.2 support. Update the coordinator's Linux condition
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
4. **Replace the fail-closed `linux_wheel.yml` scaffold.** Use `mac_wheel.yml` as
   the build/test contract, adding the necessary Linux build image/toolchain.
   Use an appropriate manylinux environment and `auditwheel` where applicable;
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
  "wheel_tag": "py3-none-manylinux_2_28_x86_64",
  "description": "NVIDIA CUDA 13.2",
  "requirements": "REPLACE WITH VERIFIED GLIBC, DRIVER, GPU AND PYTHON REQUIREMENTS"
}
```

The tag above is illustrative. pip understands OS/architecture/Python tags,
**not CUDA versions**. Start with one CUDA baseline per Linux architecture;
two CUDA variants must not produce the same wheel filename or compatibility tag.
If `auditwheel` emits multiple platform tags, put the full dot-separated tag in
`wheel_tag`; collection checks it against the expanded `Tag` headers in the wheel.

## Validation and coordination

Use a review branch containing all CUDA changes. Dispatch **one** `wheel.yml`
run on that branch. Join its run ID from both machines; each downloads its own
runner archive and receives only matching jobs. The helper preserves CUDA_HOME,
CUDA_PATH and CUDACXX and adds discoverable `nvcc`/`nvidia-smi` directories to PATH.
Any additional build setup belongs explicitly in `linux_wheel.yml`.

The Mac must remain available for coordination and its own tests. The Linux
machine must remain available for build, tests and post-publication installation
verification. No source/version changes or manual cross-machine wheel copying
are needed during a run. Both builders use the same checked-out commit.

Do not bump/publish a new package version merely to test CUDA. First pass the
branch workflow, then coordinate one new version/tag for both platforms.
