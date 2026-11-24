# Releasing muse-glimmer

One release version and Git tag cover every supported platform. Each platform
builds and tests on its own machine; the final jobs collect the wheels and
publish the same bytes to PyPI and GitHub Releases.

## Workflow ownership

| File | Responsibility |
| --- | --- |
| [workflows/wheel.yml](workflows/wheel.yml) | Coordinate platforms, require all tests, collect artifacts, request approval, publish and verify both registries. |
| [workflows/mac_wheel.yml](workflows/mac_wheel.yml) | Build the Mac wheel, check installation on Python 3.11/3.13, run the full suite and a live Metal smoke test. |
| [workflows/linux_wheel.yml](workflows/linux_wheel.yml) | Build CUDA 13.2 wheels and run installation, full-suite and live GPU tests; shared releases remain disabled pending validation. |
| [workflows/wheel-linux-cuda.yml](workflows/wheel-linux-cuda.yml) | Manually validate Linux against a local CUDA model before enabling shared releases. |
| [release-platforms.json](release-platforms.json) | Required platforms, runner labels, wheel tags, build workflow identities, and download-table descriptions. |
| [../packaging/release.py](../packaging/release.py) | Artifact metadata, complete-set validation, provenance checks, upload retries and release notes. |

The CUDA implementation and acceptance checklist are in [CUDA_RELEASE.md](CUDA_RELEASE.md).
Only `darwin-arm64` is enabled today. Adding a platform makes it mandatory for
that release; there is no automatic fallback to a partial platform set.

`wheel.yml` retains its name because PyPI Trusted Publishing is configured for
that filename and the `pypi` environment. Platform workflows are reusable
`workflow_call` workflows, called from the same commit as the coordinator.
They do not publish independently.

The coordinator currently runs on the Mac, including collection/publication.
Those operations need Python, `gh`, and network access, but no GPU or native
toolchain. We do not depend on GitHub-hosted machines. Keep all participating
runners alive through the published-installation checks.

## Artifact contract

Each build uploads an Actions artifact named `muse-glimmer-<platform>` containing:

- Exactly one standard Python `.whl` file.
- `release.json`: package version, source commit/ref, ExecuTorch gitlink, platform,
  wheel filename, size and SHA-256. Generate it with `packaging/release.py record`.
- `SHA256SUMS` for that wheel.

The current wheel tag is `py3-none-macosx_26_0_arm64`; the native worker and Metal
library target macOS 26.2, which the CLI checks before serving. Python 3.11+ is
required. ExecuTorch remains pinned by its exact gitlink, not a moving branch.
Linux wheels may use compressed platform tags (for example,
`py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64`). Their expanded tags must
match the wheel metadata and must not overlap another platform's tags.

Build workflows attest their wheels, including on non-publishing manual runs.
Verification checks the platform's reusable workflow identity, workflow commit,
source commit/ref, version, embedded ExecuTorch revision, wheel tag and checksum.
A Linux artifact never needs to execute on the Mac for collection to succeed.

After **all platform tests pass**, collection rejects missing/extra platforms
and mixed or altered artifacts. It creates one `release-candidate` Actions
artifact with every wheel, a combined manifest and `SHA256SUMS`. The publishing
job consumes only that candidate; it never rebuilds a wheel.

## Runner setup

Use a dedicated build account or machine when possible: jobs execute with the
runner account's filesystem permissions. Temporary workspaces are not a sandbox.

All machines need Python 3.11+, Git and an authenticated [GitHub CLI](https://cli.github.com/).
Registration requires repository administration access. The Mac additionally
needs macOS 26.2+, Xcode/Metal and a macOS 26.2+ SDK. Linux's GPU, driver, CUDA
and glibc baseline must be established by the CUDA owner before enabling it.

```sh
gh auth login
gh auth refresh --hostname github.com --scopes workflow
gh auth setup-git
```

The runner helper verifies the GitHub runner archive's checksum and registers
it with a label unique to this workflow run. It supports macOS arm64, Linux
x86_64 and Linux aarch64. Extra `--label` values declare capabilities; they do
not install or verify CUDA. Actual GPU validation belongs in the Linux workflow.

The Mac smoke test pulls model `1.0`'s recommended variant (currently DFlash2).
Allow roughly 20 GB for cached weights and sufficient memory. Smoke tests reuse
the machine's Hugging Face cache but isolate selection state and use a free port.

## Run without publishing

Push a review branch containing these workflows, then:

```sh
gh workflow run wheel.yml --ref YOUR_BRANCH
gh run list --workflow wheel.yml --branch YOUR_BRANCH
python packaging/release_runner.py RUN_ID
```

This runs enabled platform builds, full tests, live smoke tests, attestations,
and collection. A branch run cannot publish. Once CUDA is enabled, join that
**same RUN_ID** on the Linux machine, using its capability label:

```sh
python packaging/release_runner.py RUN_ID --label cuda-13_2
```

The example CUDA label/platform name is provisional until the CUDA owner
confirms the toolkit and architecture. See the handoff before enabling it.

Validate Linux independently with a local export on the runner host:

```sh
gh workflow run wheel-linux-cuda.yml --ref YOUR_BRANCH -f model-path=/absolute/path/to/cuda-export
gh run list --workflow wheel-linux-cuda.yml --branch YOUR_BRANCH
python packaging/release_runner.py RUN_ID --label cuda-13_2
```

For coordinated releases, set the repository Actions variable
`MUSE_GLIMMER_CUDA_MODEL_PATH` to the absolute path of the CUDA export on the
Linux test runner. Tagged builds use that path. To override it on a manual run
of `wheel.yml`, pass `-f cuda-model-path=/absolute/path/to/cuda-export`.

This builds and attests the wheel, tests installation on Python 3.11/3.13, and
runs live generation on Python 3.13. It uploads
`muse-glimmer-linux-x86_64-cuda-13_2` without publishing. The build uses
`linux_x86_64`, targets glibc 2.34+/x86-64-v2, and bundles the GNU C++ runtime.
Older-glibc compatibility is outside the initial release scope. Before PyPI
publication, validate an accepted wheel tag for the chosen modern baseline.
Shared releases still require the acceptance checks in `CUDA_RELEASE.md`.
They record manifests and checksums
through the coordinator's existing artifact contract.

## Publish

Start from an up-to-date `main`, with all intended source and workflow changes
committed. Both machines must build the same tagged commit and pinned submodule.
Set the same new, unused version in `pyproject.toml` and
`python/src/muse_glimmer/__init__.py`.

```sh
release_version=$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
git add pyproject.toml python/src/muse_glimmer/__init__.py
git commit -m "Release ${release_version}"
git tag -a "v${release_version}" -m "muse-glimmer ${release_version}"
git push --atomic origin main "v${release_version}"
gh run list --workflow wheel.yml --branch "v${release_version}"
python packaging/release_runner.py RUN_ID
```

Start the Linux runner against that same run when enabled. Jobs can begin as
machines become available, but publication waits for all of them. A permanently
missing runner leaves the release queued rather than silently dropping a platform.

When validation succeeds, inspect the artifact summaries, then approve GitHub's
protected **pypi** deployment. Trusted Publishing remains configured as:

| Field | Value |
| --- | --- |
| Owner | `mergennachin` |
| Repository | `muse-glimmer` |
| Workflow | `wheel.yml` |
| Environment | `pypi` |

No PyPI API token is needed. After upload, each platform downloads its own wheel
from PyPI, verifies bytes/provenance and tests installation on that platform.
Only when every platform succeeds does the final job publish a GitHub Release,
with all wheels, a download table and combined checksums. `.devN` releases are
marked as prereleases. PyPI's README comes from the tagged repository README.

Wait for `Result: success` on the helpers. Each removes its runner registration
and temporary checkout/build cache. **Ctrl+C on either helper cancels the entire
shared run.** Registration is temporary, so start helpers again for retries.
Transient GitHub reads retry with backoff. If status polling remains unavailable,
the helper keeps its runner online and checks again after 30 seconds; an API
outage does not cancel active jobs. A permanent monitoring error stops that
helper without cancelling the shared run; fix authentication/access and restart
it. An unexpectedly exited runner still cancels the run.

## Failure and retry

```sh
gh run view RUN_ID --log-failed
gh run rerun RUN_ID --failed
python packaging/release_runner.py RUN_ID
```

Restart the Linux helper too if it is needed. Successful jobs/artifacts are
reused by `--failed`; do not mix artifacts from different runs or commits.
Reports remain in `test-results-<platform>-<python>` Actions artifacts. The live
smoke covers tools, parallel child sessions, prefix reuse, streaming/disconnect
recovery and multi-turn isolation. Reversed tool results remain an explicit skip
pending [ExecuTorch #25299](https://github.com/pytorch/executorch/issues/25299).

PyPI cannot atomically publish multiple wheels. A failed upload may leave a
partial version visible. On retry, we compare every existing file's checksum,
skip identical files and upload missing ones. Any conflicting bytes stop the
release; use a new version rather than moving a published tag. GitHub stays a
draft until all assets are attached. Published GitHub assets are verified on
retry and never overwritten.
