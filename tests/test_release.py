"""Release assembly and retry safety using independent platform wheel archives."""

import importlib.util
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("release", ROOT / "packaging/release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "project"
    (project / ".github").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nversion = "1.2.0.dev1"\n')
    entries = {
        "darwin-arm64": {"runner": ["self-hosted", "macOS", "ARM64"], "workflow": "mac_wheel.yml", "wheel_tag": "py3-none-macosx_26_0_arm64", "description": "Metal", "requirements": "macOS 26.2+"},
        "linux-x86_64-cuda-13_2": {"runner": ["self-hosted", "Linux", "X64", "cuda-13_2"], "workflow": "linux_wheel.yml", "wheel_tag": "py3-none-manylinux_2_28_x86_64", "description": "CUDA fixture", "requirements": "Linux fixture"},
    }
    (project / ".github/release-platforms.json").write_text(json.dumps(entries))
    monkeypatch.setattr(module, "ROOT", project)
    for key, value in {
        "GITHUB_SHA": "a" * 40, "GITHUB_REF": "refs/tags/v1.2.0.dev1",
        "GITHUB_RUN_ID": "123", "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_WORKFLOW_REF": "owner/repo/.github/workflows/wheel.yml@refs/tags/v1.2.0.dev1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(key, value)

    def git(*args):
        if "rev-parse" in args:
            return "a" * 40
        if "ls-tree" in args:
            return "160000 commit " + "b" * 40 + "\tthird-party/executorch"
        raise AssertionError(args)

    monkeypatch.setattr(module, "command", git)
    artifacts = tmp_path / "artifacts"
    for platform, entry in entries.items():
        directory = artifacts / f"muse-glimmer-{platform}"
        directory.mkdir(parents=True)
        filename = f"muse_glimmer-1.2.0.dev1-{entry['wheel_tag']}.whl"
        with zipfile.ZipFile(directory / filename, "w") as archive:
            archive.writestr("muse_glimmer-1.2.0.dev1.dist-info/METADATA", "Name: muse-glimmer\nVersion: 1.2.0.dev1\n")
            archive.writestr("muse_glimmer-1.2.0.dev1.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {entry['wheel_tag']}\n")
            archive.writestr("muse_glimmer/_vendor/provenance.json", json.dumps({"revision": "b" * 40}))
            archive.writestr("muse_glimmer/_native/worker", platform)
        module.record(directory, platform)
    return module, artifacts, tmp_path / "candidate"


def test_plan_and_collect_all_platforms(release):
    module, artifacts, candidate = release
    module.plan()
    outputs = dict(line.split("=", 1) for line in Path(os.environ["GITHUB_OUTPUT"]).read_text().splitlines())
    jobs = json.loads(outputs["build"])["include"]
    assert {j["platform"] for j in jobs} == set(module.platforms())
    assert set(json.loads(outputs["platforms"])) == set(module.platforms())
    assert all("muse-glimmer-123" in j["runner"] for j in jobs)
    assert "cuda-13_2" in next(j["runner"] for j in jobs if j["platform"].startswith("linux"))
    module.collect(artifacts, candidate, False)
    manifest = module.load_bundle(candidate)
    assert len(manifest["wheels"]) == 2
    for entry in manifest["wheels"]:
        original = artifacts / f"muse-glimmer-{entry['platform']}" / entry["filename"]
        assert original.read_bytes() == (candidate / entry["filename"]).read_bytes()
        assert f"{module.digest(original)}  {original.name}\n" in (candidate / "SHA256SUMS").read_text()


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "commit", "ref", "version", "executorch", "bytes", "filename"])
def test_collect_rejects_incomplete_or_mixed_release(release, damage):
    module, artifacts, candidate = release
    directory = artifacts / "muse-glimmer-darwin-arm64"
    manifest = json.loads((directory / "release.json").read_text())
    if damage == "missing":
        shutil.rmtree(directory)
    elif damage == "extra":
        (artifacts / "muse-glimmer-unexpected").mkdir()
    elif damage == "bytes":
        with next(directory.glob("*.whl")).open("ab") as stream:
            stream.write(b"tampered")
    else:
        if damage == "duplicate":
            manifest["wheels"] *= 2
        elif damage == "filename":
            manifest["wheels"][0]["filename"] = "../outside.whl"
        else:
            manifest[damage] = "different"
        (directory / "release.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.collect(artifacts, candidate, False)
    assert not candidate.exists()


def test_record_checks_embedded_pin_and_tag(release):
    module, artifacts, _ = release
    directory = artifacts / "muse-glimmer-darwin-arm64"
    wheel = next(directory.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents["muse_glimmer/_vendor/provenance.json"] = json.dumps({"revision": "c" * 40})
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="clean, pinned"):
        module.record(directory, "darwin-arm64")


def test_duplicate_platform_tags_are_rejected(release):
    module, _, _ = release
    path = module.ROOT / ".github/release-platforms.json"
    entries = json.loads(path.read_text())
    entries["linux-x86_64-cuda-13_2"]["wheel_tag"] = entries["darwin-arm64"]["wheel_tag"]
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="duplicate wheel tag"):
        module.plan()


@pytest.mark.parametrize("metadata", ["expanded", "compressed", "missing", "extra"])
def test_linux_compressed_platform_tags(release, metadata):
    from packaging.tags import parse_tag

    module, artifacts, candidate = release
    platform = "linux-x86_64-cuda-13_2"
    tag = "py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64"
    expanded = sorted(map(str, parse_tag(tag)))
    path = module.ROOT / ".github/release-platforms.json"
    entries = json.loads(path.read_text())
    entries[platform]["wheel_tag"] = tag
    path.write_text(json.dumps(entries))
    directory = artifacts / f"muse-glimmer-{platform}"
    old_wheel = next(directory.glob("*.whl"))
    with zipfile.ZipFile(old_wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    headers = {
        "expanded": expanded,
        "compressed": [tag],
        "missing": expanded[:1],
        "extra": expanded + ["py3-none-linux_x86_64"],
    }[metadata]
    contents["muse_glimmer-1.2.0.dev1.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\n" + "".join(f"Tag: {value}\n" for value in headers)
    )
    old_wheel.unlink()
    with zipfile.ZipFile(directory / f"muse_glimmer-1.2.0.dev1-{tag}.whl", "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    if metadata in {"missing", "extra"}:
        with pytest.raises(ValueError, match="Wheel tags differ"):
            module.record(directory, platform)
    else:
        module.record(directory, platform)
        module.collect(artifacts, candidate, False)
        assert len(module.load_bundle(candidate)["wheels"]) == 2


def test_partially_overlapping_platform_tags_are_rejected(release):
    module, _, _ = release
    path = module.ROOT / ".github/release-platforms.json"
    entries = json.loads(path.read_text())
    entries["linux-other"] = {
        **entries["linux-x86_64-cuda-13_2"],
        "wheel_tag": "py3-none-manylinux_2_28_x86_64.manylinux_2_17_x86_64",
    }
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="Overlapping"):
        module.plan()


@pytest.mark.parametrize("tag", [
    "py3-none-", "py3-none-linux_x86_64.", "py3-none-linux_x86_64..manylinux2014_x86_64",
    "py3-none-../outside", "cp313-cp313-linux_x86_64", "py3-none-linux_x86_64\n",
])
def test_invalid_wheel_tags_are_rejected(release, tag):
    module, _, _ = release
    with pytest.raises(ValueError, match="Invalid wheel tag"):
        module.wheel_tags(tag)


def test_attestation_checks_each_reusable_builder_and_commit(release, monkeypatch):
    module, artifacts, candidate = release
    module.collect(artifacts, candidate, False)
    manifest = module.load_bundle(candidate)
    calls = []
    monkeypatch.setattr(module, "command", lambda *args: calls.append(args))
    module.attest(candidate, manifest)
    assert len(calls) == 2
    for entry, args in zip(manifest["wheels"], calls):
        workflow = module.platforms()[entry["platform"]]["workflow"]
        assert args[args.index("--signer-workflow") + 1] == f"owner/repo/.github/workflows/{workflow}"
        assert args[args.index("--signer-digest") + 1] == "a" * 40
        assert args[args.index("--source-digest") + 1] == "a" * 40
        assert args[args.index("--source-ref") + 1] == "refs/tags/v1.2.0.dev1"


def test_attestation_failure_blocks_collection(release, monkeypatch):
    module, artifacts, candidate = release
    monkeypatch.setattr(module, "attest", Mock(side_effect=RuntimeError("untrusted source")))
    with pytest.raises(RuntimeError, match="untrusted source"):
        module.collect(artifacts, candidate, True)
    assert not candidate.exists()


@pytest.mark.parametrize("conflicting", [False, True])
def test_partial_pypi_retry_preserves_existing_wheel(release, monkeypatch, conflicting):
    module, artifacts, candidate = release
    module.collect(artifacts, candidate, False)
    manifest = module.load_bundle(candidate)
    already, missing = manifest["wheels"]
    monkeypatch.setattr(module, "attest", lambda *args: None)
    monkeypatch.setattr(module, "published_files", lambda version: {
        already["filename"]: {"size": already["size"], "digests": {"sha256": "wrong" if conflicting else already["sha256"]}},
    })
    calls = []
    original = module.command

    def command(*args):
        if args[0] == "git":
            return original(*args)
        calls.append(args)
        return ""

    monkeypatch.setattr(module, "command", command)
    if conflicting:
        with pytest.raises(ValueError, match="different bytes"):
            module.publish(candidate)
        assert calls == []
    else:
        module.publish(candidate)
        uploads = [args for args in calls if args[:2] == ("uv", "publish")]
        assert len(uploads) == 1
        assert str(candidate / missing["filename"]) in uploads[0]
        assert all(str(candidate / already["filename"]) not in args for args in calls)


def test_downloads_foreign_platform_without_installing_it(release, monkeypatch):
    module, artifacts, candidate = release
    module.collect(artifacts, candidate, False)
    entries = module.load_bundle(candidate)["wheels"]
    remote = {w["filename"]: {"size": w["size"], "digests": {"sha256": w["sha256"]}, "url": (candidate / w["filename"]).as_uri()} for w in entries}
    monkeypatch.setattr(module, "published_files", lambda version: remote)
    original = module.command
    calls = []

    def command(*args):
        if args[0] == "git":
            return original(*args)
        calls.append(args)
        return ""

    monkeypatch.setattr(module, "command", command)
    output = candidate.parent / "published"
    module.download_published(candidate, output, None)
    assert module.load_bundle(output) == module.load_bundle(candidate)
    assert len(calls) == 2
    assert all(args[1:5] == ("-m", "pypi_attestations", "verify", "pypi") for args in calls)


def test_no_publication_from_branch(release, monkeypatch):
    module, artifacts, candidate = release
    module.collect(artifacts, candidate, False)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    manifest = json.loads((candidate / "release.json").read_text())
    manifest["ref"] = "refs/heads/main"
    (candidate / "release.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="matching version tag"):
        module.publish(candidate)


def test_workflow_requires_all_tests_before_publish():
    import yaml

    # BaseLoader retains YAML's 'on' key instead of coercing it to boolean True.
    workflow = yaml.load((ROOT / ".github/workflows/wheel.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    assert set(jobs["collect"]["needs"]) == {"plan", "mac", "linux"}
    assert "needs.mac.result == 'success'" in jobs["collect"]["if"]
    assert "needs.linux.result == 'success'" in jobs["collect"]["if"]
    assert jobs["publish"]["needs"] == "collect"
    # Explicit status checks allow disabled platform ancestors, but require the
    # immediate gate to succeed. GitHub otherwise skips publication on tags too.
    for name, dependency in (("publish", "collect"), ("verify-pypi", "publish"), ("github-release", "verify-pypi")):
        assert "!cancelled()" in jobs[name]["if"]
        assert f"needs.{dependency}.result == 'success'" in jobs[name]["if"]
    assert jobs["publish"]["environment"]["name"] == "pypi"
    assert jobs["github-release"]["needs"] == "verify-pypi"
    assert "fromJSON(needs.plan.outputs.build)" in jobs["verify-pypi"]["strategy"]["matrix"]
    assert jobs["mac"]["uses"] == "./.github/workflows/mac_wheel.yml"
    assert jobs["linux"]["uses"] == "./.github/workflows/linux_wheel.yml"
    assert "cuda-model-path" in workflow["on"]["workflow_dispatch"]["inputs"]
    assert jobs["linux"]["with"]["model-path"] == "${{ inputs.cuda-model-path || vars.MUSE_GLIMMER_CUDA_MODEL_PATH }}"
    linux = yaml.load((ROOT / jobs["linux"]["uses"]).read_text(), Loader=yaml.BaseLoader)
    assert linux["on"]["workflow_call"]["inputs"]["model-path"]["required"] == "true"
    mac = yaml.load((ROOT / ".github/workflows/mac_wheel.yml").read_text(), Loader=yaml.BaseLoader)
    assert mac["jobs"]["test"]["needs"] == "build"
    assert mac["jobs"]["test"]["strategy"]["matrix"]["python"] == ["3.11", "3.13"]


def test_published_github_release_is_never_overwritten(release, monkeypatch):
    module, artifacts, candidate = release
    module.collect(artifacts, candidate, False)
    manifest = module.load_bundle(candidate)
    monkeypatch.setattr(module, "attest", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        [], 0, json.dumps({"isDraft": False, "assets": [{"name": w["filename"]} for w in manifest["wheels"]] + [{"name": "SHA256SUMS"}]}), "",
    ))
    original = module.command
    calls = []

    def command(*args):
        if args[0] == "git":
            return original(*args)
        calls.append(args)
        assert args[:3] == ("gh", "release", "download")
        destination = Path(args[-1])
        for path in candidate.iterdir():
            shutil.copy2(path, destination / path.name)
        return ""

    monkeypatch.setattr(module, "command", command)
    module.github_release(candidate)
    assert len(calls) == 1


def test_github_draft_gets_both_wheels_before_publication(release, monkeypatch):
    module, artifacts, candidate = release
    module.collect(artifacts, candidate, False)
    monkeypatch.setattr(module, "attest", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        [], 1, "", "release not found",
    ))
    original = module.command
    calls = []

    def command(*args):
        if args[0] == "git":
            return original(*args)
        calls.append(args)
        if "--notes-file" in args:
            notes = Path(args[args.index("--notes-file") + 1]).read_text()
            assert "darwin-arm64" in notes and "linux-x86_64-cuda-13_2" in notes
        return ""

    monkeypatch.setattr(module, "command", command)
    module.github_release(candidate)
    actions = [args[2] for args in calls if args[:2] == ("gh", "release")]
    assert actions == ["create", "upload", "edit"]
    create, upload, edit = [args for args in calls if args[:2] == ("gh", "release")]
    assert "--draft" in create and "--prerelease" in create
    assert sum("prebuilt wheel" in arg for arg in upload) == 2
    assert str(candidate / "SHA256SUMS") in upload
    assert "--draft=false" in edit
