"""CLI download behavior with mocked Hub calls; no model weights are transferred."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from muse_glimmer import download, runtime
from muse_glimmer.cli import main


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(download.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(download.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(download.platform, "mac_ver", lambda: ("26.2", (), "arm64"))
    state = tmp_path / "data" / "default-model"
    monkeypatch.setattr(download, "default_model_file", lambda: state)
    snapshot = tmp_path / "cache" / "snapshots" / ("a" * 40)
    monkeypatch.setattr(download, "snapshot_download", Mock(side_effect=LocalEntryNotFoundError("not cached")))

    def fetch(repo_id, filename, revision, force_download=False):
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return str(path)

    api = Mock()
    info = SimpleNamespace(sha="a" * 40, siblings=[])
    api.model_info.return_value = info
    monkeypatch.setattr(download, "HfApi", lambda: api)
    fetch = Mock(side_effect=fetch)
    monkeypatch.setattr(download, "hf_hub_download", fetch)
    return SimpleNamespace(state=state, snapshot=snapshot, api=api, info=info, fetch=fetch)


def files(hub, *names):
    hub.info.siblings = [SimpleNamespace(rfilename=name, size=100) for name in names]


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text("""
"2.0":
  default_variant: text-solo
  variants:
    text-solo:
      darwin-arm64:
        repo_id: owner/repo
        revision: v2
        directory: "."
    text-image:
      darwin-arm64:
        repo_id: owner/repo
        revision: v2
        directory: "./vision/"
""")
    monkeypatch.setattr(download, "resource_path", lambda name: path)


@pytest.mark.parametrize(
    ("variant", "dflash"),
    [(None, True)]
    + [
        (f"muse-glimmer-k-quant-{quant}-128K-text-{mode}-metal", mode != "solo")
        for quant in ("17G", "dynamic")
        for mode in ("dflash2", "dflash", "solo")
    ],
)
def test_pull_then_serve_uses_downloaded_model(hub, monkeypatch, capsys, variant, dflash):
    folder = variant or "muse-glimmer-k-quant-17G-128K-text-dflash2-metal"
    filename = f"{folder}/{folder}.pte"
    files(hub, filename, "other/model.pte", "README.md", "tokenizer.json")
    args = ["pull", "1.0"]
    if variant is not None:
        args.extend(["--variant", variant])
    assert main(args) == 0
    output = capsys.readouterr().out
    assert f"Selected artifact: {folder}.pte" in output
    assert f"Location: {hub.snapshot / filename}" in output
    assert "Run muse-glimmer serve" in output
    assert "SocialLocalMobile" not in output
    model = hub.snapshot / filename
    assert download.default_model().path == model
    assert download.default_model().dflash is dflash
    hub.api.model_info.assert_called_once()
    assert hub.api.model_info.call_args.args == (
        "SocialLocalMobile/Muse-Glimmer-30B-ExecuTorch-PTE",
    )
    # Catalog entries must pin an immutable snapshot, not a moving branch.
    revision = hub.api.model_info.call_args.kwargs["revision"]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    hub.fetch.assert_called_once_with(
        "SocialLocalMobile/Muse-Glimmer-30B-ExecuTorch-PTE",
        filename,
        revision=hub.info.sha,
        force_download=False,
    )

    # A new invocation serves offline, without fetching metadata or downloading.
    hub.api.model_info.side_effect = AssertionError("serve must not contact the Hub")
    hub.fetch.side_effect = AssertionError("serve must not download")
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main(["serve"]) == 0
    assert serve.call_args.args[0] == model
    assert serve.call_args.args[1].dflash is dflash


@pytest.mark.parametrize(
    ("system", "machine", "variant", "repo", "directory"),
    [
        ("Darwin", "arm64", None, "vision", "vision-metal"),
        ("Darwin", "arm64", "text-solo", "mac", "apple-metal"),
        ("Linux", "x86_64", "text-solo", "linux", "cpu"),
    ],
)
def test_model_catalog_selects_variant_and_platform_location(
    hub, monkeypatch, tmp_path, system, machine, variant, repo, directory
):
    catalog = tmp_path / "models.yaml"
    catalog.write_text("""
"2.0":
  default_variant: text-image
  variants:
    text-solo:
      darwin-arm64:
        repo_id: owner/mac
        revision: mac-v2
        directory: apple-metal
      linux-x86_64:
        repo_id: owner/linux
        revision: linux-v2
        directory: cpu
    text-image:
      darwin-arm64:
        repo_id: owner/vision
        revision: vision-v2
        directory: vision-metal
""")
    monkeypatch.setattr(download, "resource_path", lambda name: catalog)
    monkeypatch.setattr(download.platform, "system", lambda: system)
    monkeypatch.setattr(download.platform, "machine", lambda: machine)
    files(
        hub,
        "apple-metal/model.pte",
        "cpu/model.pte",
        "cpu/aoti_cuda_blob.ptd",
        "vision-metal/model.pte",
        "vision-metal/pos_embed.bin",
    )

    args = ["pull", "2.0"]
    if variant is not None:
        args.extend(["--variant", variant])
    assert main(args) == 0
    hub.api.model_info.assert_called_once_with(
        f"owner/{repo}", revision=f"{repo}-v2", files_metadata=True
    )
    assert download.default_model().path == hub.snapshot / directory / "model.pte"
    expected = [
        call(f"owner/{repo}", f"{directory}/model.pte", revision=hub.info.sha, force_download=False)
    ]
    if directory in {"cpu", "vision-metal"}:
        companion = "aoti_cuda_blob.ptd" if directory == "cpu" else "pos_embed.bin"
        expected.append(
            call(
                f"owner/{repo}", f"{directory}/{companion}",
                revision=hub.info.sha, force_download=False,
            )
        )
    assert hub.fetch.call_args_list == expected


@pytest.mark.parametrize(
    ("version", "system", "machine", "message"),
    [
        ("9.9", "Darwin", "arm64", "Unknown model version '9.9'. Available model versions: 1.0"),
        ("1.0", "Darwin", "x86_64", "not supported on this platform (darwin-x86_64)"),
        ("1.0", "Linux", "aarch64", "not supported on this platform (linux-aarch64)"),
        ("1.0", "Windows", "AMD64", "not supported on this platform (windows-amd64)"),
    ],
)
def test_invalid_model_version_does_not_contact_hub_or_change_default(
    hub, monkeypatch, capsys, version, system, machine, message
):
    monkeypatch.setattr(download.platform, "system", lambda: system)
    monkeypatch.setattr(download.platform, "machine", lambda: machine)
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    assert main(["pull", version]) == 1
    assert message in capsys.readouterr().err
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize(
    "repo_id", ["owner/repo", "SocialLocalMobile/Muse-Glimmer-30B-ExecuTorch-PTE"]
)
@pytest.mark.parametrize("variant", [None, "text-solo"])
def test_repository_id_is_rejected_before_contacting_hub(hub, capsys, repo_id, variant):
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    args = ["pull", repo_id]
    if variant is not None:
        args.extend(["--variant", variant])
    assert main(args) == 1
    error = capsys.readouterr().err
    assert f"Unknown model version {repo_id!r}" in error
    assert "Available model versions: 1.0" in error
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("variant", ["missing", "unknown-folder"])
def test_invalid_model_version_variant_lists_folder_names_without_downloading(hub, capsys, variant):
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    assert main(["pull", "1.0", "--variant", variant]) == 1
    error = capsys.readouterr().err
    assert f"Unknown variant {variant!r}" in error
    assert "Available variants: " in error
    assert "muse-glimmer-k-quant-17G-128K-text-dflash-metal" in error
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_pull_forces_selected_export_only_when_requested(hub, catalog, force, entrypoint):
    files(hub, "solo/model.pte", "vision/model.pte", "vision/pos_embed.bin", "other.bin")
    if entrypoint == "api":
        if force:
            model = download.pull("2.0", "text-image", force=True)
        else:
            model = download.pull("2.0", "text-image")
        assert model == hub.snapshot / "vision/model.pte"
    else:
        args = ["pull", "2.0", "--variant", "text-image"]
        if force:
            args.append("--force")
        assert main(args) == 0
    assert download.default_model().path == hub.snapshot / "vision/model.pte"
    assert hub.fetch.call_args_list == [
        call("owner/repo", "vision/model.pte", revision=hub.info.sha, force_download=force),
        call("owner/repo", "vision/pos_embed.bin", revision=hub.info.sha, force_download=force),
    ]


@pytest.mark.parametrize(
    ("names", "variant", "message"),
    [
        (["README.md"], None, "No .pte files"),
        (["a/model.pte", "b/model.pte"], None, "No .pte files"),
        (["model.pte"], "text-image", "No .pte files"),
        (["a.pte", "b.pte"], None, "multiple .pte files"),
        (["vision/first.pte", "vision/second.pte"], "text-image", "multiple .pte files"),
    ],
)
def test_invalid_selection_does_not_download_or_change_default(
    hub, catalog, names, variant, message
):
    files(hub, *names)
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    with pytest.raises(ValueError, match=message):
        download.pull("2.0", variant)
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("failure_at", ["metadata", "model", "positions"])
@pytest.mark.parametrize("transport", ["httpx", "httpx2"])
def test_download_failure_preserves_previous_default(hub, catalog, failure_at, capsys, transport):
    if importlib.util.find_spec(transport) is None:
        pytest.skip(f"{transport} is not installed with this Hub version")
    files(hub, "model.pte", "pos_embed.bin")
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    error = importlib.import_module(transport).ConnectError("connection lost")
    if failure_at == "metadata":
        hub.api.model_info.side_effect = error
    elif failure_at == "model":
        hub.fetch.side_effect = error
    else:
        hub.fetch.side_effect = [str(hub.snapshot / "model.pte"), error]
    assert main(["pull", "2.0"]) == 1
    assert "connection lost" in capsys.readouterr().err
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("failure_at", ["model", "positions"])
def test_forced_download_reports_underlying_hub_error(hub, catalog, failure_at, capsys):
    files(hub, "model.pte", "pos_embed.bin")
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    error = ValueError("Force download failed due to the above error.")
    error.__cause__ = HfHubHTTPError(
        "403 Forbidden: access denied",
        response=httpx.Response(403, request=httpx.Request("HEAD", "https://example.com/model.pte")),
    )
    hub.fetch.side_effect = (
        error if failure_at == "model" else [str(hub.snapshot / "model.pte"), error]
    )

    assert main(["pull", "2.0", "--force"]) == 1
    message = capsys.readouterr().err
    assert "Could not download Muse Glimmer 2.0: 403 Forbidden: access denied" in message
    assert "above error" not in message
    assert hub.state.read_text() == "/previous/model.pte\n"
    assert list(hub.state.parent.iterdir()) == [hub.state]


def test_cancelled_pull_preserves_previous_default(hub, catalog):
    files(hub, "model.pte")
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    hub.fetch.side_effect = KeyboardInterrupt
    assert main(["pull", "2.0"]) == 130
    assert hub.state.read_text() == "/previous/model.pte\n"


def test_latest_pull_replaces_default(hub, catalog):
    files(hub, "first.pte")
    download.pull("2.0")
    files(hub, "vision/second.pte")
    selected = download.pull("2.0", "text-image")
    assert download.default_model().path == selected
    assert list(hub.state.parent.iterdir()) == [hub.state]


@pytest.mark.parametrize(("source", "kind"), [("cli", "file"), ("config", "directory")])
def test_explicit_model_overrides_pulled_default(hub, catalog, tmp_path, monkeypatch, source, kind):
    files(hub, "model.pte")
    downloaded = download.pull("2.0")
    hub.state.write_text(download.PulledModel(path=downloaded, dflash=False).model_dump_json())
    selected = tmp_path / "local.pte"
    selected.write_bytes(b"local")
    path = selected if kind == "file" else selected.parent
    args = ["serve"]
    if source == "cli":
        args.extend(["--model", str(path)])
    else:
        config = tmp_path / "config.toml"
        config.write_text(f'model = "{path}"\n')
        args.extend(["--config", str(config)])
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main(args) == 0
    assert serve.call_args.args[0] == selected
    assert serve.call_args.args[1].dflash is True
    assert download.default_model().path == downloaded


def test_deleted_download_has_recovery_hint(hub, catalog, capsys):
    files(hub, "model.pte")
    download.pull("2.0").unlink()
    assert main(["serve"]) == 1
    error = capsys.readouterr().err
    assert "muse-glimmer pull --list" in error
    assert "--model /path/to/model.pte" in error


def test_snapshot_symlink_is_preserved_for_vision(hub, catalog, tmp_path):
    files(hub, "model.pte", "pos_embed.bin")
    hub.snapshot.mkdir(parents=True)
    blob = tmp_path / "blob"
    blob.write_bytes(b"model")
    model = hub.snapshot / "model.pte"
    model.symlink_to(blob)
    positions = hub.snapshot / "pos_embed.bin"
    positions.write_bytes(b"positions")
    hub.fetch.side_effect = [str(model), str(positions)]
    download.pull("2.0")
    assert download.default_model().path == model
    assert runtime.resolve_model(hub.snapshot) == model
    assert download.default_model().path.with_name("pos_embed.bin").is_file()


def test_download_size_includes_only_selected_export_and_companions(hub, catalog, capsys):
    hub.info.siblings = [
        SimpleNamespace(rfilename=name, size=size)
        for name, size in {
            "model.pte": 18_000_000_000,
            "vision/model.pte": 19_000_000_000,
            "vision/pos_embed.bin": 100_000_000,
            "tokenizer.json": 28_000_000,
        }.items()
    ]
    assert main(["pull", "2.0", "--variant", "text-image"]) == 0
    assert "Downloading ./vision/ (19.1 GB)" in capsys.readouterr().out
    assert hub.fetch.call_count == 2


@pytest.mark.parametrize("variant", ["text-solo", "text-image"])
def test_root_and_nested_catalog_paths(hub, catalog, variant):
    files(hub, "root.pte", "vision/model.pte")
    model = download.pull("2.0", variant)
    expected = "root.pte" if variant == "text-solo" else "vision/model.pte"
    assert model == hub.snapshot / expected
    hub.fetch.assert_called_once_with("owner/repo", expected, revision=hub.info.sha, force_download=False)


def test_missing_file_sizes_are_not_reported_as_zero(hub, catalog, capsys):
    hub.info.siblings = [
        SimpleNamespace(rfilename="model.pte", size=None),
        SimpleNamespace(rfilename="vision/model.pte", size=100),
    ]
    assert main(["pull", "2.0"]) == 0
    assert "Downloading . (size unknown)" in capsys.readouterr().out
    hub.fetch.assert_called_once_with("owner/repo", "model.pte", revision=hub.info.sha, force_download=False)


def test_empty_default_does_not_select_current_directory(hub, monkeypatch, capsys):
    hub.state.parent.mkdir()
    hub.state.write_text("\n")
    monkeypatch.chdir(hub.state.parent)
    (hub.state.parent / "unselected.pte").write_bytes(b"model")
    assert main(["serve"]) == 1
    assert "muse-glimmer pull" in capsys.readouterr().err


@pytest.mark.parametrize("override", [None, "cli", "config"])
def test_pulled_dflash_setting_and_explicit_overrides(hub, monkeypatch, tmp_path, override):
    source = download.ModelSource(repo_id="owner/repo", revision="v1", directory=".", dflash=True)
    monkeypatch.setattr(download, "_model_source", lambda *args: source)
    files(hub, "model.pte")
    download.pull("1.0")
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    args = ["serve"]
    if override == "cli":
        args.append("--no-dflash")
    elif override == "config":
        config = tmp_path / "settings.toml"
        config.write_text("dflash = false\n")
        args.extend(["--config", str(config)])
    assert main(args) == 0
    assert serve.call_args.args[1].dflash is (override is None)

    # Pulling a solo checkpoint must reset the mode saved by a previous pull.
    source.dflash = False
    download.pull("1.0")
    assert main(["serve"]) == 0
    assert serve.call_args.args[1].dflash is False


def test_legacy_default_path_still_serves(hub, monkeypatch):
    model = hub.snapshot / "old.pte"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"old model")
    hub.state.parent.mkdir()
    hub.state.write_text(str(model) + "\n")
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main(["serve"]) == 0
    assert serve.call_args.args[0] == model
    assert serve.call_args.args[1].dflash is False


@pytest.mark.parametrize("args", [
    [], ["--force"], ["--variant", "text-solo"],
    ["--list", "--remove"], ["--list", "--force"], ["--remove", "--force"],
    ["--list", "--variant", "text-solo"],
    ["--r"], ["--remove", "--variant", "text-solo"],
])
def test_invalid_pull_options_do_not_download_or_remove(hub, args):
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    with pytest.raises(SystemExit) as error:
        main(["pull", *args])
    assert error.value.code == 2
    assert hub.state.read_text() == "/previous/model.pte\n"
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()


@pytest.mark.parametrize("args", [["--help"], ["pull", "--help"], ["serve", "--help"]])
def test_help_is_offline_and_leaves_default_unchanged(hub, capsys, args):
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 0
    output = capsys.readouterr().out
    if args[0] == "pull":
        assert all(option in output for option in ("--force", "--list", "--remove", "--variant"))
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("version", [None, "2.0", "missing"])
def test_list_is_offline_and_can_filter_model_version(hub, tmp_path, monkeypatch, capsys, version):
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text('''
"2.0":
  default_variant: text
  variants:
    text:
      darwin-arm64: {repo_id: owner/repo, revision: v2, directory: text}
    cuda-only:
      linux-x86_64: {repo_id: owner/repo, revision: v2, directory: cuda}
"3.0":
  default_variant: vision
  variants:
    vision:
      darwin-arm64: {repo_id: owner/repo, revision: v3, directory: vision}
''')
    monkeypatch.setattr(download, "resource_path", lambda _: catalog)
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    assert main(["pull", "--list", *([version] if version else [])]) == (1 if version == "missing" else 0)
    output = capsys.readouterr()
    if version == "missing":
        assert "Unknown model version" in output.err
        assert not output.out
    else:
        assert "text [recommended]" in output.out
        assert "cuda-only" not in output.out
        assert ("vision [recommended]" in output.out) is (version is None)
        assert output.out.startswith("SELECTED MODEL\n")
        assert "Artifact: model.pte [missing]" in output.out
        assert "Location: /previous/model.pte" in output.out
        assert output.out.index("Location:") < output.out.index("Available models")
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


def test_list_on_unsupported_platform(hub, catalog, monkeypatch, capsys):
    monkeypatch.setattr(download.platform, "system", lambda: "Windows")
    assert main(["pull", "--list"]) == 0
    output = capsys.readouterr().out
    assert "No models available for this platform" in output
    assert "text-solo" not in output
    assert "Artifact: none" in output
    assert not hub.state.exists()
    hub.api.model_info.assert_not_called()


@pytest.mark.parametrize("args", [["--list"], ["2.0"]])
def test_invalid_catalog_reports_error_without_traceback(hub, tmp_path, monkeypatch, capsys, args):
    catalog = tmp_path / "bad.yaml"
    catalog.write_text("invalid: [")
    monkeypatch.setattr(download, "resource_path", lambda _: catalog)
    assert main(["pull", *args]) == 1
    assert "Invalid model catalog" in capsys.readouterr().err
    hub.api.model_info.assert_not_called()


@pytest.mark.parametrize(("version", "directory"), [("1.1", "base"), ("1.1-small", "small")])
def test_model_versions_are_exact_catalog_identifiers(hub, tmp_path, monkeypatch, capsys, version, directory):
    catalog = tmp_path / "models.yaml"
    catalog.write_text('''
"1.1":
  default_variant: text
  variants:
    text:
      darwin-arm64: {repo_id: owner/base, revision: v1, directory: base}
"1.1-small":
  default_variant: text
  variants:
    text:
      darwin-arm64: {repo_id: owner/small, revision: v1, directory: small}
''')
    monkeypatch.setattr(download, "resource_path", lambda _: catalog)
    files(hub, "base/model.pte", "small/model.pte")
    assert main(["pull", version]) == 0
    hub.api.model_info.assert_called_once_with(f"owner/{directory}", revision="v1", files_metadata=True)
    assert download.default_model().path == hub.snapshot / directory / "model.pte"
    capsys.readouterr()
    assert main(["pull", version, "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.strip() for line in lines if line.startswith("  ") and not line.startswith("    ")] == [version]


@pytest.fixture
def cached_export(tmp_path, monkeypatch):
    # Real Hub cache layout, using only tiny files under pytest's temporary directory.
    repo = tmp_path / "cache" / "models--owner--repo"
    snapshot = repo / "snapshots" / ("a" * 40)
    state = tmp_path / "default-model"
    monkeypatch.setattr(download, "default_model_file", lambda: state)

    def add(name, blob_name, revision=snapshot.name, symlink=True):
        path = repo / "snapshots" / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = repo / "blobs" / blob_name if symlink else path
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(blob_name.encode())
        if symlink:
            path.symlink_to(blob)
        return path, blob

    return SimpleNamespace(repo=repo, snapshot=snapshot, state=state, add=add)


@pytest.mark.parametrize("symlink", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("companion", ["pos_embed.bin", "aoti_cuda_blob.ptd"])
def test_remove_deletes_export_but_preserves_other_variants_and_shared_blobs(cached_export, legacy, symlink, companion, capsys):
    cache = cached_export
    model, model_blob = cache.add("vision/model.pte", "model", symlink=symlink)
    positions, positions_blob = cache.add(f"vision/{companion}", "positions", symlink=symlink)
    other, other_blob = cache.add("solo/model.pte", "other")
    readme, _ = cache.add("vision/README.md", "readme")
    if symlink:
        shared, _ = cache.add(f"vision/{companion}", "positions", revision="b" * 40)
    cache.state.write_text(str(model) if legacy else download.PulledModel(path=model).model_dump_json())

    assert main(["pull", "--remove"]) == 0
    assert "Removed artifact: model.pte" in capsys.readouterr().out
    assert not cache.state.exists()
    assert not model.exists() and not model.is_symlink()
    assert not positions.exists() and not positions.is_symlink()
    assert not model_blob.exists()
    assert other.read_bytes() == other_blob.read_bytes() == b"other"
    assert readme.read_bytes() == b"readme"
    if symlink:
        assert shared.read_bytes() == positions_blob.read_bytes() == b"positions"
    else:
        assert not positions_blob.exists()


def test_remove_does_not_delete_local_model_outside_cache(cached_export, tmp_path, capsys):
    model = tmp_path / "local.pte"
    model.write_bytes(b"local weights")
    cached_export.state.write_text(str(model))
    assert main(["pull", "--remove"]) == 1
    assert "not in a Hugging Face model cache" in capsys.readouterr().err
    assert model.read_bytes() == b"local weights"
    assert cached_export.state.read_text() == str(model)


def test_remove_rejects_cache_link_to_external_file(cached_export, tmp_path, capsys):
    cache = cached_export
    model, _ = cache.add("model.pte", "model")
    outside = tmp_path / "outside.pte"
    outside.write_bytes(b"unrelated weights")
    model.unlink()
    model.symlink_to(outside)
    cache.state.write_text(str(model))
    assert main(["pull", "--remove"]) == 1
    assert "outside its blob directory" in capsys.readouterr().err
    assert outside.read_bytes() == b"unrelated weights"
    assert model.is_symlink() and cache.state.exists()


def test_remove_failure_keeps_default(cached_export, monkeypatch, capsys):
    from huggingface_hub import DeleteCacheStrategy

    model, _ = cached_export.add("model.pte", "model")
    cached_export.state.write_text(str(model))
    monkeypatch.setattr(DeleteCacheStrategy, "execute", Mock(side_effect=PermissionError("denied")))
    assert main(["pull", "--remove"]) == 1
    assert "denied" in capsys.readouterr().err
    assert model.exists() and cached_export.state.read_text() == str(model)


def test_remove_without_default_has_nothing_to_do(hub, capsys):
    assert main(["pull", "--remove"]) == 0
    assert "Nothing to remove" in capsys.readouterr().out
    assert not hub.state.exists()
    hub.api.model_info.assert_not_called()


def test_remove_clears_stale_selection_when_cache_was_deleted(cached_export):
    cache = cached_export
    cache.state.write_text(str(cache.snapshot / "missing.pte"))
    assert main(["pull", "--remove"]) == 0
    assert not cache.state.exists()


def test_remove_preserves_a_new_default_selected_during_deletion(cached_export, monkeypatch):
    from huggingface_hub import DeleteCacheStrategy

    cache = cached_export
    model, _ = cache.add("model.pte", "model")
    other, _ = cache.add("other/model.pte", "other")
    cache.state.write_text(str(model))
    execute = DeleteCacheStrategy.execute

    def select_another_model(strategy):
        cache.state.write_text(str(other))
        execute(strategy)

    monkeypatch.setattr(DeleteCacheStrategy, "execute", select_another_model)
    assert main(["pull", "--remove"]) == 0
    assert not model.exists()
    assert other.exists() and download.default_model().path == other


@pytest.fixture
def local_models(cached_export, monkeypatch):
    cache = cached_export
    catalog = cache.state.with_name("models.yaml")
    catalog.write_text(f'''
"1.1-small":
  default_variant: solo
  variants:
    solo:
      darwin-arm64:
        repo_id: owner/repo
        revision: {'a' * 40}
        directory: solo
        files: [model.pte]
    vision:
      darwin-arm64:
        repo_id: owner/repo
        revision: {'a' * 40}
        directory: vision
        files: [model.pte, pos_embed.bin]
''')
    monkeypatch.setattr(download, "resource_path", lambda _: catalog)
    monkeypatch.setattr(download.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(download.platform, "machine", lambda: "arm64")
    cache.solo, cache.solo_blob = cache.add("solo/model.pte", "solo")
    cache.vision, _ = cache.add("vision/model.pte", "vision")
    cache.positions, _ = cache.add("vision/pos_embed.bin", "positions")
    cache.local = Mock(return_value=str(cache.snapshot))
    monkeypatch.setattr(download, "snapshot_download", cache.local)
    cache.api = Mock()
    cache.api.model_info.side_effect = httpx.ConnectError("offline")
    monkeypatch.setattr(download, "HfApi", lambda: cache.api)
    cache.fetch = Mock(side_effect=AssertionError("must not download cached files"))
    monkeypatch.setattr(download, "hf_hub_download", cache.fetch)
    return cache


def test_switch_list_and_targeted_remove_work_offline(local_models, capsys):
    cache = local_models
    assert main(["pull", "1.1-small"]) == 0
    assert download.default_model().path == cache.solo
    assert main(["pull", "1.1-small", "--variant", "vision"]) == 0
    assert download.default_model().path == cache.vision
    capsys.readouterr()
    assert main(["pull", "--list"]) == 0
    output = capsys.readouterr().out
    assert "solo [recommended] [downloaded]" in output
    assert "vision [downloaded] [selected]" in output
    assert str(cache.solo) in output and str(cache.vision) in output
    assert main(["pull", "1.1-small", "--variant", "solo", "--remove"]) == 0
    assert not cache.solo.exists() and not cache.solo_blob.exists()
    assert download.default_model().path == cache.vision
    assert cache.vision.exists() and cache.positions.exists()
    assert main(["pull", "--remove"]) == 0
    assert not cache.vision.exists() and not cache.positions.exists()
    assert download.default_model() is None
    cache.api.model_info.assert_not_called()
    cache.fetch.assert_not_called()
    assert all(call.kwargs["local_files_only"] for call in cache.local.call_args_list)


@pytest.mark.parametrize("force", [False, True])
def test_incomplete_or_forced_download_needs_network_and_preserves_selection(local_models, capsys, force):
    cache = local_models
    assert main(["pull", "1.1-small"]) == 0
    if not force:
        cache.positions.unlink()
        assert main(["pull", "--list"]) == 0
        assert "vision [incomplete]" in capsys.readouterr().out
    args = ["pull", "1.1-small", "--variant", "vision"] + (["--force"] if force else [])
    assert main(args) == 1
    assert "offline" in capsys.readouterr().err
    assert download.default_model().path == cache.solo
    cache.api.model_info.assert_called_once()


def test_partial_removal_can_be_retried_without_leaking_payload(cached_export, monkeypatch, capsys):
    cache = cached_export
    model, blob = cache.add("model.pte", "weights")
    cache.state.write_text(str(model))
    original = os.remove

    def denied(path, *args, **kwargs):
        if path == blob:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "remove", denied)
        assert main(["pull", "--remove"]) == 1
    assert "retry removal" in capsys.readouterr().err
    assert blob.exists() and model.is_symlink() and cache.state.exists()
    assert main(["pull", "--remove"]) == 0
    assert not blob.exists() and not model.is_symlink() and not cache.state.exists()


def test_removal_handles_aliases_without_deleting_shared_data(cached_export, tmp_path):
    cache = cached_export
    model, blob = cache.add("model.pte", "weights")
    other, _ = cache.add("other/model.pte", "weights")
    alias = tmp_path / "cache-alias"
    alias.symlink_to(cache.repo.parent, target_is_directory=True)
    model.unlink()
    model.symlink_to(alias / cache.repo.name / "blobs" / blob.name)
    cache.state.write_text(str(alias / cache.repo.name / "snapshots" / cache.snapshot.name / "model.pte"))
    assert main(["pull", "--remove"]) == 0
    assert not model.is_symlink()
    assert other.read_bytes() == blob.read_bytes() == b"weights"


def test_remove_cleans_broken_snapshot_link(cached_export):
    model, blob = cached_export.add("model.pte", "weights")
    cached_export.state.write_text(str(model))
    blob.unlink()
    assert main(["pull", "--remove"]) == 0
    assert not model.is_symlink() and not cached_export.state.exists()


def test_corrupt_selection_can_be_listed_and_reset_without_deleting_models(local_models, capsys):
    cache = local_models
    cache.state.write_text("{broken")
    assert main(["pull", "--list"]) == 0
    output = capsys.readouterr().out
    assert "selection is invalid" in output
    assert "solo [recommended] [downloaded]" in output
    assert main(["pull", "--remove"]) == 0
    assert "No model files were removed" in capsys.readouterr().out
    assert cache.solo.exists() and cache.vision.exists() and not cache.state.exists()
