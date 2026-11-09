"""CLI download behavior with mocked Hub calls; no model weights are transferred."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from muse_glimmer import download, runtime
from muse_glimmer.cli import main


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(download.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(download.platform, "machine", lambda: "arm64")
    state = tmp_path / "data" / "default-model"
    monkeypatch.setattr(download, "default_model_file", lambda: state)
    snapshot = tmp_path / "cache" / "snapshots" / ("a" * 40)

    def fetch(repo_id, filename, revision):
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
        (f"muse-glimmer-k-quant-{quant}-128K-text-{mode}-metal", mode == "dflash")
        for quant in ("17G", "dynamic")
        for mode in ("dflash", "solo")
    ],
)
def test_pull_then_serve_uses_downloaded_model(hub, monkeypatch, capsys, variant, dflash):
    folder = variant or "muse-glimmer-k-quant-17G-128K-text-dflash-metal"
    filename = f"{folder}/{folder}.pte"
    files(hub, filename, "other/model.pte", "README.md", "tokenizer.json")
    args = ["pull", "1.0"]
    if variant is not None:
        args.extend(["--variant", variant])
    assert main(args) == 0
    output = capsys.readouterr().out
    assert f"Default model: {folder}" in output
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
        "SocialLocalMobile/Muse-Glimmer-30B-ExecuTorch-PTE", filename, revision=hub.info.sha
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
def test_release_catalog_selects_variant_and_platform_location(
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
        "cpu/pos_embed.bin",
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
    expected = [call(f"owner/{repo}", f"{directory}/model.pte", revision=hub.info.sha)]
    if directory in {"cpu", "vision-metal"}:
        expected.append(call(f"owner/{repo}", f"{directory}/pos_embed.bin", revision=hub.info.sha))
    assert hub.fetch.call_args_list == expected


@pytest.mark.parametrize(
    ("release", "system", "machine", "message"),
    [
        ("9.9", "Darwin", "arm64", "Unknown release '9.9'. Available releases: 1.0"),
        ("1.0", "Darwin", "x86_64", "not supported on this platform (darwin-x86_64)"),
        ("1.0", "Linux", "aarch64", "not supported on this platform (linux-aarch64)"),
        ("1.0", "Windows", "AMD64", "not supported on this platform (windows-amd64)"),
    ],
)
def test_invalid_release_does_not_contact_hub_or_change_default(
    hub, monkeypatch, capsys, release, system, machine, message
):
    monkeypatch.setattr(download.platform, "system", lambda: system)
    monkeypatch.setattr(download.platform, "machine", lambda: machine)
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    assert main(["pull", release]) == 1
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
    assert f"Unknown release {repo_id!r}" in error
    assert "Available releases: 1.0" in error
    hub.api.model_info.assert_not_called()
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("variant", ["missing", "unknown-folder"])
def test_invalid_release_variant_lists_folder_names_without_downloading(hub, capsys, variant):
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


def test_selected_nested_export_downloads_only_matching_vision_data(hub, catalog):
    files(hub, "solo/model.pte", "vision/model.pte", "vision/pos_embed.bin", "other.bin")
    assert main(["pull", "2.0", "--variant", "text-image"]) == 0
    assert download.default_model().path == hub.snapshot / "vision/model.pte"
    assert hub.fetch.call_args_list == [
        call("owner/repo", "vision/model.pte", revision=hub.info.sha),
        call("owner/repo", "vision/pos_embed.bin", revision=hub.info.sha),
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
    hub.state.write_text(download.PulledModel(path=downloaded, dflash=True).model_dump_json())
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
    assert serve.call_args.args[1].dflash is False
    assert download.default_model().path == downloaded


def test_deleted_download_has_recovery_hint(hub, catalog, capsys):
    files(hub, "model.pte")
    download.pull("2.0").unlink()
    assert main(["serve"]) == 1
    assert "Pull again or use --model" in capsys.readouterr().err


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
    hub.fetch.assert_called_once_with("owner/repo", expected, revision=hub.info.sha)


def test_missing_file_sizes_are_not_reported_as_zero(hub, catalog, capsys):
    hub.info.siblings = [
        SimpleNamespace(rfilename="model.pte", size=None),
        SimpleNamespace(rfilename="vision/model.pte", size=100),
    ]
    assert main(["pull", "2.0"]) == 0
    assert "Downloading . (size unknown)" in capsys.readouterr().out
    hub.fetch.assert_called_once_with("owner/repo", "model.pte", revision=hub.info.sha)


def test_empty_default_does_not_select_current_directory(hub, monkeypatch, capsys):
    hub.state.parent.mkdir()
    hub.state.write_text("\n")
    monkeypatch.chdir(hub.state.parent)
    (hub.state.parent / "unselected.pte").write_bytes(b"model")
    assert main(["serve"]) == 1
    assert "muse-glimmer pull" in capsys.readouterr().err


@pytest.mark.parametrize("override", [None, "cli", "config"])
def test_pulled_dflash_setting_and_explicit_overrides(hub, monkeypatch, tmp_path, override):
    source = download.ReleaseSource(repo_id="owner/repo", revision="v1", directory=".", dflash=True)
    monkeypatch.setattr(download, "_release_source", lambda *args: source)
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
