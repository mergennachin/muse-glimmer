"""CLI download behavior with mocked Hub calls; no model weights are transferred."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
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


def test_pull_then_serve_uses_downloaded_model(hub, monkeypatch, capsys):
    files(hub, "model.pte", "README.md", "tokenizer.json")
    assert main(["pull", "owner/repo"]) == 0
    assert "Run muse-glimmer serve" in capsys.readouterr().out
    model = hub.snapshot / "model.pte"
    assert hub.state.read_text().strip() == str(model)
    hub.fetch.assert_called_once_with("owner/repo", "model.pte", revision=hub.info.sha)

    # A new invocation serves offline, without fetching metadata or downloading.
    hub.api.model_info.side_effect = AssertionError("serve must not contact the Hub")
    hub.fetch.side_effect = AssertionError("serve must not download")
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main(["serve"]) == 0
    assert serve.call_args.args[0] == model


def test_selected_nested_export_downloads_only_matching_vision_data(hub):
    files(hub, "solo/model.pte", "vision/model.pte", "vision/pos_embed.bin", "other.bin")
    assert main(["pull", "owner/repo", "--variant", "vision"]) == 0
    assert download.default_model() == hub.snapshot / "vision/model.pte"
    assert hub.fetch.call_args_list == [
        call("owner/repo", "vision/model.pte", revision=hub.info.sha),
        call("owner/repo", "vision/pos_embed.bin", revision=hub.info.sha),
    ]


@pytest.mark.parametrize(
    ("names", "variant", "message"),
    [
        (["README.md"], None, "No .pte files"),
        (["a/model.pte", "b/model.pte"], None, "--variant"),
        (["a/model.pte"], "missing", "No export directory named"),
        (["a.pte", "b.pte"], None, "multiple .pte files"),
        (["a/first.pte", "a/second.pte"], "a", "multiple .pte files"),
    ],
)
def test_invalid_selection_does_not_download_or_change_default(hub, names, variant, message):
    files(hub, *names)
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    with pytest.raises(ValueError, match=message):
        download.pull("owner/repo", variant)
    hub.fetch.assert_not_called()
    assert hub.state.read_text() == "/previous/model.pte\n"


@pytest.mark.parametrize("failure_at", ["metadata", "model", "positions"])
def test_download_failure_preserves_previous_default(hub, failure_at, capsys):
    files(hub, "model.pte", "pos_embed.bin")
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    error = httpx.ConnectError("connection lost")
    if failure_at == "metadata":
        hub.api.model_info.side_effect = error
    elif failure_at == "model":
        hub.fetch.side_effect = error
    else:
        hub.fetch.side_effect = [str(hub.snapshot / "model.pte"), error]
    assert main(["pull", "owner/repo"]) == 1
    assert "connection lost" in capsys.readouterr().err
    assert hub.state.read_text() == "/previous/model.pte\n"


def test_cancelled_pull_preserves_previous_default(hub):
    files(hub, "model.pte")
    hub.state.parent.mkdir()
    hub.state.write_text("/previous/model.pte\n")
    hub.fetch.side_effect = KeyboardInterrupt
    assert main(["pull", "owner/repo"]) == 130
    assert hub.state.read_text() == "/previous/model.pte\n"


def test_latest_pull_replaces_default(hub):
    files(hub, "first.pte")
    download.pull("owner/first")
    files(hub, "second.pte")
    selected = download.pull("owner/second")
    assert download.default_model() == selected
    assert list(hub.state.parent.iterdir()) == [hub.state]


@pytest.mark.parametrize(("source", "kind"), [("cli", "file"), ("config", "directory")])
def test_explicit_model_overrides_pulled_default(hub, tmp_path, monkeypatch, source, kind):
    files(hub, "model.pte")
    downloaded = download.pull("owner/repo")
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
    assert download.default_model() == downloaded


def test_deleted_download_has_recovery_hint(hub, capsys):
    files(hub, "model.pte")
    download.pull("owner/repo").unlink()
    assert main(["serve"]) == 1
    assert "Pull again or use --model" in capsys.readouterr().err


def test_snapshot_symlink_is_preserved_for_vision(hub, tmp_path):
    files(hub, "model.pte", "pos_embed.bin")
    hub.snapshot.mkdir(parents=True)
    blob = tmp_path / "blob"
    blob.write_bytes(b"model")
    model = hub.snapshot / "model.pte"
    model.symlink_to(blob)
    positions = hub.snapshot / "pos_embed.bin"
    positions.write_bytes(b"positions")
    hub.fetch.side_effect = [str(model), str(positions)]
    download.pull("owner/repo")
    assert download.default_model() == model
    assert runtime.resolve_model(hub.snapshot) == model
    assert Path(hub.state.read_text().strip()).with_name("pos_embed.bin").is_file()


def test_variant_list_filters_cuda_and_includes_companion_size(hub, capsys):
    hub.info.siblings = [
        SimpleNamespace(rfilename=name, size=size)
        for name, size in {
            "muse-glimmer-text-solo-metal/solo.pte": 18_000_000_000,
            "muse-glimmer-text-image-solo-metal/vision.pte": 19_000_000_000,
            "muse-glimmer-text-image-solo-metal/pos_embed.bin": 100_000_000,
            "muse-glimmer-text-solo-sm80+ptx/solo.pte": 20_000_000,
            "muse-glimmer-text-solo-sm80+ptx/solo.ptd": 20_000_000_000,
            "tokenizer.json": 28_000_000,
        }.items()
    ]
    assert main(["pull", "owner/repo"]) == 1
    error = capsys.readouterr().err
    assert "muse-glimmer-text-solo-metal (18.0 GB)" in error
    assert "muse-glimmer-text-image-solo-metal (19.1 GB)" in error
    assert "sm80" not in error
    hub.fetch.assert_not_called()
    assert not hub.state.exists()


def test_single_metal_variant_is_selected_automatically(hub):
    metal = "muse-glimmer-text-solo-metal"
    cuda = "muse-glimmer-text-solo-sm80+ptx"
    files(hub, f"{metal}/{metal}.pte", f"{cuda}/{cuda}.pte", f"{cuda}/{cuda}.ptd")
    model = download.pull("owner/repo")
    assert model == hub.snapshot / metal / f"{metal}.pte"
    hub.fetch.assert_called_once_with("owner/repo", f"{metal}/{metal}.pte", revision=hub.info.sha)


@pytest.mark.parametrize("variant", [None, "muse-glimmer-text-solo-sm80+ptx"])
def test_cuda_only_exports_are_rejected_before_downloading(hub, variant):
    files(hub, "muse-glimmer-text-solo-sm80+ptx/model.pte")
    with pytest.raises(ValueError, match="platform"):
        download.pull("owner/repo", variant)
    hub.fetch.assert_not_called()
    assert not hub.state.exists()


@pytest.mark.parametrize(("system", "machine"), [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_metal_is_not_selected_on_other_platforms(hub, monkeypatch, system, machine):
    monkeypatch.setattr(download.platform, "system", lambda: system)
    monkeypatch.setattr(download.platform, "machine", lambda: machine)
    files(hub, "muse-glimmer-text-solo-metal/model.pte")
    with pytest.raises(ValueError, match="platform"):
        download.pull("owner/repo")
    hub.fetch.assert_not_called()


@pytest.mark.parametrize("variant", [".", "./nested/"])
def test_root_and_nested_variant_paths(hub, variant):
    files(hub, "root.pte", "nested/model.pte")
    model = download.pull("owner/repo", variant)
    expected = "root.pte" if variant == "." else "nested/model.pte"
    assert model == hub.snapshot / expected
    hub.fetch.assert_called_once_with("owner/repo", expected, revision=hub.info.sha)


def test_missing_file_sizes_are_not_reported_as_zero(hub, capsys):
    hub.info.siblings = [
        SimpleNamespace(rfilename="a/model.pte", size=None),
        SimpleNamespace(rfilename="b/model.pte", size=100),
    ]
    assert main(["pull", "owner/repo"]) == 1
    assert "a (size unknown)" in capsys.readouterr().err
    hub.fetch.assert_not_called()


def test_empty_default_does_not_select_current_directory(hub, monkeypatch, capsys):
    hub.state.parent.mkdir()
    hub.state.write_text("\n")
    monkeypatch.chdir(hub.state.parent)
    (hub.state.parent / "unselected.pte").write_bytes(b"model")
    assert main(["serve"]) == 1
    assert "muse-glimmer pull" in capsys.readouterr().err
