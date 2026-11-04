from unittest.mock import Mock

import pytest
from muse_glimmer.cli import main
from muse_glimmer.config import load_settings


def test_config_file_requires_explicit_selection(tmp_path, monkeypatch):
    expected = load_settings({})

    (tmp_path / "config.toml").write_text("port=9000\n")
    monkeypatch.chdir(tmp_path)
    assert load_settings({}) == expected


def test_explicit_config_is_preserved_and_model_flag_overrides_it(tmp_path):
    path = tmp_path / "config.toml"
    original = '# My settings\nmodel="/models/saved"\nport=8080\n'
    path.write_text(original)
    settings = load_settings({"model": "/models/selected"}, path)
    assert settings.model == "/models/selected"
    assert settings.port == 8080
    assert settings.host == "127.0.0.1"
    assert path.read_text() == original


def test_missing_explicit_config_is_not_created(tmp_path):
    path = tmp_path / "missing.toml"
    with pytest.raises(ValueError, match="Configuration file does not exist"):
        load_settings({}, path)
    assert not path.exists()


def test_default_serve_reports_missing_model(capsys, tmp_path, monkeypatch):
    from muse_glimmer import download

    monkeypatch.setattr(download, "default_model_file", lambda: tmp_path / "default-model")
    assert main(["serve"]) == 1
    error = capsys.readouterr().err
    assert "muse-glimmer pull" in error and "--model" in error


@pytest.mark.parametrize("name", ["missing.pte", "model.json"])
def test_invalid_model_path_is_reported(tmp_path, name, capsys):
    (tmp_path / "model.json").write_text("{}")
    assert main(["serve", "--model", str(tmp_path / name)]) == 1
    assert "Expected an existing .pte file" in capsys.readouterr().err


@pytest.mark.parametrize("contents", [[], ["nested/model.pte"], ["first.pte", "second.pte"]])
def test_directory_requires_exactly_one_immediate_pte(tmp_path, contents, capsys):
    for name in contents:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"model")
    assert main(["serve", "--model", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    if len(contents) == 2:
        assert "Pass the explicit file with --model" in error
        assert "first.pte" in error and "second.pte" in error
    else:
        assert "No .pte file found directly" in error


def test_explicit_file_still_works_in_directory_with_multiple_models(tmp_path, monkeypatch):
    from muse_glimmer import runtime

    for name in ["first.pte", "second.pte"]:
        (tmp_path / name).write_bytes(b"model")
    selected = tmp_path / "second.pte"
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main(["serve", "--model", str(selected)]) == 0
    assert serve.call_args.args[0] == selected


def test_no_dflash_overrides_enabled_config(tmp_path, monkeypatch):
    from muse_glimmer import runtime

    model = tmp_path / "model.pte"
    model.write_bytes(b"test model")
    config = tmp_path / "config.toml"
    config.write_text("dflash = true\n")
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main(["serve", "--model", str(model), "--config", str(config), "--no-dflash"]) == 0
    assert serve.call_args.args[1].dflash is False


@pytest.mark.parametrize("setting", ["port = 0", "concurrency = -1", "unknown = 1"])
def test_invalid_configuration_is_rejected(tmp_path, setting):
    path = tmp_path / "config.toml"
    path.write_text(setting)
    with pytest.raises(ValueError):
        load_settings({}, path)
