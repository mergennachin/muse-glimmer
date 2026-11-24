import platform
import socket
from unittest.mock import Mock

import pytest
from muse_glimmer.cli import main
from muse_glimmer.config import load_settings


@pytest.mark.parametrize("launch", [None, "muse", "open-webui"])
def test_older_macos_fails_before_loading_model_or_launching_client(monkeypatch, capsys, launch):
    from muse_glimmer import runtime

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: ("26.1", (), "arm64"))
    resolve = Mock(side_effect=AssertionError("must fail before model loading"))
    monkeypatch.setattr(runtime, "resolve_model", resolve)
    args = ["serve", "--model", "/missing/model.pte"]
    if launch:
        args += ["--launch", launch]
    assert main(args) == 1
    assert "Serving requires macOS 26.2 or newer; found 26.1" in capsys.readouterr().err
    resolve.assert_not_called()
    with pytest.raises(SystemExit) as result:
        main(["serve", "--help"])
    assert result.value.code == 0


@pytest.mark.parametrize("version", ["26.2", "26.2.1", "27.0"])
def test_supported_macos_matches_the_build_target(monkeypatch, version):
    import tomllib
    from pathlib import Path

    from muse_glimmer import runtime

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    target = project["tool"]["muse-glimmer"]["macos-deployment-target"]
    assert runtime.MINIMUM_MACOS == tuple(map(int, target.split(".")))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: (version, (), "arm64"))
    runtime.check_platform()


def test_no_command_shows_quickstart_without_starting_a_server(monkeypatch, capsys):
    from muse_glimmer import runtime

    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    assert main([]) == 0
    output = capsys.readouterr()
    assert "muse-glimmer pull 1.0" in output.out
    assert "muse-glimmer serve" in output.out
    assert not output.err
    serve.assert_not_called()


@pytest.mark.parametrize("contents", ["port = 0", "port = [", None])
def test_help_and_model_listing_survive_invalid_defaults(tmp_path, monkeypatch, capsys, contents):
    from muse_glimmer import config, download

    path = tmp_path / "default_config.toml"
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setattr(config, "resource_path", lambda _: path)
    listed = Mock()
    monkeypatch.setattr(download, "list_models", listed)
    assert main([]) == 0
    for args in (["--help"], ["--version"], ["pull", "--help"], ["serve", "--help"]):
        with pytest.raises(SystemExit) as result:
            main(args)
        assert result.value.code == 0
    assert main(["pull", "--list"]) == 0
    listed.assert_called_once_with(None)
    assert not capsys.readouterr().err


@pytest.mark.parametrize("override", ["cli", "config"])
def test_valid_override_repairs_invalid_default_before_validation(tmp_path, monkeypatch, override):
    from muse_glimmer import config, runtime

    path = tmp_path / "default_config.toml"
    path.write_text(config.resource_path("default_config.toml").read_text().replace("port = 8000", "port = 0"))
    monkeypatch.setattr(config, "resource_path", lambda _: path)
    model = tmp_path / "model.pte"
    model.touch()
    serve = Mock()
    monkeypatch.setattr(runtime, "serve", serve)
    args = ["serve", "--model", str(model)]
    if override == "cli":
        args += ["--port", "8001"]
    else:
        overrides = tmp_path / "overrides.toml"
        overrides.write_text("port = 8001")
        args += ["--config", str(overrides)]
    assert main(args) == 0
    assert serve.call_args.args[1].port == 8001


def test_occupied_port_fails_before_loading_native_runtime(tmp_path, monkeypatch, capsys):
    from muse_glimmer import runtime

    model = tmp_path / "model.pte"
    model.touch()
    native = Mock(side_effect=AssertionError("must not load runtime"))
    monkeypatch.setattr(runtime, "native_dir", native)
    with socket.create_server(("127.0.0.1", 0)) as existing:
        port = existing.getsockname()[1]
        assert main(["serve", "--model", str(model), "--port", str(port)]) == 1
        error = capsys.readouterr().err
        assert f"Address already in use: 127.0.0.1:{port}" in error
        assert "--port " in error
        with socket.create_connection(existing.getsockname(), timeout=1):
            pass
    native.assert_not_called()


def test_localhost_reserves_both_loopbacks_and_releases_them_on_failure(tmp_path, monkeypatch):
    from muse_glimmer import runtime

    try:
        reservation = socket.create_server(("::1", 0), family=socket.AF_INET6)
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    with reservation:
        port = reservation.getsockname()[1]

    def inspect_listeners():
        for host in ("127.0.0.1", "::1"):
            with socket.create_connection((host, port), timeout=1):
                pass
        return tmp_path / "missing-runtime"

    monkeypatch.setattr(runtime, "native_dir", inspect_listeners)
    with pytest.raises(ValueError, match="Native runtime missing"):
        runtime.serve(tmp_path / "model.pte", load_settings({"host": "localhost", "port": port}))
    for host, family in (("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)):
        with socket.create_server((host, port), family=family):
            pass


def test_second_bind_failure_releases_first_listener(tmp_path, monkeypatch):
    from muse_glimmer import runtime

    try:
        occupied = socket.create_server(("::1", 0), family=socket.AF_INET6)
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    with occupied:
        port = occupied.getsockname()[1]
        addresses = socket.getaddrinfo("localhost", port, type=socket.SOCK_STREAM)
        monkeypatch.setattr(runtime.socket, "getaddrinfo", lambda *a, **kw: sorted(addresses))
        native = Mock(side_effect=AssertionError("must not load runtime"))
        monkeypatch.setattr(runtime, "native_dir", native)
        with pytest.raises(ValueError, match="Address already in use"):
            runtime.serve(tmp_path / "model.pte", load_settings({"host": "localhost", "port": port}))
        with socket.create_server(("127.0.0.1", port)):
            pass
        native.assert_not_called()


def test_startup_failure_releases_reserved_port(tmp_path, monkeypatch):
    from muse_glimmer import runtime

    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
    monkeypatch.setattr(runtime, "native_dir", lambda: tmp_path / "missing-runtime")
    with pytest.raises(ValueError, match="Native runtime missing"):
        runtime.serve(tmp_path / "model.pte", load_settings({"port": port}))
    with socket.create_server(("127.0.0.1", port)):
        pass


def test_config_file_requires_explicit_selection(tmp_path, monkeypatch):
    expected = load_settings({})

    (tmp_path / "config.toml").write_text("port=9000\n")
    monkeypatch.chdir(tmp_path)
    assert load_settings({}) == expected


@pytest.mark.parametrize(("system", "context", "dflash"), [("Darwin", 16384, True), ("Linux", 8192, False)])
def test_backend_tables_and_override_precedence(tmp_path, monkeypatch, system, context, dflash):
    monkeypatch.setattr(platform, "system", lambda: system)
    assert load_settings({}).dflash is dflash
    assert load_settings({}, model_defaults={"dflash": not dflash}).dflash is not dflash
    path = tmp_path / "config.toml"
    path.write_text(
        "context = 4096\nport = 8001\n"
        "[backends.metal]\ncontext = 16384\ndflash = true\n"
        "[backends.cuda]\ncontext = 8192\ndflash = false\n"
    )
    settings = load_settings({}, path, model_defaults={"dflash": not dflash})
    assert (settings.port, settings.context, settings.dflash) == (8001, context, dflash)
    assert load_settings({"context": 2048}, path).context == 2048


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
    assert serve.call_args.args[1].dflash is (platform.system() != "Linux")


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


@pytest.mark.parametrize("setting", [
    "port = 0", "concurrency = -1", "unknown = 1", "backends = 1",
    "[backends.cdua]\ncontext = 4096", "[backends.cuda]\nunknown = 1",
])
def test_invalid_configuration_is_rejected(tmp_path, setting):
    path = tmp_path / "config.toml"
    path.write_text(setting)
    with pytest.raises(ValueError):
        load_settings({}, path)
