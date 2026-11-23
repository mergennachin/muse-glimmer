import httpx
import pytest
from muse_glimmer import Glimmer


def test_generate_sends_independent_prompts(monkeypatch):
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs))
        message = {"reasoning_content": "thinking", "content": "Hello!"}
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"choices": [{"message": message}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    client = Glimmer()
    assert client.generate("First", system="Be brief.", temperature=0, max_tokens=64) == "Hello!"
    assert client.generate("Second") == "Hello!"
    url, kwargs = requests[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert kwargs["json"] == {
        "model": "muse-glimmer-30B",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "First"},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    assert requests[1][1]["json"] == {
        "model": "muse-glimmer-30B",
        "messages": [{"role": "user", "content": "Second"}],
    }


def test_generate_returns_empty_string_for_reasoning_only_reply(monkeypatch):
    message = {"role": "assistant", "reasoning_content": "Let me think..."}
    reply = {"choices": [{"message": message, "finish_reason": "length"}]}
    client = Glimmer()
    monkeypatch.setattr(client, "chat", lambda messages, **options: reply)

    assert client.generate("Hi", max_tokens=1) == ""


def test_chat_preserves_request_options_and_full_response(monkeypatch):
    messages = [{"role": "user", "content": "Hello"}]
    completion = {"choices": [{"message": {"content": "Hi"}}], "usage": {"total_tokens": 16}}

    def post(url, *, json, timeout, trust_env, headers):
        assert url == "http://localhost:8001/v1/chat/completions"
        assert json == {"model": "custom", "messages": messages, "tools": []}
        assert timeout == 42
        return httpx.Response(200, request=httpx.Request("POST", url), json=completion)

    monkeypatch.setattr(httpx, "post", post)
    client = Glimmer(base_url="http://localhost:8001/v1/", model="custom", timeout=42)
    assert client.chat(messages, tools=[]) == completion
    with pytest.raises(ValueError, match="stream=False"):
        client.chat(messages, stream=True)


def test_connection_error_includes_startup_command(monkeypatch):
    def post(url, **kwargs):
        raise httpx.ConnectError("Connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(httpx.ConnectError, match="muse-glimmer serve"):
        Glimmer().generate("Hello")


def test_request_timing_is_opt_in_and_goes_to_stderr(monkeypatch, capsys):
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://localhost"),
        json={"choices": [{"message": {"content": "ok"}}]},
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    assert Glimmer().generate("Hi") == "ok"
    assert capsys.readouterr().err == ""
    assert Glimmer(timings=True).generate("Hi") == "ok"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "server timings unavailable\n"


def test_chat_reports_server_measurements(monkeypatch, capsys):
    def post(url, *, headers, **kwargs):
        assert headers == {"x-glimmer-timings": "1"}
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={},
            headers={"x-glimmer-timings": '{"ttft_ms":180,"decode_tok_s":42}'},
        )

    monkeypatch.setattr(httpx, "post", post)
    assert Glimmer(timings=True).chat([])["timings"] == {"ttft_ms": 180, "decode_tok_s": 42}
    assert capsys.readouterr().err == "TTFT 180.0 ms | prefill n/a | decode 42.0 tok/s\n"
