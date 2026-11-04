from muse_glimmer.resources import resource_path
from transformers import AutoTokenizer


def test_packaged_tokenizer_encodes_and_renders_chat_without_network(monkeypatch):
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("Packaged tokenizer must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    tokenizer = AutoTokenizer.from_pretrained(resource_path("tokenizer"), local_files_only=True)
    text = "Hello from Muse Glimmer."
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    assert text in prompt
    assert "assistant" in prompt
    assert tokenizer.convert_tokens_to_ids("<|eot|>") == 200008
    assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") == 200001
