# muse-glimmer

Serve Muse Glimmer locally. Development preview.

## Supported platforms

- macOS on Apple silicon

## Install and serve

You need Python 3.11+, Git, and Xcode with the Metal toolchain.
The first installation needs an internet connection. From this repository:

```sh
pip install -e .
muse-glimmer pull SocialLocalMobile/Muse-Glimmer-30B-ExecuTorch-PTE
muse-glimmer serve
```

For repositories with several exports, `pull` lists variants and download sizes.
Choose one with `muse-glimmer pull HF/repo_id --variant <directory-name>`.
Downloads use the Hugging Face cache; the last successful pull becomes the default.
Use `muse-glimmer serve --model <path>` for a local `.pte` file or directory containing one.

The tokenizer is included. The server listens at `http://127.0.0.1:8000`;
press **Ctrl+C** to stop it. Use `--port` to choose another port.

## Smoke test

With the server running, open another terminal:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "muse-glimmer-30B",
    "messages": [
      {"role": "system", "content": "Reasoning strength: low"},
      {"role": "user", "content": "Say hello."}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

A successful response contains an assistant message in `choices[0].message.content`.

## Configuration

[default_config.toml](default_config.toml) documents all settings and defaults.
To override them, pass a TOML file explicitly:

```sh
muse-glimmer serve --config ./config.toml
```

Run `muse-glimmer serve --help` for command-line options.
