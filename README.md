# muse-glimmer

Serve Muse Glimmer locally. Development preview.

## Supported platforms

- macOS on Apple silicon

## Install and serve

You need Python 3.11+, Git, and Xcode with the Metal toolchain.
The first installation needs an internet connection. From this repository:

```sh
pip install -e .
muse-glimmer pull 1.0
muse-glimmer serve
```

`pull` selects the preferred export for your platform using [models.yaml](models.yaml),
which defines each release's default variant and maps variants and platforms to
Hugging Face repositories, pinned revisions, and export folders.

Select a variant explicitly with:

```sh
muse-glimmer pull 1.0 --variant text-solo
```

The backend is selected automatically. Unknown releases, unknown variants, and unsupported
platforms are reported before downloading.

Use `serve --model <path>` for a local model.

The server runs at `http://127.0.0.1:8000`. Press **Ctrl+C** to stop.

## Run Muse (coding agent) with the local model

With Muse installed, run this from your project directory:

```sh
muse-glimmer serve --launch muse
```

Muse uses the local model through its OpenRouter provider. No API key is needed.
Exiting Muse also stops the server.

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
