# muse-glimmer

Serve [Muse Glimmer 30B](https://ai.meta.com/research/publications/muse-glimmer-30b-an-on-device-agentic-model/) locally. Development preview.

**Supported platforms:** macOS 26.2+ on Apple silicon and Linux x86_64 with CUDA 13.2. Requires Python 3.11+.
The CUDA model was tested on an 80 GB A100 and exceeds Ada GPUs’ shared-memory limit.

## Quickstart

Until the CUDA wheel is published on PyPI, replace the first command on Linux with
`pip install -e .` from this checkout or `pip install ./muse_glimmer-*-linux_x86_64.whl` for a downloaded wheel.

```sh
pip install --pre muse-glimmer
muse-glimmer pull 1.0
muse-glimmer serve
```

Serves at `http://127.0.0.1:8000` with a live terminal HUD. **Ctrl+C** stops it.

<details>
<summary>Create a Python environment (Conda or uv)</summary>

With Conda:

```sh
conda create -yn glimmer python=3.12
conda activate glimmer
```

Or [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
```

</details>

<details>
<summary>Install from source</summary>

Linux builds require Git, Make, the CUDA 13.2 toolkit, and a C++ toolchain with a GCC 12+ runtime.
On macOS, set up Xcode with the Metal toolchain:

```sh
sudo xcodebuild -license accept
xcodebuild -downloadComponent MetalToolchain
```

```sh
git clone --recursive https://github.com/mergennachin/muse-glimmer.git
cd muse-glimmer
pip install -e .
```

</details>

## Coding agents

Install [Muse](https://dev.meta.ai/products/muse-code), [Pi](https://github.com/earendil-works/pi),
[Oh My Pi](https://omp.sh), [Hermes Agent](https://hermes-agent.nousresearch.com/),
or [OpenCode](https://opencode.ai), then launch from your project directory:

```sh
muse-glimmer serve --launch muse
```

Replace `muse` with `pi`, `omp`, `hermes`, or `opencode` as needed.
No API key is needed. Exiting the agent stops the server.

<img src="img/muse-code-demo.png" alt="Muse coding agent building a Pomodoro timer with Muse Glimmer" width="600">

## Chat UI

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, set up and launch [Open WebUI](https://openwebui.com/):

```sh
uv tool install --python 3.12 open-webui==0.13.3
muse-glimmer serve --launch open-webui
```

Your browser opens at `http://127.0.0.1:8080`. Chats are saved locally. **Ctrl+C** stops both servers.

<img src="img/open-webui-demo.png" alt="Chatting with Muse Glimmer in Open WebUI" width="600">

## Smoke test

With the server running, open another terminal:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "muse-glimmer-30B",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

## Options

For model selection, downloads, and server options:

```sh
muse-glimmer pull --help
muse-glimmer serve --help
```

[default_config.toml](default_config.toml) documents the configuration settings.
