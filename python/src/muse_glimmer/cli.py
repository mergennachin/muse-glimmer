from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Settings, load_settings


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="muse-glimmer", description="Serve Muse Glimmer locally")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("pull", help="Download a model from Hugging Face")
    command.add_argument("repo_id", help="Hugging Face repository, e.g. owner/repo")
    command.add_argument(
        "--variant", help="Export directory within the repository ('.' for its root)"
    )
    command = commands.add_parser("serve", help="Start the local server")
    command.add_argument("--config", type=Path)
    command.add_argument("--model", help="Local .pte file or directory containing one .pte")
    command.add_argument("--launch", choices=["muse"], help="Launch Muse using the local server")
    command.add_argument("--host")
    command.add_argument("--port", type=int)
    command.add_argument("--context", type=int)
    command.add_argument("--concurrency", type=int)
    command.add_argument("--prefix-cache-entries", type=int)
    command.add_argument("--dflash", action=argparse.BooleanOptionalAction, default=None)
    command.add_argument("--dflash-n-draft", type=int)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "pull":
            from .download import pull

            model = pull(args.repo_id, args.variant)
            print(f"Default model: {model}\nRun muse-glimmer serve to start serving.")
            return 0

        values = {key: value for key, value in vars(args).items() if key in Settings.model_fields}
        settings = load_settings(values, args.config)
        if settings.model:
            model = Path(settings.model)
        else:
            from .download import default_model

            model = default_model()
        if model is None:
            raise ValueError("Run muse-glimmer pull HF/repo_id or pass --model /path/to/model.pte.")
        from .runtime import resolve_model, serve

        model = resolve_model(model)
        print(f"Model: {model.absolute()}", file=sys.stderr, flush=True)
        print(
            f"Config: {settings.model_dump_json(exclude={'model'}, indent=2)}",
            file=sys.stderr,
            flush=True,
        )
        if args.launch == "muse":
            from .launch import launch_muse

            return launch_muse(model, settings)
        serve(model, settings)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, ImportError) as error:
        print(f"muse-glimmer: {error}", file=sys.stderr)
        return 1
