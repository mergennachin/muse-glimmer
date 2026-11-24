"""Install the checkout using pip's standard build backend."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def install(*, editable: bool = False) -> None:
    command = [sys.executable, "-m", "pip", "install"]
    if editable:
        command.append("--editable")
    subprocess.run([*command, "."], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editable", action="store_true", help="Install using pip install -e .")
    install(editable=parser.parse_args().editable)
