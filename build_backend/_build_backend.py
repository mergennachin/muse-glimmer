"""Compile the ExecuTorch worker before setuptools assembles an installation."""

import runpy
from pathlib import Path

from setuptools import build_meta as _setuptools
from setuptools.build_meta import (  # noqa: F401
    build_sdist,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)


def _compile(*, editable=False):
    build = runpy.run_path(str(Path(__file__).resolve().parents[1] / "packaging/build_native.py"))
    build["main"](editable=editable)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _compile()
    return _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _compile(editable=True)
    return _setuptools.build_editable(wheel_directory, config_settings, metadata_directory)
