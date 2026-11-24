from __future__ import annotations

import platform
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .resources import resource_path


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    host: str
    port: int = Field(ge=1, le=65535)
    context: int = Field(gt=1)
    concurrency: int = Field(ge=1, le=64)
    prefix_cache_entries: int = Field(ge=0)
    prefix_checkpoint_prompt_end: Literal["initial", "every"]
    dflash: bool
    dflash_n_draft: int = Field(ge=0)


def _read_config(path: Path) -> dict:
    with path.open("rb") as file:
        values = tomllib.load(file)
    groups = values.pop("backends", {})
    if not isinstance(groups, dict):
        raise ValueError("Expected [backends.metal] or [backends.cuda] tables")
    for name, section in groups.items():
        if name not in {"metal", "cuda"} or not isinstance(section, dict):
            raise ValueError(f"Invalid backend config {name!r}. Use [backends.metal] or [backends.cuda].")
        unknown = section.keys() - Settings.model_fields.keys()
        if unknown:
            raise ValueError(f"Unknown setting in [backends.{name}]: {', '.join(sorted(unknown))}")
    backend = "cuda" if platform.system() == "Linux" else "metal"
    values.update(groups.get(backend, {}))
    return values


def load_defaults() -> dict:
    return _read_config(resource_path("default_config.toml"))


def load_settings(
    overrides: dict, config_path: Path | None = None, *, model_defaults: dict | None = None
) -> Settings:
    values = load_defaults()
    values.update(model_defaults or {})
    if config_path is not None:
        if not config_path.is_file():
            raise ValueError(f"Configuration file does not exist: {config_path}")
        values.update(_read_config(config_path))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Settings.model_validate(values)
