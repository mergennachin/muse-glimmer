from __future__ import annotations

import tomllib
from pathlib import Path

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
    dflash: bool
    dflash_n_draft: int = Field(ge=0)


def load_settings(
    overrides: dict, config_path: Path | None = None, *, model_defaults: dict | None = None
) -> Settings:
    values = tomllib.loads(resource_path("default_config.toml").read_text())
    values.update(model_defaults or {})
    if config_path is not None:
        if not config_path.is_file():
            raise ValueError(f"Configuration file does not exist: {config_path}")
        values.update(tomllib.loads(config_path.read_text()))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Settings.model_validate(values)
