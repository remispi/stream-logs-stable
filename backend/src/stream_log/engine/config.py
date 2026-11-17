"""Where the app keeps models, projects and scratch files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "StreamLog"
HOME_ENV_VAR = "STREAM_LOG_HOME"


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def projects(self) -> Path:
        return self.home / "projects"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def profiles(self) -> Path:
        return self.home / "profiles"

    def ensure(self) -> Paths:
        for directory in (self.models, self.projects, self.cache, self.profiles):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    """Data home: $STREAM_LOG_HOME, else %LOCALAPPDATA%\\StreamLog."""
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Paths(Path(override).expanduser())
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return Paths(root / APP_NAME)
