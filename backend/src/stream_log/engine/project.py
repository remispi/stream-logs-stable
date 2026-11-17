"""A project is one VOD: a folder of cached stage artefacts plus `project.json` metadata.

Each pipeline stage records a key (hash of its version, settings and the upstream stage keys) in
`stages.json`. A stage re-runs only when its key changes or an output is missing, and because
upstream keys feed downstream keys, changing a setting re-runs everything after it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stream_log.engine.config import Paths

PROJECT_FILE = "project.json"
STAGES_FILE = "stages.json"
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts", ".m4v", ".avi")


class ProjectError(RuntimeError):
    pass


def _to_json(value: Any) -> Any:
    """numpy scalars/arrays leak out of the signal code; store them as plain numbers/lists."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Any, *, indent: int | None = 1) -> None:
    """Write atomically so an interrupted run never leaves half a file behind. The temporary
    file's name is unique, so two writers of one file never write into each other's."""
    text = json.dumps(data, indent=indent, ensure_ascii=False, default=_to_json)
    write_text(path, text)


def write_text(path: Path, text: str, *, newline: str | None = None) -> None:
    """`write_json` for text."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_text(text, encoding="utf-8", newline=newline)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache(path: Path | None) -> dict:
    """A cache file of LLM answers: empty when there is none yet, or when it can't be read (a
    copy cut short): those answers are asked again."""
    if path is None or not path.exists():
        return {}
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def stage_key(name: str, version: int, config: dict, upstream: list[str]) -> str:
    payload = json.dumps(
        {"stage": name, "version": version, "config": config, "upstream": upstream},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def id(self) -> str:
        return self.root.name

    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def meta(self) -> dict:
        return read_json(self.path(PROJECT_FILE))

    def update_meta(self, **changes: Any) -> dict:
        meta = self.meta
        meta.update(changes)
        write_json(self.path(PROJECT_FILE), meta)
        return meta

    @property
    def source_path(self) -> Path:
        """The VOD: the file it was ingested from, else a `source.<ext>` in the project folder
        (downloads land there)."""
        recorded = Path(self.meta["source_path"])
        if recorded.exists():
            return recorded
        return self.local_source() or recorded

    def local_source(self) -> Path | None:
        for suffix in VIDEO_SUFFIXES:
            if self.path("source" + suffix).exists():
                return self.path("source" + suffix)
        return None

    # ------------------------------------------------------------------ stage cache

    def _stages(self) -> dict:
        path = self.path(STAGES_FILE)
        return read_json(path) if path.exists() else {}

    def stage_record(self, name: str) -> dict | None:
        return self._stages().get(name)

    def is_fresh(self, name: str, key: str, outputs: list[str]) -> bool:
        record = self.stage_record(name)
        return (
            record is not None
            and record.get("key") == key
            and all(self.path(output).exists() for output in outputs)
        )

    def record_stage(self, name: str, key: str, outputs: list[str], seconds: float) -> None:
        stages = self._stages()
        stages[name] = {
            "key": key,
            "outputs": outputs,
            "seconds": round(seconds, 2),
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_json(self.path(STAGES_FILE), stages)


def slugify(text: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-") or "vod"


def create_project(paths: Paths, project_id: str, meta: dict) -> Project:
    root = paths.projects / project_id
    root.mkdir(parents=True, exist_ok=True)
    existing = read_json(root / PROJECT_FILE) if (root / PROJECT_FILE).exists() else {}
    merged = {
        "id": project_id,
        "created": existing.get("created") or datetime.now(UTC).isoformat(timespec="seconds"),
        **existing,
        **meta,
    }
    write_json(root / PROJECT_FILE, merged)
    return Project(root)


def open_project(paths: Paths, ref: str) -> Project:
    """Open by id, by unique id prefix, or by folder path."""
    as_path = Path(ref)
    if (as_path / PROJECT_FILE).exists():
        return Project(as_path)
    exact = paths.projects / ref
    if (exact / PROJECT_FILE).exists():
        return Project(exact)
    matches = [p for p in list_projects(paths) if p.id.startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ProjectError(f"no project {ref!r} in {paths.projects}")
    raise ProjectError(f"{ref!r} matches several projects: {', '.join(p.id for p in matches)}")


def list_projects(paths: Paths) -> list[Project]:
    if not paths.projects.exists():
        return []
    projects = [Project(d) for d in paths.projects.iterdir() if (d / PROJECT_FILE).exists()]
    return sorted(projects, key=lambda p: p.path(PROJECT_FILE).stat().st_mtime, reverse=True)


def clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
