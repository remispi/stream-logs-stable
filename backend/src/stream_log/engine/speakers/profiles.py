"""Streamer profiles: one folder per channel with the enrolled voice and (later) other settings."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from stream_log.engine.config import Paths
from stream_log.engine.project import read_json, slugify, write_json

ENROLL_SECONDS = 30.0


class ProfileError(RuntimeError):
    pass


def profile_dir(paths: Paths, name: str) -> Path:
    return paths.profiles / slugify(name)


def create_profile(paths: Paths, name: str, **fields) -> Path:
    directory = profile_dir(paths, name)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "profile.json"
    meta = read_json(meta_path) if meta_path.exists() else {"name": name}
    meta.update(fields)
    write_json(meta_path, meta)
    return directory


def load_profile(paths: Paths, name: str) -> dict:
    meta_path = profile_dir(paths, name) / "profile.json"
    if not meta_path.exists():
        raise ProfileError(f"no profile {name!r}: add the folder profiles/{name} to the data folder")
    return read_json(meta_path)


def list_profiles(paths: Paths) -> list[dict]:
    if not paths.profiles.exists():
        return []
    return [read_json(d / "profile.json") for d in sorted(paths.profiles.iterdir())
            if (d / "profile.json").exists()]


def load_voice(paths: Paths, name: str | None) -> np.ndarray | None:
    if not name:
        return None
    path = profile_dir(paths, name) / "voice.npy"
    return np.load(path) if path.exists() else None


def save_voice(paths: Paths, name: str, voice: np.ndarray, *, seconds: float, origin: str) -> None:
    directory = create_profile(paths, name)
    np.save(directory / "voice.npy", voice.astype(np.float32))
    create_profile(
        paths, name,
        voice={"seconds": round(seconds, 1), "origin": origin,
               "enrolled": datetime.now(UTC).isoformat(timespec="seconds")},
    )


def pick_enrollment_lines(
    lines: list[dict], embeddings: dict[int, np.ndarray], line_ids: list[int],
    max_seconds: float = ENROLL_SECONDS,
) -> tuple[np.ndarray, float]:
    """Most typical lines of a voice cluster (closest to its centroid), up to `max_seconds`."""
    by_id = {line["id"]: line for line in lines}
    members = [i for i in line_ids if i in embeddings]
    if not members:
        raise ProfileError("that voice has no lines long enough to enroll from")
    vectors = np.stack([embeddings[i] for i in members])
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    order = np.argsort(-(vectors @ centroid))
    chosen, seconds = [], 0.0
    for index in order:
        line = by_id[members[index]]
        chosen.append(vectors[index] * (line["end"] - line["start"]))
        seconds += line["end"] - line["start"]
        if seconds >= max_seconds:
            break
    voice = np.sum(chosen, axis=0)
    return voice / (np.linalg.norm(voice) + 1e-9), seconds
