"""The video a project analyses: a file on this computer (Twitch downloads are in `twitch.py`)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_FINGERPRINT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    ref: str  # the file's full path

    @property
    def project_id(self) -> str:
        from stream_log.engine.project import slugify

        path = Path(self.ref)
        digest = hashlib.sha1(str(path.resolve()).lower().encode()).hexdigest()[:6]
        return f"{slugify(path.stem)}-{digest}"


def classify(ref: str) -> Source:
    path = Path(ref).expanduser()
    if path.is_file():
        return Source(str(path.resolve()))
    raise ValueError(f"no such video file: {ref}")


def fingerprint(path: Path) -> str:
    """Cheap content identity for multi-GB files: size + first and last 4 MiB."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_BYTES))
        if size > 2 * _FINGERPRINT_BYTES:
            handle.seek(-_FINGERPRINT_BYTES, 2)
            digest.update(handle.read(_FINGERPRINT_BYTES))
    return digest.hexdigest()[:16]
