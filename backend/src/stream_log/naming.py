"""File names for downloaded VODs: TwitchStream<YYYYMMDD>-<TitleInPascalCase>.

The date is the day the stream started, in this computer's time zone. The title part keeps every
word's letters and digits, capitalises each word's first letter and leaves the rest as typed (so
"BBBC day 12" becomes "BBBCDay12" and "POV" stays "POV"); chat commands like "!scores" are dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

PREFIX = "TwitchStream"
MAX_TITLE = 60
_WORD = re.compile(r"[^\W_]+")  # runs of letters and digits, any script
_APOSTROPHES = str.maketrans("", "", "'’`")


def title_part(title: str, limit: int = MAX_TITLE) -> str:
    out = ""
    for raw in title.translate(_APOSTROPHES).split():
        if raw.startswith("!"):
            continue
        for word in _WORD.findall(raw):
            word = word[0].upper() + word[1:]
            if len(out) + len(word) > limit:
                return out or word[:limit]
            out += word
    return out


def file_stem(started: datetime, title: str) -> str:
    """`started` is aware (UTC from Twitch) or naive local time."""
    local = started.astimezone() if started.tzinfo else started
    part = title_part(title)
    return f"{PREFIX}{local:%Y%m%d}" + (f"-{part}" if part else "")


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


def bad_stem(name: str) -> str | None:
    """Why `name` can't be a Windows file name (without .mp4), or None when it can. Also stops a
    name from pointing outside the download folder (no separators, no drive, no "..")."""
    if not name.strip():
        return "the file name is empty"
    if match := _BAD_CHARS.search(name):
        return f"a file name can't contain {match.group()!r}" if match.group().isprintable() \
            else "a file name can't contain control characters"
    if name != name.rstrip(" ."):
        return "a file name can't end with a space or a dot"
    if name.split(".")[0].strip().lower() in _RESERVED:
        return f"{name!r} is reserved by Windows"
    if len(name) > 200:
        return "the file name is too long"
    return None


def free_path(folder: Path, stem: str, suffix: str = ".mp4",
              reuse: Callable[[Path], bool] | None = None) -> Path:
    """`folder/stem.mp4`, or `stem-2.mp4`, `stem-3.mp4`... when that name is taken. With `reuse`,
    the first taken name it accepts is returned instead (the same video, downloaded before)."""
    path = folder / f"{stem}{suffix}"
    n = 2
    while path.exists():
        if reuse is not None and reuse(path):
            return path
        path = folder / f"{stem}-{n}{suffix}"
        n += 1
    return path


_DATE = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def date_in_name(name: str) -> str | None:
    """'TwitchStream20261114-Title.mp4' -> '2026-11-14' (to match a file with its Twitch VOD)."""
    match = _DATE.search(name)
    if not match:
        return None
    year, month, day = match.groups()
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return None
    return f"{year}-{month}-{day}"
