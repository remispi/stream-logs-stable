"""Fetch model weights listed in the manifest into the models folder.

Each download runs in a child process, watched from here. A Hugging Face download can hang
without an error (no data moving, no exception, for hours), and a hung download can't be
stopped from inside its own process. The child reports every progress update of the download
library to a small heartbeat file; when nothing moves for `STALL_SECONDS`, the child is killed
and started again. Files that finished are kept, so each try resumes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from stream_log.engine.models.manifest import ModelEntry

STALL_SECONDS = 180  # no progress for this long: the download is stuck
POLL_SECONDS = 5.0
TRIES = 3

Progress = Callable[[str, int, int], None]  # (label, bytes done, bytes total or 0)

# The child: `fetch(heartbeat, repo, local_dir, *patterns)`.
_CHILD = "import sys; from stream_log.engine.models.download import fetch; fetch(*sys.argv[1:])"


class NotDownloadable(RuntimeError):
    pass


class DownloadFailed(RuntimeError):
    pass


def download(entry: ModelEntry, models_root: Path, *, say: Callable[[str], None] = print,
             progress: Progress | None = None, stall: float = STALL_SECONDS,
             poll: float = POLL_SECONDS, tries: int = TRIES) -> Path:
    """Download `entry` into `<models_root>/<id>` (resumes partial downloads)."""
    if not entry.source.startswith("hf:"):
        raise NotDownloadable(
            f"{entry.id}: source {entry.source or '(not chosen)'!r} is not a Hugging Face repo"
        )
    target = entry.local_dir(models_root)
    handle, name = tempfile.mkstemp(prefix="stream-log-download-", suffix=".txt")
    os.close(handle)
    heartbeat = Path(name)
    command = [sys.executable, "-s", "-c", _CHILD, str(heartbeat),
               entry.source.removeprefix("hf:"), str(target), *entry.files]
    report = (lambda done, total: progress(entry.id, done, total)) if progress else None
    try:
        for attempt in range(1, tries + 1):
            code = run_watched(command, heartbeat, stall=stall, poll=poll, progress=report)
            if code == 0:
                return target
            if code is not None:
                raise DownloadFailed(f"{entry.id}: the download failed (exit code {code}), "
                                     "see above")
            if attempt < tries:
                wait = f"{stall / 60:g} min" if stall >= 60 else f"{stall:g} s"
                say(f"{entry.id}: no data for {wait}, the download is stuck; "
                    f"starting it again (try {attempt + 1} of {tries})")
    finally:
        heartbeat.unlink(missing_ok=True)
    raise DownloadFailed(f"{entry.id}: the download got stuck {tries} times. Check the internet "
                         "connection and run it again: what finished is kept.")


def kill_tree(pid: int) -> None:
    if os.name == "nt":  # the child's own children too
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(pid, 9)
        except OSError:
            pass


def run_watched(command: list[str], heartbeat: Path, *, stall: float = STALL_SECONDS,
                poll: float = POLL_SECONDS,
                progress: Callable[[int, int], None] | None = None) -> int | None:
    """Run `command` until it exits (its exit code) or `heartbeat` stops changing for `stall`
    seconds (then it is killed: None)."""
    heartbeat.write_text("", encoding="utf-8")
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    child = subprocess.Popen(command, env=env)
    seen, since = "", time.monotonic()
    try:
        while True:
            try:
                code = child.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                code = None
            beat = _read(heartbeat, seen)
            if code is not None:
                if code == 0 and progress and (numbers := _numbers(beat)) and numbers[2]:
                    progress(numbers[2], numbers[2])  # done: the bar ends full
                return code
            if beat != seen:
                seen, since = beat, time.monotonic()
                if progress and (numbers := _numbers(beat)):
                    progress(numbers[1], numbers[2])
            elif time.monotonic() - since > stall:
                kill_tree(child.pid)
                child.wait()
                return None
    finally:
        if child.poll() is None:  # interrupted (Ctrl+C): don't leave it running
            kill_tree(child.pid)


def _read(heartbeat: Path, default: str) -> str:
    try:
        return heartbeat.read_text(encoding="utf-8")
    except OSError:
        return default


def _numbers(beat: str) -> tuple[int, int, int] | None:
    parts = beat.split()
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


# The two byte bars of `snapshot_download`: received from the network (xet), written to disk.
_BARS = {"huggingface_hub.snapshot_download.transfer": "received",
         "huggingface_hub.snapshot_download": "written"}


def fetch(heartbeat: str, repo: str, local_dir: str, *patterns: str) -> None:
    """In the child: download, writing "<updates> <bytes done> <bytes expected>" to `heartbeat`
    as data moves (at most once a second)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import tqdm as hf_tqdm

    lock = threading.Lock()
    state = {"moved": 0, "received": 0, "written": 0, "total": 0, "at": 0.0}
    path = Path(heartbeat)

    def beat(force: bool = False) -> None:
        now = time.monotonic()
        if force or now - state["at"] >= 1.0:
            state["at"] = now
            done = max(state["received"], state["written"])
            path.write_text(f"{state['moved']} {done} {max(state['total'], done)}",
                            encoding="utf-8")

    class Beat(hf_tqdm):
        """The download's progress bars (shown or not): every update is a sign of life."""

        def __init__(self, *args, **kwargs):
            self.kind = _BARS.get(kwargs.get("name") or "")
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1):
            with lock:
                state["moved"] += int(n or 0)
                if self.kind:  # both bars' total: the sum of the files' sizes
                    state[self.kind] += int(n or 0)
                    state["total"] = max(state["total"], int(self.total or 0))
                beat()
            return super().update(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        snapshot_download(repo, local_dir=local_dir, allow_patterns=list(patterns) or None,
                          tqdm_class=Beat)
    with lock:
        beat(force=True)


def require(entry: ModelEntry, models_root: Path) -> Path:
    """Return the model folder, or explain how to download it."""
    if not entry.is_downloaded(models_root):
        raise FileNotFoundError(
            f"model {entry.id!r} is not downloaded: run `stream-log setup`"
        )
    return entry.local_dir(models_root)
