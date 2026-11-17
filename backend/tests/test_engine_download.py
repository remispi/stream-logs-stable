import importlib
import subprocess
import sys
import time

import psutil
import pytest

from stream_log.engine.models import load_manifest

# The module (the package's `download` is the function).
dl = importlib.import_module("stream_log.engine.models.download")
ENTRY = load_manifest().get("bge-small-en")


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_watched_child_that_keeps_moving_runs_to_the_end(tmp_path):
    heartbeat = tmp_path / "beat.txt"
    beats = (f"import time, pathlib\np = pathlib.Path(r'{heartbeat}')\n"
             "for i in range(1, 9):\n"
             "    p.write_text(f'{i} {i * 100} 800'); time.sleep(0.15)\n")
    seen = []
    code = dl.run_watched(_child(beats), heartbeat, stall=1.0, poll=0.1,
                          progress=lambda done, total: seen.append((done, total)))
    assert code == 0
    assert seen and seen[-1] == (800, 800)  # the last call fills the bar
    assert all(total == 800 for _, total in seen)


def test_watched_child_that_hangs_is_killed(tmp_path):
    started = time.monotonic()
    pids = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        pids.append(process.pid)
        return process

    code = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dl.subprocess, "Popen", popen)
        code = dl.run_watched(_child("import time; time.sleep(60)"), tmp_path / "beat.txt",
                              stall=1.0, poll=0.1)
    assert code is None
    assert time.monotonic() - started < 20
    assert not psutil.pid_exists(pids[0]) or psutil.Process(pids[0]).status() == "zombie"


def test_failed_child_returns_its_exit_code(tmp_path):
    code = dl.run_watched(_child("raise SystemExit(3)"), tmp_path / "beat.txt", stall=5, poll=0.1)
    assert code == 3


def test_download_retries_a_stuck_download_then_gives_up(tmp_path, monkeypatch):
    calls, said = [], []
    monkeypatch.setattr(dl, "run_watched", lambda command, *a, **k: calls.append(command))
    with pytest.raises(dl.DownloadFailed, match="stuck 3 times"):
        dl.download(ENTRY, tmp_path, say=said.append, stall=60)
    assert len(calls) == 3
    assert [s.split(":")[0] for s in said] == ["bge-small-en", "bge-small-en"]
    assert "try 2 of 3" in said[0] and "try 3 of 3" in said[1]
    # the child downloads that repo into the model's folder, only the listed files
    assert calls[0][-len(ENTRY.files) - 2:] == ["BAAI/bge-small-en-v1.5",
                                                 str(ENTRY.local_dir(tmp_path)), *ENTRY.files]


def test_download_recovers_after_a_stuck_try(tmp_path, monkeypatch):
    results = iter([None, 0])
    monkeypatch.setattr(dl, "run_watched", lambda *a, **k: next(results))
    said = []
    assert dl.download(ENTRY, tmp_path, say=said.append) == ENTRY.local_dir(tmp_path)
    assert len(said) == 1


def test_download_reports_a_failure_without_retrying(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "run_watched", lambda *a, **k: calls.append(1) or 2)
    with pytest.raises(dl.DownloadFailed, match="exit code 2"):
        dl.download(ENTRY, tmp_path, say=lambda _: None)
    assert len(calls) == 1


def test_fetch_beats_with_the_progress_bars_hidden(tmp_path, monkeypatch):
    """The library's own bars (made by its factory, hidden as in the app) drive the heartbeat."""
    import huggingface_hub
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
    from huggingface_hub.utils.tqdm import _create_progress_bar

    heartbeat = tmp_path / "beat.txt"
    seen = []

    def fake_snapshot(repo, *, local_dir, allow_patterns, tqdm_class):
        assert (repo, local_dir, allow_patterns) == ("org/repo", "here", ["a.gguf"])
        transfer = _create_progress_bar(cls=tqdm_class, log_level=20, total=0,
                                        name="huggingface_hub.snapshot_download.transfer")
        written = _create_progress_bar(cls=tqdm_class, log_level=20, total=0,
                                       name="huggingface_hub.snapshot_download")
        assert transfer.disable and written.disable
        transfer.total = written.total = 1000  # as a file registers its size
        transfer.update(300)
        seen.append(heartbeat.read_text())
        time.sleep(1.05)
        transfer.update(300)
        seen.append(heartbeat.read_text())
        written.update(1000)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    disable_progress_bars()  # as HF_HUB_DISABLE_PROGRESS_BARS=1 does in the child
    try:
        dl.fetch(str(heartbeat), "org/repo", "here", "a.gguf")
    finally:
        enable_progress_bars()
    assert seen == ["300 300 1000", "600 600 1000"]
    assert heartbeat.read_text() == "1600 1000 1000"
