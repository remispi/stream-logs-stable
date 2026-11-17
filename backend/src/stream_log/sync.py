"""Line up two VODs that share audio (a costream: the same voice chat or broadcast in both).

The two streams' start times give a first guess of the offset. Then, every STEP seconds, a WINDOW
of A's loudness pattern (20 frames a second, level changes only, so different mic volumes don't
matter) is matched against B around that guess. Windows where the match stands out clearly from
every other lag are kept; neighbouring windows that agree on the lag become one piece. A piece is
a stretch the two streams were together, with B's time = A's time + lag. Outside the pieces the
streams had nothing in common, so no time is given there.

The result is saved in both projects (`stream_log_sync.json`), and each list then shows the other
VOD's time next to every moment inside a piece.
"""

from __future__ import annotations

import itertools
import json
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

FPS = 20
WINDOW = 240
STEP = 120
SEARCH = 240  # seconds either side of the guess
MIN_PEAK = 0.25  # correlation at the best lag
MIN_MARGIN = 0.08  # ... above the best lag more than a second away
SAME_LAG = 0.6  # seconds: windows this close in lag are one piece
SYNC_FILE = "stream_log_sync.json"
VIDEOS_FILE = "stream_log_videos.json"


def envelope(wav_path: Path, fps: int = FPS) -> np.ndarray:
    """Log loudness at `fps` frames a second with its 1-second moving average taken off."""
    with wave.open(str(wav_path)) as wav:
        rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
        if width != 2:
            raise ValueError(f"{wav_path}: expected 16-bit audio")
        hop = rate // fps
        out = []
        chunk = hop * fps * 60
        while True:
            raw = wav.readframes(chunk)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if channels > 1:
                x = x.reshape(-1, channels).mean(axis=1)
            usable = len(x) // hop * hop
            frames = x[:usable].reshape(-1, hop)
            out.append(np.log10((frames ** 2).mean(axis=1) + 1.0))
    env = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    kernel = np.ones(fps) / fps
    return (env - np.convolve(env, kernel, mode="same")).astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of `a` with every same-length slice of `b` (len(b) >= len(a))."""
    n = len(a)
    a = (a - a.mean()) / (a.std() + 1e-9)
    size = 1 << int(np.ceil(np.log2(len(b) + n)))
    dot = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)[: len(b) - n + 1]
    c1 = np.concatenate([[0.0], np.cumsum(b, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(b.astype(np.float64) ** 2)])
    sums = c1[n:] - c1[:-n]
    squares = c2[n:] - c2[:-n]
    std = np.sqrt(np.maximum(squares / n - (sums / n) ** 2, 1e-12))
    return dot / (n * std)


def windows(env_a: np.ndarray, env_b: np.ndarray, guess: float,
            progress: Callable[[float], None] | None = None) -> list[dict]:
    out = []
    length_a = len(env_a) / FPS
    starts = np.arange(0, max(0.0, length_a - WINDOW), STEP)
    for i, start in enumerate(starts):
        a = env_a[int(start * FPS): int((start + WINDOW) * FPS)]
        lo = max(0.0, start + guess - SEARCH)
        hi = min(len(env_b) / FPS, start + guess + WINDOW + SEARCH)
        if hi - lo < WINDOW + 10 or a.std() < 1e-3:
            continue
        r = ncc(a, env_b[int(lo * FPS): int(hi * FPS)])
        k = int(np.argmax(r))
        away = np.abs(np.arange(len(r)) - k) > FPS
        second = float(r[away].max()) if away.any() else 0.0
        lag = lo + k / FPS - start
        out.append({"t": float(start), "lag": round(float(lag), 2), "r": round(float(r[k]), 3),
                    "margin": round(float(r[k]) - second, 3)})
        if progress:
            progress((i + 1) / len(starts))
    return out


def pieces(found: list[dict], env_a: np.ndarray | None = None,
           env_b: np.ndarray | None = None) -> list[dict]:
    """Runs of confident windows that agree on the lag: {a_start, a_end, lag}. With the
    envelopes, the point where the lag jumps between two pieces is found to within ~10 s."""
    good = [w for w in found if w["r"] >= MIN_PEAK and w["margin"] >= MIN_MARGIN]
    runs: list[list[dict]] = []
    for w in good:
        last = runs[-1] if runs else None
        if last and abs(w["lag"] - last[-1]["lag"]) <= SAME_LAG \
                and w["t"] - last[-1]["t"] <= WINDOW + 2 * STEP:
            last.append(w)
        else:
            runs.append([w])
    runs = [run for run in runs if len(run) >= 2]  # one window alone can be chance
    result = [{"a_start": run[0]["t"], "a_end": run[-1]["t"] + WINDOW,
               "lag": round(float(np.median([w["lag"] for w in run])), 2),
               "windows": len(run)} for run in runs]
    for first, second in itertools.pairwise(result):
        if first["a_end"] < second["a_start"] - STEP:
            continue  # apart for a while: no shared audio in between
        lo, hi = second["a_start"] - STEP, first["a_end"] + STEP
        missing = first["lag"] - second["lag"]
        have_audio = env_a is not None and env_b is not None
        if missing > 0:
            # B's offset drops: B's VOD lacks `missing` seconds (its stream dropped), so A's
            # [end of the match, + missing) has nothing in B.
            cut = end_of_match(env_a, env_b, first["lag"], lo, hi) if have_audio else (lo + hi) / 2
            first["a_end"], second["a_start"] = round(cut, 1), round(cut + missing, 1)
        else:
            # A's VOD lacks time: A jumps from one offset to the other at one point.
            cut = jump_point(env_a, env_b, first["lag"], second["lag"], lo, hi) if have_audio \
                else (lo + hi) / 2
            first["a_end"] = second["a_start"] = round(cut, 1)
    return result


def _score(env_a: np.ndarray, env_b: np.ndarray, t: float, lag: float, size: float) -> float:
    a = env_a[int(t * FPS): int((t + size) * FPS)]
    start = int((t + lag) * FPS)
    b = env_b[max(0, start): start + len(a)]
    if len(a) < size * FPS * 0.9 or len(b) != len(a) or a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def end_of_match(env_a: np.ndarray, env_b: np.ndarray, lag: float, lo: float, hi: float,
                 size: float = 30.0, step: float = 5.0, floor: float = 0.15) -> float:
    """The last A time in [lo, hi] that still matches B at `lag`."""
    times = np.arange(lo, hi - size, step)
    if len(times) == 0:
        return (lo + hi) / 2
    gain = np.cumsum([_score(env_a, env_b, t, lag, size) - floor for t in times])
    k = int(np.argmax(gain))
    return float(times[k] + size)


def jump_point(env_a: np.ndarray, env_b: np.ndarray, lag1: float, lag2: float,
               lo: float, hi: float, size: float = 30.0, step: float = 5.0) -> float:
    """Where B's offset changes from lag1 to lag2 between A's times lo and hi: the split that
    agrees best with short windows scored at either lag."""
    times = np.arange(lo, hi - size, step)
    if len(times) == 0:
        return (lo + hi) / 2
    diff = np.array([_score(env_a, env_b, t, lag1, size) - _score(env_a, env_b, t, lag2, size)
                     for t in times])  # > 0: still lag1
    best, best_k = -np.inf, 0
    for k in range(len(times) + 1):
        value = diff[:k].sum() - diff[k:].sum()
        if value > best:
            best, best_k = value, k
    return float(times[best_k] + size / 2) if best_k < len(times) else float(times[-1] + size)


def mapper(sync_pieces: list[dict]) -> Callable[[float], float | None]:
    def to_other(t: float) -> float | None:
        for piece in sync_pieces:
            if piece["start"] <= t <= piece["end"]:
                return t + piece["lag"]
        return None
    return to_other


def _clock(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(round(seconds))
    return f"{sign}{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def describe(other_label: str, sync_pieces: list[dict]) -> str:
    if not sync_pieces:
        return f"No shared audio found with {other_label}'s VOD."
    first, last = sync_pieces[0], sync_pieces[-1]
    text = (f"Synced with {other_label}'s VOD from the audio you share: their time is shown next "
            f"to each moment from {_clock(first['start'])} to {_clock(last['end'])} here "
            f"(their {_clock(first['start'] + first['lag'])} to {_clock(last['end'] + last['lag'])})"
            f"; at {_clock(first['start'])} their time = this time "
            f"{'+' if first['lag'] >= 0 else '−'} {_clock(abs(first['lag']))}.")
    for p, q in itertools.pairwise(sync_pieces):
        if q["start"] - p["end"] >= 1:
            text += (f" Their VOD has nothing for {_clock(p['end'])}–{_clock(q['start'])} here"
                     " (their stream dropped or you were apart).")
        their_gap = (q["start"] + q["lag"]) - (p["end"] + p["lag"])
        if their_gap >= 1:
            text += (f" This VOD has nothing for their {_clock(p['end'] + p['lag'])}–"
                     f"{_clock(q['start'] + q['lag'])} (this stream dropped).")
    return text


# ---------------------------------------------------------------------- projects


def _project_for(paths, ref: str):
    from stream_log.engine.project import list_projects, open_project

    path = Path(ref)
    if path.suffix:
        target = str(path.resolve()).lower()
        for project in list_projects(paths):
            videos = [project.meta.get("source_path", "")] + _list_videos(project)
            if target in (str(v).lower() for v in videos):
                return project
        raise FileNotFoundError(f"no processed project for {path.name}: make its list first")
    return open_project(paths, ref)


def _videos_record(project) -> dict:
    path = project.path(VIDEOS_FILE)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _list_videos(project) -> list[str]:
    return _videos_record(project).get("videos", [])


def remember_video(project, video: Path, list_dir: Path | None = None) -> None:
    """Note which video a list was made for (a reused project's own copy has another path) and,
    with `list_dir`, the folder its list went to, so writing it again (`relist`) goes there too."""
    stored = _videos_record(project)
    videos = stored.setdefault("videos", [])
    target = str(Path(video).resolve())
    changed = target not in videos
    if changed:
        videos.append(target)
    if list_dir is not None:
        folders = stored.setdefault("list_dirs", {})
        folder = str(Path(list_dir).resolve())
        if folders.get(target) != folder:
            folders[target] = folder
            changed = True
    if changed:
        project.path(VIDEOS_FILE).write_text(json.dumps(stored, indent=1), encoding="utf-8")


def list_dir_for(project, video: Path) -> Path:
    """Where `video`'s list was written last time (default: next to the video)."""
    folder = _videos_record(project).get("list_dirs", {}).get(str(Path(video).resolve()))
    return Path(folder) if folder else Path(video).parent


def label_of(project) -> str:
    return project.meta.get("channel") or project.id


def run(a_ref: str, b_ref: str, label_a: str | None, label_b: str | None,
        emit: Callable[[dict], None], per_hour: int | None = None, *,
        find_games: bool = True) -> dict:
    """Sync two processed VODs, save it in both projects and write both lists again."""
    result = compute(a_ref, b_ref, label_a, label_b, emit)
    from stream_log.engine.config import get_paths
    from stream_log.engine.project import open_project
    from stream_log.listing import PER_HOUR
    from stream_log.run import relist

    paths = get_paths()
    for project_id in (result["a"], result["b"]):
        videos = _list_videos(open_project(paths, project_id))
        if videos and Path(videos[-1]).exists():
            relist(project_id, Path(videos[-1]), per_hour or PER_HOUR, emit,
                   find_games=find_games)
    emit(result)
    return result


def compute(a_ref: str, b_ref: str, label_a: str | None, label_b: str | None,
            emit: Callable[[dict], None]) -> dict:
    from stream_log.engine.config import get_paths

    paths = get_paths()
    pa, pb = _project_for(paths, a_ref), _project_for(paths, b_ref)
    label_a, label_b = label_a or label_of(pa), label_b or label_of(pb)
    sa, sb = pa.meta.get("started_at"), pb.meta.get("started_at")
    if not (sa and sb):  # checked before reading hours of audio
        raise ValueError("both VODs need their stream start time (run them with their Twitch link)")
    guess = float(sa - sb)

    def step(fraction: float, text: str) -> None:
        emit({"type": "step", "id": "sync", "fraction": fraction, "text": text})

    step(0.0, f"reading {label_a}'s audio")
    env_a = envelope(pa.path("audio.wav"))
    step(0.2, f"reading {label_b}'s audio")
    env_b = envelope(pb.path("audio.wav"))
    found = windows(env_a, env_b, guess, lambda f: step(0.4 + 0.6 * f, "matching"))
    parts = pieces(found, env_a, env_b)
    a_side = [{"start": p["a_start"], "end": p["a_end"], "lag": p["lag"]} for p in parts]
    b_side = [{"start": p["a_start"] + p["lag"], "end": p["a_end"] + p["lag"], "lag": -p["lag"]}
              for p in parts]
    save(pa, pb, label_b, a_side)
    save(pb, pa, label_a, b_side)
    return {"type": "sync", "a": pa.id, "b": pb.id, "guess": guess, "pieces": parts,
            "note_a": describe(label_b, a_side), "note_b": describe(label_a, b_side),
            "windows": found}


def save(project, other, other_label: str, sync_pieces: list[dict]) -> None:
    path = project.path(SYNC_FILE)
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"others": []}
    stored["others"] = [o for o in stored["others"] if o["project"] != other.id]
    stored["others"].append({"project": other.id, "label": other_label,
                             "url": other.meta.get("url"), "pieces": sync_pieces})
    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


def others(project) -> list[dict]:
    """The synced VODs for a list: label, url, note and a time mapper."""
    path = project.path(SYNC_FILE)
    if not path.exists():
        return []
    out = []
    for other in json.loads(path.read_text(encoding="utf-8"))["others"]:
        if not other["pieces"]:
            continue
        out.append({"label": other["label"], "url": other.get("url"),
                    "note": describe(other["label"], other["pieces"]),
                    "map": mapper(other["pieces"])})
    return out
