"""Where each TFT game starts and ends, from the video alone.

One frame every 10 s of the project's proxy goes through SigLIP (the image model the analysis
already uses). A small classifier trained on hand-labelled stream frames (`screens.npz`, made by
`tools/train_screens.py`) says whether the frame shows a game in progress, a loading screen, the
post-game lobby or anything else. An HMM smooths that over time, and runs of "game" become games:
a short break with no lobby or loading screen in it (a crash, an alt-tab) doesn't end a game, and
runs shorter than MIN_GAME are dropped.

A game being watched (a tournament lobby in the spectator client, a broadcast) looks the same as a
game being played, so `mark_watched` decides from what the streamer says during it instead.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

STEP = 10  # seconds between frames (= the analysis's MODE_STEP, the proxy keyframe rate)
STATES = ("game", "loading", "lobby", "other")
GAME, LOADING, LOBBY, OTHER = range(4)
MIN_GAME = 8 * 60
# Longest break inside one game (a crash and reconnect took ~2 min in the labelled VODs) when no
# lobby or loading screen shows; the gap between two games is usually longer.
MAX_BREAK = 150
SEPARATOR_STEPS = 2  # this many lobby/loading frames in a break end the game
STAY = 0.9
CLASSIFIER = Path(__file__).with_name("screens.npz")
CACHE = "stream_log_games.json"
VISION_MODEL = "siglip-base"

# Someone playing talks about their own board ("I", "my"); someone watching talks about the player
# ("he", "his"). Share of "he" words among I-or-he words, per game: 0.01-0.17 in 18 games played
# (4 VODs), 0.32-0.41 in 8 games or stretches watched (2 VODs of the same tournament).
WATCHED_SHARE = 0.25
MIN_PRONOUNS = 30  # fewer than this in a game: too quiet to tell, it stays a game played
FIRST_PERSON = frozenset({"i", "i'm", "im", "i'll", "i've", "i'd", "my", "me", "mine", "myself"})
THIRD_PERSON = frozenset({"he", "he's", "hes", "he'll", "he'd", "his", "him", "himself", "she",
                          "she's", "her", "hers", "herself"})
_WORD = re.compile(r"[a-z']+")

Progress = Callable[[float, str], None]


def load_classifier(path: Path = CLASSIFIER) -> dict:
    data = np.load(path)
    return {"w": data["w"], "b": data["b"], "scale": float(data["scale"]),
            "version": str(data["version"])}


def screen_probs(embeddings: np.ndarray, clf: dict) -> np.ndarray:
    z = embeddings * clf["scale"] @ clf["w"] + clf["b"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def viterbi(probs: np.ndarray, stay: float = STAY) -> np.ndarray:
    steps, states = probs.shape
    if steps == 0:
        return np.zeros(0, dtype=int)
    move = (1 - stay) / (states - 1)
    log_trans = np.log(np.full((states, states), move) + np.eye(states) * (stay - move))
    log_emit = np.log(probs + 1e-6)
    score = np.log(np.full(states, 1 / states)) + log_emit[0]
    back = np.zeros((steps, states), dtype=int)
    for t in range(1, steps):
        candidates = score[:, None] + log_trans
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(states)] + log_emit[t]
    path = np.zeros(steps, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(path: np.ndarray, state: int) -> list[tuple[int, int]]:
    """[start, end) index runs where `path == state`."""
    out, start = [], None
    for i, value in enumerate(list(path) + [-1]):
        if value == state and start is None:
            start = i
        elif value != state and start is not None:
            out.append((start, i))
            start = None
    return out


def games_from_path(path: np.ndarray, step: int = STEP) -> list[dict]:
    runs = _runs(path, GAME)
    merged: list[list[int]] = []
    for start, end in runs:
        if merged:
            gap = path[merged[-1][1]:start]
            separated = max((len(r) for r in _split(gap)), default=0) >= SEPARATOR_STEPS
            if not separated and (start - merged[-1][1]) * step <= MAX_BREAK:
                merged[-1][1] = end
                continue
        merged.append([start, end])
    games = []
    for start, end in merged:
        if (end - start) * step < MIN_GAME:
            continue
        loading = None
        before = path[max(0, start - 18):start]  # a loading screen up to 3 min before
        hits = np.flatnonzero(before == LOADING)
        if len(hits):
            loading = (start - len(before) + int(hits[0])) * step
        games.append({"start": start * step, "end": end * step, "loading": loading})
    return games


def _split(gap: np.ndarray) -> list[np.ndarray]:
    """Runs of lobby/loading frames inside a break."""
    out, current = [], []
    for value in gap:
        if value in (LOADING, LOBBY):
            current.append(value)
        elif current:
            out.append(np.array(current))
            current = []
    if current:
        out.append(np.array(current))
    return out


def he_share(lines: list[dict], start: float, end: float) -> float | None:
    """Share of "he" words among I-or-he words said in [start, end), None if too few."""
    first = third = 0
    for line in lines:
        if start <= line["start"] < end:
            for word in _WORD.findall(line["text"].lower().replace("’", "'")):
                first += word in FIRST_PERSON
                third += word in THIRD_PERSON
    if first + third < MIN_PRONOUNS:
        return None
    return third / (first + third)


def mark_watched(found: list[dict], transcript: dict) -> list[dict]:
    """Each game with "watched": True when the streamer talks about the player, not themself.
    English only (the pronoun lists); other languages leave every game played."""
    english = (transcript.get("language") or "en").startswith("en")
    lines = transcript.get("lines") or []
    out = []
    for game in found:
        share = he_share(lines, game["start"], game["end"]) if english else None
        out.append({**game, "watched": share is not None and share > WATCHED_SHARE})
    return out


def frame_embeddings(models_root: Path, proxy: Path) -> np.ndarray:
    from stream_log.engine.models import load_manifest, require
    from stream_log.engine.onnx_models import Siglip
    from stream_log.engine.signals.visual import rgb_frames

    siglip = Siglip(require(load_manifest().get(VISION_MODEL), models_root))
    return siglip.embed_images(rgb_frames(proxy, step=STEP))


def detect(paths, project, progress: Progress | None = None,
           log: Callable[[str], None] = lambda _: None) -> list[dict]:
    """The project's games, cached in the project folder next to the analysis files."""
    clf = load_classifier()
    proxy = project.path("proxy.mp4")
    if not proxy.exists():
        log("no preview copy (proxy.mp4): games can't be found")
        return []
    key = f"{clf['version']}:{proxy.stat().st_size}:{int(proxy.stat().st_mtime)}"
    cache = project.path(CACHE)
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("key") == key:
            return stored["games"]
    if progress:
        progress(0.1, "reading frames")
    embeddings = frame_embeddings(paths.models, proxy)
    if progress:
        progress(0.8, "classifying screens")
    path = viterbi(screen_probs(embeddings, clf))
    games = games_from_path(path)
    states = "".join("GLHO"[s] for s in path)
    cache.write_text(json.dumps({"key": key, "step": STEP, "games": games, "states": states}),
                     encoding="utf-8")
    log(f"games: {len(games)} found")
    return games
