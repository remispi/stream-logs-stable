"""The stream list: game and topic blocks as headers, then one short line per moment, in order.

Built from a project analysed up to its `events` stage, plus the games found in the
video (`games.py`). Each line is a cluster of nearby evidence (within MERGE seconds): something the
LLM described from the transcript (a story, an opinion, an announcement, a play), a reaction heard
in the audio (laugh, yell, cheer, gasp), a chat spike, a raid or sub. The line's text is the LLM's
title when there is one, else the words said at that moment; its tags say why it was flagged.

To keep the list readable at about a minute per hour of stream, each block keeps at most
PER_HOUR lines per hour of its length (the strongest ones); the rest are folded into "+N smaller".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stream_log.games import mark_watched

PER_HOUR = 14
MERGE = 30.0  # seconds: evidence this close together is one moment
MIN_BLOCK = 5 * 60  # a stretch between games shorter than this joins the game before it
MAX_GAME = 60 * 60  # longer "games" aren't one game (see is_game)
BLOCK_TARGET = 25 * 60  # outside games, blocks of about this length along the chapters
MAX_TEXT = 72

IMPORTANCE = {"key": 3.0, "notable": 2.0, "minor": 1.0}
CATEGORY_BONUS = {"conversation": 0.6, "announcement": 0.6, "community": 0.3, "health": 0.0,
                  "chat": 0.0, "gameplay": 0.0}
CATEGORY_TAG = {"conversation": "topic", "announcement": "topic", "gameplay": "play",
                "chat": "chat", "community": "community", "health": "stream"}
SKIP_TYPES = {"loading", "start", "ending", "final statement", "starting_soon", "brb",
              "stream_end", "frozen_video"}
# Reactions heard in the audio: the analysis's sound-event score (0-1 per second), tag, and the
# score a peak needs. On real VODs these scores sit near 0 all stream (cheer and gasp mostly pick
# up game sounds, so they aren't used); laughter above 0.05 came about once an hour, each time a
# real laugh.
REACTIONS = [("laughter", "laugh", 0.05), ("scream", "yell", 0.05)]
LOUD_PERCENTILE = 99.6  # The analysis's voice energy ("arousal") in its top 0.4 %: "loud"
LOUD_PER_HOUR = 3
CHAT_RATIO = 3.0  # chat this many times busier than usual around it
CHAT_MIN_RATE = 0.2  # messages per second at the spike, so a 3-message burst isn't "×5"
TAG_ORDER = ["topic", "play", "stream", "laugh", "yell", "loud", "chat", "raid", "subs",
             "bits", "donation", "community"]


@dataclass
class Evidence:
    t: float
    weight: float
    tags: list[str]
    text: str = ""
    text_rank: int = 0  # 3 = LLM title, 2 = community notice, 1 = quote


@dataclass
class Line:
    t: float
    text: str
    tags: list[str]
    score: float


@dataclass
class Block:
    start: float
    end: float
    title: str
    kind: str  # game | stretch
    lines: list[Line] = field(default_factory=list)
    folded: list[Line] = field(default_factory=list)
    game_start: float | None = None  # games: when play starts (the block may start earlier)
    game_end: float | None = None


@dataclass
class StreamList:
    name: str
    title: str
    channel: str
    date: str
    duration: float
    url: str | None
    video: str
    summary: str
    games: int  # games played (watched ones aren't counted)
    game_seconds: float
    blocks: list[Block]
    per_hour: int
    others: list[dict] = field(default_factory=list)  # synced VODs: {name, url, offsets}
    watched: int = 0  # games watched (spectated, a broadcast)


# ---------------------------------------------------------------------- helpers


def clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def minutes(seconds: float) -> str:
    m = round(seconds / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


_LEADING = re.compile(r"^(the\s+)?(streamer|streamers|remi)('s)?\s+", re.IGNORECASE)


def tidy(text: str) -> str:
    text = " ".join(text.split()).strip(" .")
    stripped = _LEADING.sub("", text)
    if stripped and stripped != text:
        text = stripped[0].upper() + stripped[1:]
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 1].rsplit(" ", 1)[0] + "…"
    return text


def _robust_z(x: np.ndarray) -> np.ndarray:
    from stream_log.engine.signals.common import robust_z

    return np.nan_to_num(robust_z(x.astype(np.float32)))


def _peaks(x: np.ndarray, threshold: float, gap: int = 30) -> list[int]:
    from stream_log.engine.signals.common import peaks

    return peaks(x, min_gap=gap, threshold=threshold, limit=400)


# ---------------------------------------------------------------------- evidence


def llm_evidence(events: list[dict]) -> list[Evidence]:
    out = []
    for e in events:
        sources = e.get("sources") or []
        category, type_ = e.get("category", ""), str(e.get("type", "")).lower()
        if type_ in SKIP_TYPES or category == "structure":
            continue
        if "llm" in sources:
            if category == "health" and e.get("importance") != "key":
                continue
            weight = IMPORTANCE.get(e.get("importance"), 1.0) + CATEGORY_BONUS.get(category, 0.0)
            weight += 0.4 * min(max(int(e.get("agreement", 1)) - 1, 0), 2)
            out.append(Evidence(float(e["start"]), weight, [CATEGORY_TAG.get(category, "topic")],
                                tidy(e.get("title", "")), 3))
        elif category == "community":
            tag = {"gift": "subs", "sub": "subs"}.get(type_, type_)
            if tag == "tts":
                continue
            weight = {"key": 2.0, "notable": 1.2}.get(e.get("importance"), 0.6)
            out.append(Evidence(float(e["start"]), weight, [tag], tidy(e.get("title", "")), 2))
    return out


def reaction_evidence(signals: dict[str, np.ndarray]) -> list[Evidence]:
    out = []
    for name, tag, threshold in REACTIONS:
        x = signals.get(name)
        if x is None:
            continue
        for t in _peaks(x, threshold):
            out.append(Evidence(float(t), min(1.0 + 4.0 * float(x[t]), 2.0), [tag]))
    arousal = signals.get("arousal")
    if arousal is not None and len(arousal) > 600:
        limit = max(1, int(LOUD_PER_HOUR * len(arousal) / 3600))
        threshold = float(np.percentile(arousal, LOUD_PERCENTILE))
        loudest = sorted(_peaks(arousal, threshold, gap=120), key=lambda t: -arousal[t])[:limit]
        out += [Evidence(float(t), 0.6, ["loud"]) for t in loudest]
    return out


def chat_messages(project, chat_file: Path | None) -> list:
    """The project's chat (imported with the video), else a downloaded chat file."""
    from stream_log.engine.ingest.chat import load_chat, read_saved_chat

    messages = read_saved_chat(project.path("chat.json"))
    if not messages and chat_file and chat_file.exists():
        _, messages = load_chat(chat_file, duration=float(project.meta["media"]["duration"]))
    return messages


def chat_rate(messages: list, duration: float) -> np.ndarray:
    """Viewer messages per second (bots, subs and other notices left out)."""
    rate = np.zeros(int(duration) + 1)
    for message in messages:
        if message.kind == "message" and 0 <= message.t < len(rate):
            rate[int(message.t)] += 1
    return rate


def chat_evidence(rate: np.ndarray) -> list[Evidence]:
    if not np.any(rate):
        return []
    rate = rate.astype(np.float64)
    n = len(rate)
    window = np.convolve(rate, np.ones(15) / 15, mode="same")  # 15 s of chat
    out = []
    z = _robust_z(window)
    for t in _peaks(z, 3.0, gap=60):
        lo, hi = max(0, t - 600), min(n, t + 600)
        usual = max(float(np.median(window[lo:hi])), 0.02)
        ratio = window[t] / usual
        if ratio < CHAT_RATIO or window[t] < CHAT_MIN_RATE:
            continue
        out.append(Evidence(max(0.0, t - 8.0), min(ratio / 4.0, 1.5),
                            [f"chat ×{round(min(ratio, 99))}"]))
    return out


def quote_at(lines: list[dict], t: float, words: int = 10) -> str:
    near = [line for line in lines if line["start"] - 6 <= t <= line["end"] + 6]
    if not near:
        return ""
    line = min(near, key=lambda line: abs((line["start"] + line["end"]) / 2 - t))
    text = " ".join(line["text"].split()[:words]).strip(" ,.")
    return f"“{text}”" if text else ""


# ---------------------------------------------------------------------- moments


def cluster(evidence: list[Evidence], lines: list[dict]) -> list[Line]:
    out: list[Line] = []
    group: list[Evidence] = []

    def flush() -> None:
        if not group:
            return
        texts = sorted((e for e in group if e.text), key=lambda e: (-e.text_rank, -e.weight))
        text = texts[0].text if texts else ""
        strongest = max(group, key=lambda e: e.weight)
        if not text:
            text = quote_at(lines, strongest.t)
        if not text:
            return
        tags: list[str] = []
        for e in group:
            for tag in e.tags:
                if tag not in tags and not (tag.startswith("chat ×") and any(
                        t.startswith("chat ×") for t in tags)):
                    tags.append(tag)
        tags.sort(key=lambda tag: TAG_ORDER.index(tag.split(" ")[0])
                  if tag.split(" ")[0] in TAG_ORDER else len(TAG_ORDER))
        # Several LLM events in one moment count once, plus a little for each other kind.
        llm = sorted((e.weight for e in group if e.text_rank == 3), reverse=True)
        other = [e.weight for e in group if e.text_rank != 3]
        score = (llm[0] + 0.25 * sum(llm[1:]) if llm else 0.0) + sum(other)
        if len({tag.split(" ")[0] for tag in tags}) >= 3:
            score += 0.5  # several kinds of evidence agreeing
        out.append(Line(min(e.t for e in group), text, tags, round(score, 2)))

    for item in sorted(evidence, key=lambda e: e.t):
        if group and item.t - group[0].t > MERGE:
            flush()
            group = []
        group.append(item)
    flush()
    return out


# ---------------------------------------------------------------------- blocks


def _chapter_titles(project_root: Path) -> list[dict]:
    """The analysis's chapters with the LLM's titles when it wrote them."""
    path = project_root / "chapters.json"
    if not path.exists():
        return []
    chapters = json.loads(path.read_text(encoding="utf-8"))["chapters"]
    llm_path = project_root / "llm.json"
    titles = {}
    if llm_path.exists():
        titles = json.loads(llm_path.read_text(encoding="utf-8")).get("chapters") or {}
    out = []
    for chapter in chapters:
        llm = titles.get(chapter["id"]) if isinstance(titles, dict) else None
        out.append({"start": chapter["start"], "end": chapter["end"], "mode": chapter.get("mode"),
                    "title": tidy((llm or {}).get("title") or chapter.get("title") or "")})
    return out


def _overlap_title(chapters: list[dict], start: float, end: float) -> str:
    best, best_len = "", 0.0
    for chapter in chapters:
        overlap = min(end, chapter["end"]) - max(start, chapter["start"])
        if overlap > best_len:
            best, best_len = chapter["title"], overlap
    return best


def _category_at(twitch_chapters: list[dict], t: float) -> str:
    for chapter in twitch_chapters:
        if chapter.get("start", 0) <= t < (chapter.get("end") or 1e12):
            return chapter.get("title") or ""
    return ""


def chapter_blocks(chapters: list[dict], start: float, end: float, prefix: str = "") -> list[Block]:
    """[start, end) cut into blocks of about BLOCK_TARGET along the analysis's chapters."""
    blocks: list[Block] = []
    inside = [c for c in chapters if c["end"] > start and c["start"] < end]
    for chapter in inside:
        lo, hi = max(start, chapter["start"]), min(end, chapter["end"])
        if blocks and blocks[-1].end - blocks[-1].start < BLOCK_TARGET:
            blocks[-1].end = hi
        else:
            title = " · ".join(p for p in (prefix, chapter["title"]) if p) or "Stream"
            blocks.append(Block(lo, hi, title, "stretch"))
    if not blocks:
        return [Block(start, end, prefix or "Stream", "stretch")]
    blocks[0].start, blocks[-1].end = start, end
    if len(blocks) > 1 and blocks[-1].end - blocks[-1].start < BLOCK_TARGET / 3:
        tiny = blocks.pop()  # no tiny block at the end
        blocks[-1].end = tiny.end
    return blocks


def is_game(game: dict) -> bool:
    """TFT games last 20-45 min. A longer "game" is a tournament broadcast or a watch party
    moving from game to game with no lobby or loading screen in between."""
    return game["end"] - game["start"] <= MAX_GAME


def make_blocks(duration: float, games: list[dict], chapters: list[dict],
                twitch_chapters: list[dict]) -> list[Block]:
    if not games:
        return chapter_blocks(chapters, 0.0, duration)
    blocks: list[Block] = []

    def stretch(start: float, end: float, where: str) -> Block:
        category = _category_at(twitch_chapters, (start + end) / 2)
        topic = _overlap_title(chapters, start, end)
        label = category if category and category.lower() not in ("teamfight tactics",) else ""
        title = " · ".join(part for part in (label or where, topic) if part)
        return Block(start, end, title, "stretch")

    cursor, number = 0.0, 0
    for game in games:
        start = game["start"]
        gap = start - cursor
        if gap >= MIN_BLOCK or (not blocks and gap >= 120):
            blocks.append(stretch(cursor, start,
                                  "Between games" if blocks else "Before the first game"))
        elif blocks:
            blocks[-1].end = start  # post-game talk stays with the game before
        else:
            start = 0.0
        watched = bool(game.get("watched"))
        if is_game(game):
            if watched:
                title = f"Watched game · {minutes(game['end'] - game['start'])}"
            else:
                number += 1
                title = f"Game {number} · {minutes(game['end'] - game['start'])}"
            blocks.append(Block(start, game["end"], title, "game",
                                game_start=game["start"], game_end=game["end"]))
        else:
            blocks.extend(chapter_blocks(chapters, start, game["end"],
                                         "Watching TFT" if watched else "TFT on screen"))
        cursor = game["end"]
    if duration - cursor >= 120:
        blocks.append(stretch(cursor, duration, "After the last game"))
    elif blocks:
        blocks[-1].end = duration
    return blocks


def fill(blocks: list[Block], lines: list[Line], per_hour: int) -> None:
    for line in lines:
        block = next((b for b in blocks if b.start <= line.t < b.end), blocks[-1])
        block.lines.append(line)
    for block in blocks:
        hours = (block.end - block.start) / 3600
        allowed = max(1, round(per_hour * hours))
        if len(block.lines) > allowed:
            ranked = sorted(block.lines, key=lambda line: -line.score)
            keep = {id(line) for line in ranked[:allowed]}
            block.folded = [line for line in block.lines if id(line) not in keep]
            block.lines = [line for line in block.lines if id(line) in keep]


# ---------------------------------------------------------------------- build


def build(project, games: list[dict], video: Path, *, per_hour: int = PER_HOUR,
          chat_file: Path | None = None) -> StreamList:
    root = project.root
    meta = project.meta
    duration = float(meta["media"]["duration"])
    events = json.loads((root / "events.json").read_text(encoding="utf-8"))["events"]
    signals: dict[str, np.ndarray] = {}
    if (root / "signals.npz").exists():
        with np.load(root / "signals.npz") as data:
            signals = {k: data[k] for k in data.files}
    transcript = json.loads((root / "transcript.json").read_text(encoding="utf-8"))
    lines = transcript["lines"]
    games = mark_watched(games, transcript)
    played = [g for g in games if is_game(g) and not g["watched"]]
    rate = chat_rate(chat_messages(project, chat_file), duration)

    evidence = llm_evidence(events) + reaction_evidence(signals) + chat_evidence(rate)
    moments = cluster(evidence, lines)
    chapters = _chapter_titles(root)
    blocks = make_blocks(duration, games, chapters, meta.get("chapters") or [])
    fill(blocks, moments, per_hour)

    summary = ""
    llm_path = root / "llm.json"
    if llm_path.exists():
        summary = (json.loads(llm_path.read_text(encoding="utf-8")).get("summary") or "").strip()
    date = ""
    if meta.get("started_at"):
        from datetime import datetime

        date = datetime.fromtimestamp(meta["started_at"]).strftime("%d %b %Y").lstrip("0")
    return StreamList(
        name=video.stem, title=meta.get("title") or video.stem, channel=meta.get("channel") or "",
        date=date, duration=duration, url=meta.get("url"), video=str(video), summary=summary,
        games=len(played), game_seconds=sum(g["end"] - g["start"] for g in played),
        watched=sum(is_game(g) and g["watched"] for g in games),
        blocks=blocks, per_hour=per_hour)
