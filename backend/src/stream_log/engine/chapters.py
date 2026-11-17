"""Chapters: topic shifts + mode changes + game switches, titled by the LLM.

Without an LLM, titles fall back to "<mode>: <keywords>" and summaries stay empty.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from stream_log.engine.llm import LlmClient, LlmError
from stream_log.engine.project import clock

BLOCK = 60  # seconds of transcript per embedding block
TILING_K = 3  # blocks on each side when comparing
MIN_CHAPTER = 300
MIN_FORCED_SEGMENT = 180

STOPWORDS = {"a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "being", "but", "by", "can", "could", "did", "do", "does", "doing", "don't", "down", "for", "from", "get", "got", "gonna", "had", "has", "have", "having", "he", "her", "here", "him", "his", "how", "i", "i'm", "if", "in", "into", "is", "it", "it's", "its", "just", "know", "like", "literally", "me", "more", "my", "no", "not", "now", "of", "off", "oh", "ok", "okay", "on", "one", "or", "our", "out", "over", "really", "right", "said", "say", "says", "see", "she", "so", "some", "than", "that", "that's", "the", "their", "them", "then", "there", "they", "think", "this", "to", "too", "um", "uh", "up", "us", "very", "wanna", "was", "way", "we", "well", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "yeah", "yes", "you", "you're", "your", "actually", "going", "want", "something", "thing", "things", "let's", "let", "mean", "dude", "guys", "chat"}


def blocks(lines: list[dict], duration: int) -> list[str]:
    texts = [[] for _ in range(max(1, math.ceil(duration / BLOCK)))]
    for line in lines:
        texts[min(int(line["start"] // BLOCK), len(texts) - 1)].append(line["text"])
    return [" ".join(t) for t in texts]


def depth_scores(embs: np.ndarray, k: int = TILING_K) -> np.ndarray:
    """TextTiling depth at each block boundary (index i = boundary before block i)."""
    n = len(embs)
    sims = np.ones(n)
    for i in range(1, n):
        left = embs[max(0, i - k) : i].mean(axis=0)
        right = embs[i : i + k].mean(axis=0)
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        sims[i] = float(left @ right / denom) if denom > 0 else 1.0
    depth = np.zeros(n)
    for i in range(1, n):
        left_peak = sims[max(1, i - k) : i + 1].max()
        right_peak = sims[i : i + k + 1].max()
        depth[i] = (left_peak - sims[i]) + (right_peak - sims[i])
    return depth


def choose_boundaries(depth: np.ndarray, forced: list[float], duration: int) -> list[float]:
    """Forced boundaries first, then the deepest topic shifts at least MIN_CHAPTER apart."""
    chosen = sorted({0.0, *[f for f in forced if 0 < f < duration]})
    if len(depth) > 1:
        threshold = depth[1:].mean() + 0.5 * depth[1:].std()
        for i in np.argsort(-depth):
            if i == 0 or depth[i] < threshold:
                continue
            t = float(i * BLOCK)
            if all(abs(t - c) >= MIN_CHAPTER for c in chosen):
                chosen.append(t)
    chosen = sorted(chosen)
    # Drop boundaries that leave a sliver at the end.
    while len(chosen) > 1 and duration - chosen[-1] < MIN_CHAPTER / 2:
        chosen.pop()
    return chosen


def keywords(text: str, background: Counter, top: int = 4) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z']{2,}", text.lower()) if w not in STOPWORDS]
    counts = Counter(words)
    total = sum(background.values()) or 1
    scored = {w: c * math.log(total / (background[w] + 1)) for w, c in counts.items() if c >= 2}
    return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:top]]


def build_chapters(
    lines: list[dict], duration: int, embedder, segments: list[dict], twitch_chapters: list[dict],
    fused: np.ndarray,
) -> list[dict]:
    texts = blocks(lines, duration)
    embs = embedder.embed([t or "(silence)" for t in texts])
    forced = [s["start"] for s in segments if s["end"] - s["start"] >= MIN_FORCED_SEGMENT]
    forced += [c["start"] for c in twitch_chapters if c.get("start")]
    bounds = choose_boundaries(depth_scores(embs), forced, duration)
    background = Counter(w for t in texts for w in re.findall(r"[a-z][a-z']{2,}", t.lower()))
    chapters = []
    for index, start in enumerate(bounds):
        end = bounds[index + 1] if index + 1 < len(bounds) else float(duration)
        part = [ln for ln in lines if start <= ln["start"] < end]
        text = " ".join(ln["text"] for ln in part)
        mode_time: Counter = Counter()
        for s in segments:
            overlap = min(s["end"], end) - max(s["start"], start)
            if overlap > 0:
                mode_time[s["mode"]] += overlap
        mode = mode_time.most_common(1)[0][0] if mode_time else "chatting"
        game = next((c["title"] for c in reversed(twitch_chapters)
                     if c.get("start") is not None and c["start"] <= start + 1), None)
        lo, hi = int(start), max(int(start) + 1, int(end))
        curve = [round(float(fused[m : m + 60].mean()), 2) for m in range(lo, hi, 60)]
        top = np.argsort(-fused[lo:hi])[:1]
        words = keywords(text, background)
        chapters.append({
            "id": f"ch_{index + 1:02d}",
            "start": round(start, 1),
            "end": round(end, 1),
            "mode": mode,
            "game": game,
            "title": f"{mode.title()}: {', '.join(words)}" if words else mode.title(),
            "summary": None,
            "keywords": words,
            "energy_curve": curve,
            "peak": float(lo + top[0]) if len(top) else start,
        })
    return chapters


CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "activity": {"type": "string"},
    },
    "required": ["title", "summary", "activity"],
    "additionalProperties": False,
}

SYSTEM = (
    "You label chapters of a livestream VOD for the streamer's own records. Only describe what the "
    "transcript shows; never invent events. Titles are 3-8 words, specific and in plain English. "
    "Summaries are 1-3 sentences."
)


def transcript_excerpt(lines: list[dict], tags: dict[int, str], start: float, end: float,
                       limit: int = 12000) -> str:
    out = []
    for line in lines:
        if start <= line["start"] < end:
            who = tags.get(line["id"], "streamer")
            out.append(f"[{clock(line['start'])}] {who}: {line['text']}")
    text = "\n".join(out)
    if len(text) > limit:  # keep the start and the end of very long chapters
        text = text[: limit // 2] + "\n[...]\n" + text[-limit // 2 :]
    return text


def llm_titles(llm: LlmClient, chapters: list[dict], lines: list[dict], tags: dict[int, str],
               channel_note: str = "", errors: list[str] | None = None) -> int:
    done = 0
    for chapter in chapters:
        excerpt = transcript_excerpt(lines, tags, chapter["start"], chapter["end"])
        if not excerpt.strip():
            continue
        prompt = (
            f"{channel_note}\nChapter {clock(chapter['start'])}-{clock(chapter['end'])}, "
            f"stream mode: {chapter['mode']}"
            + (f", game: {chapter['game']}" if chapter["game"] else "")
            + ".\nActivity is one of: banter, focused_gameplay, storytelling, explaining, music, "
            "afk_loading, reacting, community.\n\nTranscript:\n" + excerpt
        )
        try:
            result = llm.chat_json(SYSTEM, prompt, CHAPTER_SCHEMA, max_tokens=600)
        except LlmError as error:
            if errors is not None:
                errors.append(f"chapter {chapter['id']}: {error}")
            continue
        chapter["title"] = result.get("title") or chapter["title"]
        chapter["summary"] = result.get("summary")
        chapter["activity"] = result.get("activity")
        done += 1
    return done


def stream_summary(llm: LlmClient, chapters: list[dict], errors: list[str] | None = None
                   ) -> str | None:
    outline = "\n".join(
        f"- {clock(c['start'])} {c['title']}: {c.get('summary') or ''}" for c in chapters
    )
    try:
        text = llm.chat(SYSTEM, "Write a 3-5 sentence summary of the whole stream from these "
                        "chapters, as one plain-text paragraph (no headings, lists or "
                        "Markdown):\n" + outline, max_tokens=500)
    except LlmError as error:
        if errors is not None:
            errors.append(f"stream summary: {error}")
        return None
    return plain_paragraph(text)


def plain_paragraph(text: str) -> str:
    """Drop Markdown headings / bold title lines some models add despite being asked not to."""
    kept = [ln.strip() for ln in text.strip().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
            and not re.fullmatch(r"\s*\*\*[^*]+\*\*\s*", ln)]
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", " ".join(kept)).strip()


def youtube_chapters(chapters: list[dict]) -> str:
    """YouTube needs the first chapter at 0:00 and chapters >= 10 s long."""
    out = []
    for chapter in chapters:
        if out and chapter["end"] - chapter["start"] < 10:
            continue
        stamp = "0:00" if not out else clock(chapter["start"]).removeprefix("0:")
        out.append(f"{stamp} {chapter['title']}")
    return "\n".join(out)


# ---------------------------------------------------------------------- lore


GENERIC_CHAT = {"nice", "gg", "hi", "hello", "hey", "lmao", "lmfao", "xd", "omg", "wow", "w", "l",
                "f", "rip", "true", "real", "same", "ez", "pog", "poggers", "lol", "lul", "yes",
                "no", "what", "why", "bruh", "sheesh", "based", "?", "!", "hahaha", "haha"}


def lore(lines: list[dict], chat_texts: list[str], min_count: int = 4) -> list[dict]:
    """Phrases the streamer and chat repeat far more than normal speech would.

    Repeated identical chat messages from several people (copypastas, running jokes) and
    transcript 2-3 word phrases without stopwords that recur.
    """
    from stream_log.engine.signals.chat import EMOTE_MEANINGS

    found: list[dict] = []
    normalized = Counter(t.strip().lower() for t in chat_texts if len(t.strip()) >= 3)
    for text, count in normalized.most_common(60):
        words = text.split()
        generic = len(words) < 2 and (text in EMOTE_MEANINGS or text in STOPWORDS
                                      or text in GENERIC_CHAT)
        if count >= max(min_count, 5) and not generic:
            found.append({"phrase": text, "count": count, "source": "chat"})
    words = [w for ln in lines for w in re.findall(r"[a-z][a-z']+", ln["text"].lower())]
    grams: Counter = Counter()
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            if sum(w in STOPWORDS for w in gram) == 0:
                grams[" ".join(gram)] += 1
    for phrase, count in grams.most_common(20):
        if count >= min_count:
            found.append({"phrase": phrase, "count": count, "source": "transcript"})
    return found[:25]
