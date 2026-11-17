"""Chat signals judged by vibe, not keywords.

Behaviour (rate, unique chatters, message length, caps, emote density, convergence) plus meaning:
each message gets a vibe distribution from go_emotions and from its embedding's similarity to
vibe prototypes. Emotes are replaced by short descriptions before embedding so "KEKW" reads as
laughter. Everything is weighted by how active chat is, so tiny chats never dominate.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher

import numpy as np

from stream_log.engine.chapters import STOPWORDS
from stream_log.engine.ingest.chat import ChatMessage
from stream_log.engine.signals.common import counts_per_second, smooth

VIBES = ["hype", "laughter", "shock", "cringe", "sympathy", "wholesome", "tension", "conversation"]

VIBE_PROTOTYPES = {
    "hype": ["LETS GOOOO", "that was insane", "huge play", "W streamer", "clutch"],
    "laughter": ["LMAO", "i'm dying laughing", "that's so funny", "hahahaha", "crying laughing"],
    "shock": ["WHAT", "no way", "did that just happen", "omg", "what the hell was that"],
    "cringe": ["that's so cringe", "awkward", "L", "yikes", "secondhand embarrassment"],
    "sympathy": ["so sad", "that's rough", "sorry for your loss", "we love you", "rip"],
    "wholesome": ["so wholesome", "this is so cute", "love this community", "happy for you",
                  "aww"],
    "tension": ["oh no", "don't die", "careful", "this is scary", "please please please"],
    "conversation": ["what game is this", "how was your day", "where are you from",
                     "what do you think about", "can you explain"],
}

# go_emotions label -> vibe.
GO_EMOTION_VIBES = {
    "admiration": "hype", "excitement": "hype", "pride": "hype",
    "amusement": "laughter", "joy": "laughter",
    "surprise": "shock", "realization": "shock",
    "embarrassment": "cringe", "disgust": "cringe", "disapproval": "cringe",
    "sadness": "sympathy", "grief": "sympathy", "remorse": "sympathy", "caring": "sympathy",
    "love": "wholesome", "gratitude": "wholesome", "approval": "wholesome", "optimism": "wholesome",
    "fear": "tension", "nervousness": "tension",
    "curiosity": "conversation", "confusion": "conversation", "neutral": "conversation",
}

# Seed emote meanings (Twitch / BTTV / FFZ / 7TV classics). Unknown emotes are left as-is; their
# meaning comes from the surrounding text. There is no per-channel learning.
EMOTE_MEANINGS = {
    "kekw": "laughing hard", "lul": "laughing", "lulw": "laughing", "omegalul": "laughing hard",
    "icant": "laughing so hard", "lmao": "laughing", "lol": "laughing", "kekl": "laughing",
    "pog": "amazing hype", "pogchamp": "amazing hype", "poggers": "amazing hype",
    "pogu": "amazing hype", "letsgo": "lets go hype", "ez": "easy win", "clap": "applause",
    "monkas": "nervous scary", "monkaw": "very scary", "pepega": "stupid funny",
    "sadge": "sad", "pepehands": "sad crying", "biblethump": "sad crying", "feelsbadman": "sad",
    "widepeepohappy": "happy wholesome", "peepohappy": "happy", "<3": "love", "catjam": "music vibing",
    "kreygasm": "amazing", "residentsleeper": "boring", "notlikethis": "oh no",
    "d:": "shocked", "wutface": "shocked disgusted", "o7": "respect salute", "5head": "smart",
    "copium": "denial coping", "aware": "uneasy realization", "ratjam": "music vibing",
    "monkaomega": "terrified", "pausechamp": "suspense", "hypers": "hype", "omegaroll": "laughing",
}

_TOKEN = re.compile(r"\S+")
_UPPER = re.compile(r"[A-Z]")
_LETTER = re.compile(r"[A-Za-z]")

MAX_DELAY = 20
DEFAULT_DELAY = 4  # typical stream latency between the video and chat reacting
MIN_DELAY_CORRELATION = 0.2
INTERACTION_LOOKBACK = 60.0


def describe_emotes(text: str) -> tuple[str, int]:
    """Replace known emotes with descriptions; returns (text, emote count)."""
    count = 0
    words = []
    for token in _TOKEN.findall(text):
        meaning = EMOTE_MEANINGS.get(token.lower())
        if meaning:
            count += 1
            words.append(meaning)
        else:
            words.append(token)
    return " ".join(words), count


def message_features(messages: list[ChatMessage]) -> dict[str, np.ndarray]:
    lengths, caps, emotes = [], [], []
    for message in messages:
        text = message.text
        letters = len(_LETTER.findall(text))
        caps.append(len(_UPPER.findall(text)) / letters if letters else 0.0)
        _, known = describe_emotes(text)
        tokens = max(len(_TOKEN.findall(text)), 1)
        emotes.append(min((known + len(message.emotes)) / tokens, 1.0))
        lengths.append(len(text))
    return {"length": np.array(lengths, dtype=np.float32),
            "caps": np.array(caps, dtype=np.float32),
            "emote": np.array(emotes, dtype=np.float32)}


def vibe_scores(
    messages: list[ChatMessage], embedder, emotions, prototype_embs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(N x len(VIBES)) vibe distribution per message, and message embeddings."""
    texts = [describe_emotes(m.text)[0] or m.text for m in messages]
    embs = embedder.embed(texts)
    if prototype_embs is None:
        prototype_embs = prototype_matrix(embedder)
    proto = embs @ prototype_embs.T  # cosine to each vibe's mean prototype
    proto = np.exp((proto - proto.max(axis=1, keepdims=True)) * 20)
    proto /= proto.sum(axis=1, keepdims=True)

    emo = emotions.scores(texts)
    from_emotions = np.zeros((len(messages), len(VIBES)), dtype=np.float32)
    for column, label in enumerate(emotions.labels):
        vibe = GO_EMOTION_VIBES.get(label)
        if vibe:
            from_emotions[:, VIBES.index(vibe)] += emo[:, column]
    from_emotions /= from_emotions.sum(axis=1, keepdims=True) + 1e-9
    return (0.5 * proto + 0.5 * from_emotions).astype(np.float32), embs


def prototype_matrix(embedder) -> np.ndarray:
    rows = []
    for vibe in VIBES:
        vectors = embedder.embed(VIBE_PROTOTYPES[vibe])
        mean = vectors.mean(axis=0)
        rows.append(mean / (np.linalg.norm(mean) + 1e-9))
    return np.stack(rows)


def _window_unique(messages: list[ChatMessage], duration: int, window: int = 10) -> np.ndarray:
    buckets: list[set[str]] = [set() for _ in range(duration)]
    for message in messages:
        t = int(message.t)
        if 0 <= t < duration:
            buckets[t].add(message.user.lower())
    out = np.zeros(duration, dtype=np.float32)
    for t in range(duration):
        lo, hi = max(0, t - window // 2), min(duration, t + window // 2 + 1)
        out[t] = len(set().union(*buckets[lo:hi]))
    return out


def convergence(times: np.ndarray, embs: np.ndarray, duration: int, window: int = 10
                ) -> np.ndarray:
    """Mean pairwise similarity of messages around each second: chat 'saying the same thing'."""
    out = np.full(duration, np.nan, dtype=np.float32)
    if len(times) == 0:
        return out
    order = np.argsort(times)
    times, embs = times[order], embs[order]
    for t in range(0, duration, 2):
        lo, hi = np.searchsorted(times, [t - window / 2, t + window / 2])
        if hi - lo >= 3:
            chunk = embs[lo:hi][-40:]
            sims = chunk @ chunk.T
            n = len(chunk)
            out[t : t + 2] = (sims.sum() - n) / (n * (n - 1))
    return out


def estimate_delay(chat_rate: np.ndarray, arousal: np.ndarray) -> tuple[int, float]:
    """Chat reacts after the video. Find the lag (0-20 s) where chat activity best follows audio
    arousal. Falls back to DEFAULT_DELAY when the correlation is too weak to trust."""
    a = np.nan_to_num(arousal - np.nanmean(arousal))
    c = np.nan_to_num(chat_rate - np.nanmean(chat_rate))
    if len(a) < 120 or np.std(a) == 0 or np.std(c) == 0:
        return DEFAULT_DELAY, 0.0
    best_lag, best = DEFAULT_DELAY, 0.0
    for lag in range(MAX_DELAY + 1):
        x, y = a[: len(a) - lag], c[lag:]
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else 0.0
        if corr > best:
            best_lag, best = lag, corr
    if best < MIN_DELAY_CORRELATION:
        return DEFAULT_DELAY, best
    return best_lag, best


def shift_back(x: np.ndarray, delay: int) -> np.ndarray:
    """Move a chat signal `delay` seconds earlier, so it lines up with the moment it reacts to."""
    if delay <= 0:
        return x
    return np.concatenate([x[delay:], np.full(delay, x[-1] if len(x) else 0, dtype=x.dtype)])


def kl_shift(vibes: np.ndarray, baseline_window: int = 600) -> np.ndarray:
    """KL divergence of the current 30 s vibe mix from the surrounding 10-min mix."""
    duration = len(vibes)
    eps = 1e-3
    current = np.stack([smooth(vibes[:, k], 30) for k in range(vibes.shape[1])], axis=1) + eps
    base = np.stack([smooth(vibes[:, k], baseline_window) for k in range(vibes.shape[1])],
                    axis=1) + eps
    # Messages in the current window; with no chat, there is no vibe to have shifted.
    activity = np.clip((current.sum(axis=1) - eps * vibes.shape[1]) * 30 / 3, 0, 1)
    current /= current.sum(axis=1, keepdims=True)
    base /= base.sum(axis=1, keepdims=True)
    out = (current * np.log(current / base)).sum(axis=1) * activity
    return out.astype(np.float32)[:duration]


def chat_signals(
    messages: list[ChatMessage], duration: int, embedder, emotions,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Raw (not yet delay-shifted) per-second chat signals."""
    chats = [m for m in messages if m.kind in ("message", "bits", "donation", "member")]
    times = np.array([m.t for m in chats], dtype=np.float64)
    rate = smooth(counts_per_second(times, duration), 5)
    out = {"chat_rate": rate, "chat_unique": _window_unique(chats, duration)}
    if not chats:
        zeros = np.zeros(duration, dtype=np.float32)
        out |= {"chat_length": zeros, "chat_caps": zeros, "chat_emote": zeros,
                "chat_convergence": zeros, "chat_vibe_shift": zeros}
        out |= {f"vibe_{v}": zeros for v in VIBES}
        return out
    feats = message_features(chats)
    seconds = np.clip(times.astype(int), 0, duration - 1)

    def mean_per_second(values: np.ndarray, window: int = 10) -> np.ndarray:
        total = np.bincount(seconds, weights=values, minlength=duration)[:duration]
        count = np.bincount(seconds, minlength=duration)[:duration]
        total, count = smooth(total, window) * window, smooth(count, window) * window
        return np.where(count > 0, total / np.maximum(count, 1e-9), np.nan).astype(np.float32)

    out["chat_length"] = mean_per_second(feats["length"])
    out["chat_caps"] = mean_per_second(feats["caps"])
    out["chat_emote"] = mean_per_second(feats["emote"])
    if progress:
        progress(0.1, f"embedding {len(chats)} messages")
    vibes, embs = vibe_scores(chats, embedder, emotions)
    out["chat_convergence"] = convergence(times, embs, duration)
    per_vibe = np.zeros((duration, len(VIBES)), dtype=np.float32)
    for k in range(len(VIBES)):
        per_vibe[:, k] = np.bincount(seconds, weights=vibes[:, k], minlength=duration)[:duration]
    for k, vibe in enumerate(VIBES):
        out[f"vibe_{vibe}"] = smooth(per_vibe[:, k], 10)
    out["chat_vibe_shift"] = kl_shift(per_vibe)
    return out


def chat_reliability(unique_per_10s: np.ndarray) -> float:
    """Fusion weight of chat: clamp(baseline unique chatters per minute / 15, 0, 1)."""
    if len(unique_per_10s) == 0:
        return 0.0
    per_minute = float(np.median(unique_per_10s)) * 6 * 0.6  # 10 s windows overlap across a min
    return float(np.clip(per_minute / 15.0, 0.0, 1.0))


# ---------------------------------------------------------------------- interactions

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def find_interactions(
    lines: list[dict], tags: dict[int, str], messages: list[ChatMessage]
) -> list[dict]:
    """Streamer lines that read a chat message, say a chatter's name, or answer a question."""
    chats = [m for m in messages if m.kind == "message"]
    times = np.array([m.t for m in chats])
    users = {m.user.lower(): m.user for m in chats if len(m.user) >= 4}
    out = []
    for line in lines:
        if tags.get(line["id"], "streamer") != "streamer":
            continue
        words = _words(line["text"])
        if not words:
            continue
        lo, hi = np.searchsorted(times, [line["start"] - INTERACTION_LOOKBACK, line["start"] + 2])
        recent = chats[lo:hi]
        found = None
        for word in set(words):
            if word in users and any(m.user.lower() == word for m in recent):
                found = {"kind": "says_username", "user": users[word], "score": 1.0}
                break
        if found is None and len(words) >= 3:
            for message in reversed(recent):
                candidate = _words(message.text)
                if len(candidate) < 3:
                    continue
                ratio = SequenceMatcher(None, words, candidate).ratio()
                if ratio >= 0.6:
                    found = {"kind": "reads_chat", "user": message.user, "score": round(ratio, 2),
                             "message": message.text, "message_t": message.t}
                    break
        if found is None:
            content = {w for w in words if len(w) > 3 and w not in STOPWORDS}
            for message in reversed(recent[-30:]):
                if message.text.rstrip().endswith("?"):
                    shared = content & set(_words(message.text))
                    if len(shared) >= 2:
                        found = {"kind": "answers_question", "user": message.user,
                                 "score": round(len(shared) / max(len(content), 1), 2),
                                 "message": message.text, "message_t": message.t}
                        break
        if found:
            out.append({"t": line["start"], "line_id": line["id"], **found})
    return out
