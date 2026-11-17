"""Cluster line embeddings into voices and tag each line streamer | guest | media | tts.

- **streamer**: the cluster closest to the profile's enrolled voice, or the cluster with the most
  speech when the profile has no voice yet (then suggested for enrollment).
- **tts**: a non-streamer line that repeats a recent chat message (donation / bits TTS bots).
- **guest**: any other voice.
- **media** (game voice lines, song lyrics, video audio): a line that repeats word for word
  elsewhere in the stream in a voice unlike the streamer's. On a real TFT VOD, champion lines
  ("I seek only the strongest") and song choruses were too short and too varied to form a voice
  cluster, but they repeat; the streamer's own catchphrases repeat in the streamer's voice. Whole
  voices can also be labelled media in the project's speaker_labels.json.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

from stream_log.engine.ingest.chat import ChatMessage

CLUSTER_THRESHOLD = 0.5  # cosine similarity to join a voice cluster
STREAMER_THRESHOLD = 0.45  # cluster centroid vs. enrolled voice
LINE_STREAMER_MARGIN = 0.12  # lines in the streamer cluster may sit this far under the threshold
MIN_CLUSTER_SHARE = 0.02  # clusters below 2 % of speech merge into their nearest neighbour
TTS_LOOKBACK = 120.0
TTS_MIN_WORDS = 3
TTS_MATCH = 0.7
# Repeated-line media rule, calibrated on a 3.7 h VOD: below 0.25 similarity to the streamer every
# repeated 3+ word line was game or music audio; at 0.35 streamer lines started to appear.
MEDIA_SIM = {2: 0.15, 3: 0.25}  # max similarity to the streamer, by line length in words
MEDIA_REPEAT_MATCH = 0.8
MEDIA_REPEAT_GAP = 30.0
MEDIA_PARTNER_SLACK = 0.1
# Quiet guests: someone who talks for a while in one part of a long stream (a parent helping with
# a PC build) is a small share of its speech, and their short, off-mic lines match each other too
# loosely (~0.4) to form a cluster, so they end up in the streamer's. On a 7.8 h VOD, lines in the
# streamer's cluster that sound unlike the streamer regrouped at a looser threshold into the guest
# (270 s within 1.4 h) and song lyrics (140 s over 5.5 h); how dense a voice is tells them apart.
GUEST_SEED_SIM = 0.3  # seed lines sound at most this much like the streamer...
GUEST_SEED_WORDS = 4  # ...and are sentences: "uh" or "yeah" embed as noise
GUEST_SEED_SECONDS = 1.5
GUEST_CLUSTER_THRESHOLD = 0.4
GUEST_PRESENCE = 60.0  # seconds of seed lines within some GUEST_WINDOW
GUEST_WINDOW = 1800.0
GUEST_MARGIN = 0.1  # a line joins the guest when it matches the guest this much better...
GUEST_REACH = 600.0  # ...and lies within 10 min of a seed line
TAGS = ("streamer", "guest", "media", "tts")


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cluster(
    embeddings: np.ndarray, weights: np.ndarray, threshold: float = CLUSTER_THRESHOLD,
    min_share: float = MIN_CLUSTER_SHARE,
) -> np.ndarray:
    """Leader clustering (longest lines first) + weighted k-means refinement.

    Streams have few voices, so this O(n·k) approach is enough and stays fast for 10k+ lines.
    Labels are ordered by total weight: 0 is the voice with the most speech.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    centroids: list[np.ndarray] = []
    totals: list[float] = []
    for i in np.argsort(-weights):
        if centroids:
            sims = np.asarray(centroids) @ embeddings[i]
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                totals[best] += weights[i]
                centroids[best] = _unit(centroids[best] + (embeddings[i] - centroids[best])
                                        * weights[i] / totals[best])
                continue
        centroids.append(embeddings[i].copy())
        totals.append(float(weights[i]))

    labels = np.zeros(n, dtype=int)
    for _ in range(6):
        matrix = np.asarray(centroids)
        labels = np.argmax(embeddings @ matrix.T, axis=1)
        share = np.bincount(labels, weights=weights, minlength=len(matrix)) / weights.sum()
        keep = [k for k in range(len(matrix)) if share[k] >= min_share]
        if not keep:
            keep = [int(np.argmax(share))]
        centroids = [
            _unit((embeddings[labels == k] * weights[labels == k, None]).sum(axis=0))
            for k in keep if np.any(labels == k)
        ]
        # Merge voices that ended up too similar.
        merged: list[np.ndarray] = []
        for c in centroids:
            if merged and max(float(m @ c) for m in merged) >= threshold + 0.1:
                continue
            merged.append(c)
        centroids = merged

    labels = np.argmax(embeddings @ np.asarray(centroids).T, axis=1)
    order = np.argsort(-np.bincount(labels, weights=weights, minlength=len(centroids)))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels]


def _centroids(embeddings: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return _unit(np.stack([
        (embeddings[labels == k] * weights[labels == k, None]).sum(axis=0)
        for k in range(int(labels.max()) + 1)
    ]))


def _presence(starts: np.ndarray, weights: np.ndarray, window: float) -> float:
    """Most speech (s) the lines put inside any stretch of `window` seconds."""
    order = np.argsort(starts)
    s, total = starts[order], np.concatenate([[0.0], np.cumsum(weights[order])])
    ends = np.searchsorted(s, s + window)
    return float((total[ends] - total[:-1]).max())


def split_quiet_voices(
    embeddings: np.ndarray, weights: np.ndarray, labels: np.ndarray, streamer: int,
    reference: np.ndarray, starts: np.ndarray, words: np.ndarray,
) -> np.ndarray:
    """Give voices hidden in the streamer's cluster labels of their own (see GUEST_*)."""
    labels = labels.copy()
    sims = embeddings @ reference
    seeds = np.flatnonzero((labels == streamer) & (sims < GUEST_SEED_SIM)
                           & (weights >= GUEST_SEED_SECONDS) & (words >= GUEST_SEED_WORDS))
    if len(seeds) < 2:
        return labels
    groups = cluster(embeddings[seeds], weights[seeds], GUEST_CLUSTER_THRESHOLD, min_share=0.0)
    for g in range(int(groups.max()) + 1):
        members = seeds[groups == g]
        if _presence(starts[members], weights[members], GUEST_WINDOW) < GUEST_PRESENCE:
            continue  # sparse: song lyrics, game voices
        centroid = _unit((embeddings[members] * weights[members, None]).sum(axis=0))
        seed_times = np.sort(starts[members])
        pos = np.searchsorted(seed_times, starts)
        gap = np.minimum(np.abs(starts - seed_times[np.maximum(pos - 1, 0)]),
                         np.abs(seed_times[np.minimum(pos, len(seed_times) - 1)] - starts))
        join = ((labels == streamer) & (gap <= GUEST_REACH)
                & (embeddings @ centroid > sims + GUEST_MARGIN))
        labels[join] = labels.max() + 1
    return labels


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def matches_chat(text: str, start: float, chat: list[ChatMessage]) -> ChatMessage | None:
    words = _norm_words(text)
    if len(words) < TTS_MIN_WORDS:
        return None
    for message in chat:
        if message.t > start + 5:
            break
        if message.t < start - TTS_LOOKBACK:
            continue
        candidate = _norm_words(message.text)
        if len(candidate) < TTS_MIN_WORDS:
            continue
        if SequenceMatcher(None, words, candidate).ratio() >= TTS_MATCH:
            return message
    return None


def repeated_media_lines(lines: list[dict], streamer_sims: dict[int, float]) -> dict[int, int]:
    """Lines that repeat elsewhere (>= 30 s apart) in a voice unlike the streamer's.

    Returns {line id: number of repeats}.
    """
    words = {line["id"]: _norm_words(line["text"]) for line in lines}
    candidates = [line for line in lines if len(words[line["id"]]) >= 2]
    found: dict[int, int] = {}
    for line in candidates:
        sim = streamer_sims.get(line["id"])
        limit = MEDIA_SIM[min(len(words[line["id"]]), 3)]
        if sim is None or sim >= limit:
            continue
        mine = words[line["id"]]
        # The other occurrence must also sound unlike the streamer: game lines repeat in the same
        # foreign voice, while "it's all right" repeats in the streamer's (one noisy embedding on
        # a short clip is not enough).
        repeats = sum(
            1 for other in candidates
            if abs(other["start"] - line["start"]) >= MEDIA_REPEAT_GAP
            and streamer_sims.get(other["id"], 1.0) < limit + MEDIA_PARTNER_SLACK
            and abs(len(words[other["id"]]) - len(mine)) <= 2
            and SequenceMatcher(None, mine, words[other["id"]]).ratio() >= MEDIA_REPEAT_MATCH
        )
        if repeats:
            found[line["id"]] = repeats
    return found


def tag_lines(
    lines: list[dict],
    embeddings: dict[int, np.ndarray],
    *,
    voice: np.ndarray | None,
    chat: list[ChatMessage],
    overrides: dict[str, str] | None = None,
) -> dict:
    """Build speakers.json: clusters, per-line tags and an enrollment suggestion."""
    overrides = overrides or {}
    ids = [line["id"] for line in lines if line["id"] in embeddings]
    durations = {line["id"]: max(line["end"] - line["start"], 0.1) for line in lines}
    result: dict = {"clusters": [], "lines": [], "overrides": overrides, "streamer_cluster": None,
                    "voice_enrolled": voice is not None}
    if not ids:
        result["lines"] = [{"id": line["id"], "cluster": None, "tag": "streamer",
                            "reason": "no speaker embeddings"} for line in lines]
        return result

    matrix = np.stack([embeddings[i] for i in ids])
    weights = np.array([durations[i] for i in ids])
    labels = cluster(matrix, weights)
    centroids = _centroids(matrix, weights, labels)
    voice_sims = centroids @ voice if voice is not None else None
    if voice_sims is not None and voice_sims.max() >= STREAMER_THRESHOLD:
        streamer = int(np.argmax(voice_sims))
    elif voice is None:
        streamer = 0  # most speech
    else:
        streamer = None  # enrolled voice not present (e.g. guest-only segment)
    result["streamer_cluster"] = streamer
    # Reference for "sounds like the streamer": the enrolled voice, else the streamer cluster.
    reference = voice if voice is not None else (
        centroids[streamer] if streamer is not None else None)

    if streamer is not None:
        by_id = {line["id"]: line for line in lines}
        labels = split_quiet_voices(
            matrix, weights, labels, streamer, reference,
            starts=np.array([by_id[i]["start"] for i in ids]),
            words=np.array([len(_norm_words(by_id[i]["text"])) for i in ids]),
        )
        centroids = _centroids(matrix, weights, labels)
        voice_sims = centroids @ voice if voice is not None else None
    label_of = dict(zip(ids, labels.tolist(), strict=True))
    count = len(centroids)

    # Short lines had no embedding: give them the voice of the nearest embedded line in time.
    by_time = sorted((line for line in lines if line["id"] in label_of), key=lambda x: x["start"])
    for line in lines:
        if line["id"] not in label_of and by_time:
            nearest = min(by_time, key=lambda x: abs(x["start"] - line["start"]))
            label_of[line["id"]] = label_of[nearest["id"]]

    cluster_tags: dict[int, str] = {}
    for k in range(count):
        tag = overrides.get(str(k)) or ("streamer" if k == streamer else "guest")
        cluster_tags[k] = tag

    media_lines: dict[int, int] = {}
    if reference is not None:
        streamer_sims = {i: float(embeddings[i] @ reference) for i in ids}
        media_lines = repeated_media_lines(lines, streamer_sims)

    tts_hits: dict[int, int] = {}
    for line in lines:
        k = label_of.get(line["id"])
        tag = cluster_tags.get(k, "streamer")
        entry: dict = {"id": line["id"], "cluster": k, "tag": tag}
        if line["id"] in embeddings and voice is not None:
            sim = float(embeddings[line["id"]] @ voice)
            entry["voice_sim"] = round(sim, 3)
            # Per-line correction: a strong match to the enrolled voice is the streamer even if
            # clustering put the line elsewhere (e.g. shouting).
            if tag == "guest" and sim >= STREAMER_THRESHOLD + 0.1 and str(k) not in overrides:
                entry["tag"], entry["reason"] = "streamer", "matches enrolled voice"
        if line["id"] in media_lines and str(k) not in overrides:
            entry["tag"] = "media"
            entry["reason"] = (f"repeats {media_lines[line['id']]}x elsewhere in a different "
                               "voice (game or video audio, song lyrics)")
        if entry["tag"] not in ("streamer", "media") and (hit := matches_chat(
            line["text"], line["start"], chat
        )):
            entry["tag"], entry["reason"] = "tts", f"repeats chat from {hit.user} at {hit.t:.0f}s"
            tts_hits[k] = tts_hits.get(k, 0) + 1
        result["lines"].append(entry)

    for k in range(count):
        members = [i for i, label in label_of.items() if label == k]
        seconds = sum(durations[i] for i in members)
        # A voice that mostly repeats chat is a TTS voice: tag all its lines.
        if str(k) not in overrides and tts_hits.get(k, 0) >= max(2, len(members) // 2):
            cluster_tags[k] = "tts"
            for entry in result["lines"]:
                if entry["cluster"] == k and entry["tag"] == "guest":
                    entry["tag"] = "tts"
        result["clusters"].append({
            "id": k,
            "tag": cluster_tags[k],
            "lines": len(members),
            "seconds": round(seconds, 1),
            "voice_sim": None if voice_sims is None else round(float(voice_sims[k]), 3),
        })

    if voice is None:
        result["enrollment_suggestion"] = {"cluster": 0, "seconds": result["clusters"][0]["seconds"]}
    return result
