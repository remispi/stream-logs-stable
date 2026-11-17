"""Filler words (um, uh, ...) and "speech the ASR didn't write down" gaps."""

from __future__ import annotations

import re

# Hesitation sounds only. Discourse markers ("like", "you know") need context, so the LLM handles
# those later.
FILLERS = {
    "um", "umm", "uhm", "uh", "uhh", "er", "erm", "ah", "ahh", "eh", "hmm", "hm", "mm", "mmm",
}
PROBABLE_FILLER = "[filler?]"

# A stretch of VAD speech with no word on it is probably a filler the ASR dropped. Longer ones are
# usually laughter, singing or unclear speech, kept as "unrecognised" for later stages.
MIN_GAP_FILLER = 0.25
MAX_GAP_FILLER = 1.2
WORD_PAD = 0.04

_STRIP = re.compile(r"^[\s\[\(\"'.,!?;:-]+|[\s\]\)\"'.,!?;:-]+$")


def normalize_filler(token: str) -> str | None:
    """'[UM]' / ' Um,' / 'uhh...' -> 'um' / 'um' / 'uhh'; None if it isn't a filler."""
    core = _STRIP.sub("", token).lower()
    return core if core in FILLERS else None


def uncovered_speech(
    speech: list[tuple[float, float]], words: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Parts of the VAD speech regions not covered by any (padded) word span."""
    covered: list[tuple[float, float]] = []
    for start, end in sorted((s - WORD_PAD, e + WORD_PAD) for s, e in words):
        if covered and start <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], end))
        else:
            covered.append((start, end))
    gaps = []
    index = 0
    for region_start, region_end in sorted(speech):
        cursor = region_start
        while index < len(covered) and covered[index][1] <= cursor:
            index += 1
        probe = index
        while probe < len(covered) and covered[probe][0] < region_end:
            word_start, word_end = covered[probe]
            if word_start > cursor:
                gaps.append((cursor, min(word_start, region_end)))
            cursor = max(cursor, word_end)
            probe += 1
        if cursor < region_end:
            gaps.append((cursor, region_end))
    return [(start, end) for start, end in gaps if end - start > 0]


def insert_gap_fillers(
    lines: list[dict],
    speech: list[tuple[float, float]],
    extra_spans: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Add `[filler?]` words for short unworded speech inside or next to a line.

    `extra_spans` also count as covered: pass Whisper's original word spans. CTC alignment gives
    words short, spiky spans, and on a real 3.7 h VOD the gaps between them alone produced 710
    filler candidates mid-sentence; counting Whisper's wider spans too left 263.

    Returns the longer unworded regions (laughter, unclear speech) as `{"start", "end"}` dicts.
    """
    spans = [(w["start"], w["end"]) for line in lines for w in line["words"]]
    spans += extra_spans or []
    unrecognised = []
    for start, end in uncovered_speech(speech, spans):
        length = end - start
        if length < MIN_GAP_FILLER:
            continue
        line = _line_near(lines, start, end) if length <= MAX_GAP_FILLER else None
        if line is not None and (filler := _filler_next_to(line, start, end)):
            # CTC aligners mark fillers as short spikes; the rest of the sound is this gap.
            filler["start"] = round(min(filler["start"], start), 3)
            filler["end"] = round(max(filler["end"], end), 3)
            continue
        if line is None:
            unrecognised.append({"start": round(start, 3), "end": round(end, 3)})
            continue
        line["words"].append(
            {"w": PROBABLE_FILLER, "start": round(start, 3), "end": round(end, 3), "p": 0.0,
             "filler": True, "probable": True}
        )
        line["words"].sort(key=lambda w: w["start"])
        line["start"] = min(line["start"], round(start, 3))
        line["end"] = max(line["end"], round(end, 3))
    return unrecognised


def _filler_next_to(line: dict, start: float, end: float, reach: float = 0.4) -> dict | None:
    for word in line["words"]:
        if word["filler"] and not word.get("probable") and (
            0 <= start - word["end"] <= reach or 0 <= word["start"] - end <= reach
        ):
            # Only if no other word sits between the filler and the gap.
            low, high = min(word["end"], start), max(word["start"], end)
            between = [w for w in line["words"] if w is not word and low < w["start"] < high]
            if not between:
                return word
    return None


def _line_near(lines: list[dict], start: float, end: float, reach: float = 0.5) -> dict | None:
    """The line the gap sits in, or the closest line ending/starting within `reach` seconds."""
    best, best_distance = None, reach
    for line in lines:
        if line["start"] <= start and end <= line["end"]:
            return line
        distance = max(line["start"] - end, start - line["end"], 0.0)
        if distance <= best_distance:
            best, best_distance = line, distance
    return best
