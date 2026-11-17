"""Stream events: cheap detectors -> LLM pass per chapter -> grounding -> merge.

Every LLM event must quote the transcript (or chat) near its timestamp; events whose quotes can't
be found are dropped, so the LLM can't invent things that didn't happen.
"""

from __future__ import annotations

import itertools
import re
from difflib import SequenceMatcher

import numpy as np

from stream_log.engine.chapters import transcript_excerpt
from stream_log.engine.ingest.chat import BOTS, ChatMessage
from stream_log.engine.llm import LlmClient, LlmError
from stream_log.engine.project import clock
from stream_log.engine.signals.chat import VIBES
from stream_log.engine.signals.common import peaks, robust_z, smooth

CATEGORIES = ["structure", "gameplay", "community", "conversation", "announcement", "chat",
              "health", "custom"]
IMPORTANCE = ["minor", "notable", "key"]
# Events are placed at the quote's real time, so a wider window only lets in real quotes whose
# timestamp the model got wrong (Ministral was often 1-2 min off on a 3.7 h VOD).
GROUND_WINDOW = 120.0
GROUND_MATCH = 0.9  # share of the quote's words found, in order, in up to GROUND_SPAN lines
GROUND_SPAN = 4
GROUND_MIN_WORDS = 2
GROUND_GENERIC = 3  # a phrase said in more lines than this proves nothing about one moment
MERGE_WINDOW = 30.0


def event(start: float, end: float, category: str, type_: str, title: str, *, source: str,
          description: str = "", evidence: list[dict] | None = None, confidence: float = 0.6,
          importance: str = "minor", people: list[str] | None = None, **extra) -> dict:
    return {
        "start": round(float(start), 1), "end": round(float(max(end, start)), 1),
        "category": category, "type": type_, "title": title, "description": description,
        "sources": [source], "evidence": evidence or [], "confidence": round(confidence, 2),
        "importance": importance, "people": people or [], **extra,
    }


# ---------------------------------------------------------------------- cheap detectors


def structure_events(segments: list[dict], twitch_chapters: list[dict], speech: np.ndarray,
                     duration: float) -> list[dict]:
    out = []
    talking = smooth(speech, 15) > 0.1
    for segment in segments:
        if segment["mode"] != "static":
            continue
        # Only the silent parts of a static-looking picture are break screens.
        for start, end in _runs(~talking, int(segment["start"]), int(segment["end"]), 30):
            if start < 60:
                type_, title = "starting_soon", "Starting soon screen"
            elif end > duration - 60:
                type_, title = "ending", "Ending screen"
            else:
                type_, title = "brb", "Break (BRB screen)"
            out.append(event(start, end, "structure", type_, title, source="modes",
                             confidence=segment["confidence"], importance="notable"))
    for previous, chapter in itertools.pairwise(twitch_chapters):
        if chapter.get("title") != previous.get("title") and chapter.get("start"):
            out.append(event(chapter["start"], chapter["start"] + 1, "structure", "game_switch",
                             f"Switched to {chapter['title']}", source="twitch_chapters",
                             confidence=0.95, importance="key"))
    # Long silences outside static screens.
    quiet = speech < 0.05
    start = None
    for t, q in enumerate(np.r_[quiet, False]):
        if q and start is None:
            start = t
        elif not q and start is not None:
            if t - start >= 180 and not any(
                e["start"] <= start + 30 and e["end"] >= t - 30 for e in out
            ):
                out.append(event(start, t, "structure", "silence",
                                 f"{(t - start) // 60} min without talking", source="transcript",
                                 confidence=0.8))
            start = None
    return out


def _runs(mask: np.ndarray, lo: int, hi: int, min_length: int) -> list[tuple[int, int]]:
    """[start, end) runs of True in mask[lo:hi] lasting at least `min_length` seconds."""
    runs, start = [], None
    for t in range(lo, min(hi, len(mask)) + 1):
        value = t < min(hi, len(mask)) and bool(mask[t])
        if value and start is None:
            start = t
        elif not value and start is not None:
            if t - start >= min_length:
                runs.append((start, t))
            start = None
    return runs


# "$25", "$25.00", "€2,50", "$1,000.00", "€1.000,50": thousands grouped, or cents after the point
_AMOUNT = re.compile(r"[$€£]\s?(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)")
_GIFT_COUNT = re.compile(r"\b(?:gifting|gifted) (\d+)\b", re.IGNORECASE)
_RAID_COUNT = re.compile(r"\b(\d+)\s*(?:viewers|raiders)\b", re.IGNORECASE)
_BOT_SUBJECT = re.compile(r"^@?([A-Za-z0-9_]{3,25})\b(?= (?:just|has|is|tipped|donated|raided))")


def _amount_value(number: str) -> float:
    """An `_AMOUNT` number as a float: a point or comma before the last 1-2 digits is the
    decimal point, the others group thousands."""
    whole, cents = number, "0"
    if decimal := re.search(r"[.,](\d{1,2})$", number):
        whole, cents = number[: decimal.start()], decimal.group(1)
    return float(re.sub(r"\D", "", whole) + "." + cents)


def _who(message: ChatMessage) -> str:
    """Bots announce other people's donations/raids: credit the person named in the text."""
    if message.user.lower() in BOTS and (match := _BOT_SUBJECT.match(message.text)):
        return match.group(1)
    return message.user


def community_events(chat: list[ChatMessage], tts_lines: list[dict]) -> list[dict]:
    out = []
    for message in chat:
        kind = message.kind
        if kind not in ("sub", "gift", "raid", "bits", "donation", "member"):
            continue
        evidence = [{"t": clock(message.t), "quote": message.text[:160], "source": "chat"}]
        who = _who(message)
        importance, title = "minor", f"{kind.title()} from {who}"
        if kind == "gift":
            count = int(m.group(1)) if (m := _GIFT_COUNT.search(message.text)) else 1
            title = f"{who} gifted {count} sub{'s' if count != 1 else ''}"
            importance = "key" if count >= 5 else "notable"
        elif kind == "raid":
            count = int(m.group(1)) if (m := _RAID_COUNT.search(message.text)) else None
            title = f"Raid from {who} ({count} viewers)" if count else f"Raid from {who}"
            importance = "key"
        elif kind == "bits":
            title = f"{message.bits or 'Some'} bits from {who}"
            importance = "notable" if message.bits >= 500 else "minor"
        elif kind == "donation":
            amount = message.amount or (m.group(0) if (m := _AMOUNT.search(message.text)) else "")
            title = f"Donation {amount} from {who}".replace("  ", " ")
            found = _AMOUNT.search(amount or "")
            value = _amount_value(found.group(1)) if found else 0.0
            importance = "key" if value >= 10 else "notable"
        elif kind == "sub":
            title = f"{who} subscribed"
        out.append(event(message.t, message.t + 1, "community", kind, title, source="chat_notice",
                         evidence=evidence, confidence=0.95, importance=importance,
                         people=[who]))
    for line in tts_lines:
        out.append(event(line["start"], line["end"], "community", "tts", "TTS message read out",
                         source="speaker_tags", description=line["text"][:200],
                         evidence=[{"t": clock(line["start"]), "quote": line["text"][:160],
                                    "source": "transcript"}], confidence=0.8))
    return merge_bursts(out)


def merge_bursts(events: list[dict], window: float = 60.0) -> list[dict]:
    """Collapse gift/sub storms into one event per type."""
    out: list[dict] = []
    for item in sorted(events, key=lambda e: e["start"]):
        last = next((e for e in reversed(out) if e["type"] == item["type"]), None)
        if (last and item["type"] in ("sub", "gift", "bits") and item["start"] - last["end"]
                <= window):
            last["end"] = item["end"]
            last["people"] = sorted(set(last["people"]) | set(item["people"]))
            last["evidence"] = (last["evidence"] + item["evidence"])[:5]
            last["count"] = last.get("count", 1) + 1
            last["title"] = f"{last['count']} {item['type']}s ({', '.join(last['people'][:3])}…)"
            if last["count"] >= 5:
                last["importance"] = "key"
        else:
            out.append(item)
    return out


def chat_moment_events(signals: dict[str, np.ndarray], reliability: float) -> list[dict]:
    if reliability < 0.15:
        return []  # too few chatters for chat peaks to mean anything
    rate_z = robust_z(signals["chat_rate"])
    shift_z = robust_z(signals["chat_vibe_shift"])
    rate_z, shift_z = np.nan_to_num(rate_z), np.nan_to_num(shift_z)
    # A vibe shift only counts while chat is at least as busy as usual around it.
    score = np.maximum(rate_z, np.where(rate_z >= 0, shift_z, 0))
    out = []
    for t in peaks(score, min_gap=90, threshold=3.0, limit=40):
        lo, hi = max(0, t - 10), t + 10
        mix = {v: float(np.mean(signals[f"vibe_{v}"][lo:hi])) for v in VIBES}
        vibe = max(mix, key=mix.get)
        importance = "key" if score[t] >= 6 else "notable"
        out.append(event(t - 5, t + 15, "chat", f"chat_{vibe}",
                         f"Chat {vibe} spike", source="chat_signals",
                         confidence=min(0.4 + score[t] / 15, 0.95), importance=importance,
                         score=round(float(score[t]), 2)))
    return out


def gameplay_events(visual_probs: dict[str, np.ndarray], step: int, segments: list[dict]
                    ) -> list[dict]:
    titles = {"death": "Death / game over", "victory": "Victory screen", "boss": "Boss fight",
              "loading": "Loading screen"}
    out = []
    gaming = [(s["start"], s["end"]) for s in segments if s["mode"] == "gaming"]
    for name, probs in visual_probs.items():
        for i in np.flatnonzero(probs > 0.6):
            t = int(i) * step
            if not any(a <= t < b for a, b in gaming):
                continue
            if out and out[-1]["type"] == name and t - out[-1]["end"] <= step:
                out[-1]["end"] = t + step
                continue
            out.append(event(t, t + step, "gameplay", name, titles[name], source="siglip",
                             confidence=float(probs[i]) * 0.8))
    return [e for e in out if not (e["type"] == "loading" and e["end"] - e["start"] < 20)]


def health_events(signals: dict[str, np.ndarray], frozen: list[tuple[int, int]],
                  segments: list[dict], chat: list[ChatMessage]) -> list[dict]:
    out = []
    clip = signals["clipping"] > 0.001
    start = None
    for t, c in enumerate(np.r_[clip, False]):
        if c and start is None:
            start = t
        elif not c and start is not None:
            if t - start >= 3:
                out.append(event(start, t, "health", "audio_clipping", "Audio clipping",
                                 source="audio", confidence=0.9, importance="notable"))
            start = None
    static = [(s["start"], s["end"]) for s in segments if s["mode"] == "static"]
    for a, b in frozen:
        if not any(sa <= a and b <= sb for sa, sb in static):
            out.append(event(a, b, "health", "frozen_video", f"Video frozen for {b - a} s",
                             source="visual", confidence=0.7, importance="notable"))
    complaint = re.compile(r"\b(?:no audio|can'?t hear|mic(?:'s)? (?:is )?(?:off|muted)|"
                           r"no sound|you(?:'re)? muted|frozen|stream (?:is )?lagging)\b",
                           re.IGNORECASE)
    hits = [m for m in chat if m.kind == "message" and complaint.search(m.text)]
    for m in hits:
        nearby = [h for h in hits if abs(h.t - m.t) <= 60]
        if len(nearby) >= 2 and not any(abs(e["start"] - m.t) < 120 and e["type"] ==
                                        "chat_reports_problem" for e in out):
            out.append(event(m.t, m.t + 60, "health", "chat_reports_problem",
                             "Chat reports a stream problem", source="chat",
                             evidence=[{"t": clock(h.t), "quote": h.text[:120], "source": "chat"}
                                       for h in nearby[:3]], confidence=0.75,
                             importance="notable"))
    return out


_ANNOUNCE = re.compile(
    r"\b(?:i(?:'ll| will) (?:be )?(?:stream|live)|stream(?:ing)? (?:tomorrow|next week|on "
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day)|schedule|giveaway|sponsor(?:ed)?|collab|"
    r"merch|when we hit \d+|at \d+ (?:subs|followers))\b",
    re.IGNORECASE,
)


def announcement_events(lines: list[dict], tags: dict[int, str]) -> list[dict]:
    out = []
    for line in lines:
        if tags.get(line["id"], "streamer") != "streamer":
            continue
        if _ANNOUNCE.search(line["text"]):
            out.append(event(line["start"], line["end"], "announcement", "announcement",
                             "Announcement / promise", source="transcript_regex",
                             description=line["text"][:200], follow_up=True, confidence=0.5,
                             evidence=[{"t": clock(line["start"]), "quote": line["text"][:160],
                                        "source": "transcript"}]))
    return out


def conversation_events(interactions: list[dict], speakers: dict, lines: list[dict]
                        ) -> list[dict]:
    out = []
    # Chat Q&A: 3+ interactions within 2 minutes.
    times = sorted(i["t"] for i in interactions)
    i = 0
    while i < len(times):
        j = i
        while j + 1 < len(times) and times[j + 1] - times[i] <= 120:
            j += 1
        if j - i >= 2:
            people = sorted({x["user"] for x in interactions if times[i] <= x["t"] <= times[j]})
            out.append(event(times[i], times[j] + 10, "conversation", "chat_qa",
                             f"Talking with chat ({len(people)} chatters)", source="chat_matching",
                             people=people, confidence=0.7, importance="notable"))
        i = j + 1
    # Guests: a guest voice talking for a minute or more.
    by_id = {line["id"]: line for line in lines}
    for cluster in speakers.get("clusters", []):
        if cluster["tag"] != "guest" or cluster["seconds"] < 60:
            continue
        member = [by_id[e["id"]] for e in speakers["lines"] if e["cluster"] == cluster["id"]
                  and e["id"] in by_id]
        if member:
            out.append(event(member[0]["start"], member[-1]["end"], "conversation", "guest",
                             f"Guest voice #{cluster['id']} ({cluster['seconds'] / 60:.0f} min)",
                             source="speaker_tags", confidence=0.6, importance="notable"))
    return out


def custom_events(profile: dict, lines: list[dict], chat: list[ChatMessage]) -> list[dict]:
    """Profile `custom_events`: [{"name", "pattern", "source": "transcript"|"chat"}]."""
    out = []
    for rule in profile.get("custom_events", []):
        pattern = re.compile(rule["pattern"], re.IGNORECASE)
        source = rule.get("source", "transcript")
        items = ([(ln["start"], ln["text"]) for ln in lines] if source == "transcript"
                 else [(m.t, m.text) for m in chat])
        for t, text in items:
            if pattern.search(text):
                out.append(event(t, t + 3, "custom", rule["name"], rule["name"], source=source,
                                 evidence=[{"t": clock(t), "quote": text[:160], "source": source}],
                                 confidence=0.8))
    return out


# ---------------------------------------------------------------------- LLM pass + grounding

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "H:MM:SS"},
                    "end_time": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "type": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "quote": {"type": "string", "description": "exact words from the transcript"},
                    "importance": {"type": "string", "enum": IMPORTANCE},
                    "people": {"type": "array", "items": {"type": "string"}},
                    "follow_up": {"type": "boolean"},
                },
                "required": ["time", "end_time", "category", "type", "title", "description",
                             "quote", "importance", "people", "follow_up"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}

EVENT_SYSTEM = (
    "You extract a factual event log from a livestream transcript for the streamer. "
    "Report things that happened: topics and stories, opinions, chat Q&A, guests, gameplay "
    "highs and lows, announcements and promises (schedule, collabs, sponsor reads, giveaways - "
    "mark follow_up true), community moments. Each event needs a short exact quote copied from "
    "the transcript at that time. Do not guess; if nothing notable happens, return no events. "
    "Categories: conversation = topics, stories, opinions, talking with chat or guests; "
    "gameplay = what happens in the game; announcement = schedule, collabs, sponsors, giveaways, "
    "promises; community = subs, raids, donations, bits only; chat = what chat does; "
    "structure = start, breaks, game switches, ending; health = technical problems. "
    "Importance: key = a viewer of the VOD must not miss it; notable = worth a line in a recap; "
    "minor = background detail. Announcements and promises are at least notable."
)


def parse_clock(text: str) -> float | None:
    """'1:02:03' -> 3723. Some models write a range ('0:36:35-0:36:57'); use its start."""
    match = re.search(r"\d+(?::\d+(?:\.\d+)?)+|\d+(?:\.\d+)?", text)
    if match is None:
        return None
    parts = match.group(0).split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def find_quote(quote: str, t: float, lines: list[dict], chat: list[ChatMessage],
               window: float = GROUND_WINDOW) -> tuple[float, str] | None:
    """Locate `quote` in the transcript (or chat) within `window` s of `t`. None if not found.

    Quotes stitched from several lines with '...' are grounded only if every piece is found;
    the event then sits at the first piece. Pieces that are too short or too common to prove
    anything ("i mean", "let's go") are skipped; a quote made only of those is not grounded.
    Measured on a 3.7 h VOD: 287/291 real quotes kept; moved, spliced,
    made-up and generic quotes dropped; 13 % of invented endings and 75 % of single swapped words
    still pass (catching those is the panel's cross-check).
    """
    pieces = [p for p in re.split(r"\.\.\.|…", quote) if _distinctive(_norm(p), lines, chat)]
    if not pieces:
        return None
    found = [_find_piece(_norm(p), t, lines, chat, window) for p in pieces]
    if any(f is None for f in found):
        return None
    return min(found)


def _distinctive(target: str, lines: list[dict], chat: list[ChatMessage]) -> bool:
    if len(target.split()) < GROUND_MIN_WORDS:
        return False
    said = 0
    for text in itertools.chain((ln["text"] for ln in lines), (m.text for m in chat)):
        if target in _norm(text):
            said += 1
            if said > GROUND_GENERIC:
                return False
    return True


def _find_piece(target: str, t: float, lines: list[dict], chat: list[ChatMessage],
                window: float) -> tuple[float, str] | None:
    words = target.split()
    near = [ln for ln in lines if abs(ln["start"] - t) <= window]
    # Quotes often span several lines.
    candidates = [(near[i]["start"], " ".join(ln["text"] for ln in near[i:i + size]), "transcript")
                  for size in range(1, GROUND_SPAN + 1) for i in range(len(near) - size + 1)]
    candidates += [(m.t, m.text, "chat") for m in chat if abs(m.t - t) <= window]
    for start, text, source in candidates:
        norm = _norm(text)
        if target in norm:
            return start, source
        matcher = SequenceMatcher(None, words, norm.split(), autojunk=False)
        if sum(b.size for b in matcher.get_matching_blocks()) / len(words) >= GROUND_MATCH:
            return start, source
    return None


def llm_events(llm: LlmClient, chapters: list[dict], lines: list[dict], tags: dict[int, str],
               chat: list[ChatMessage], channel_note: str = "", errors: list[str] | None = None
               ) -> tuple[list[dict], int]:
    """Returns (grounded events, number dropped by the grounding check)."""
    kept, dropped = [], 0
    for chapter in chapters:
        excerpt = transcript_excerpt(lines, tags, chapter["start"], chapter["end"], limit=16000)
        if not excerpt.strip():
            continue
        prompt = (f"{channel_note}\nChapter '{chapter['title']}' ({chapter['mode']}), "
                  f"{clock(chapter['start'])}-{clock(chapter['end'])}.\n\nTranscript:\n{excerpt}")
        try:
            result = llm.chat_json(EVENT_SYSTEM, prompt, EVENT_SCHEMA, max_tokens=4096)
        except LlmError as error:
            if errors is not None:
                errors.append(f"events for {chapter['id']}: {error}")
            continue
        for raw in result.get("events", []):
            t = parse_clock(str(raw.get("time", "")))
            if t is None:
                dropped += 1
                continue
            grounded = find_quote(str(raw.get("quote", "")), t, lines, chat)
            if grounded is None:
                dropped += 1
                continue
            start, source = grounded
            end = parse_clock(str(raw.get("end_time", ""))) or start + 10
            category = raw.get("category") if raw.get("category") in CATEGORIES else "conversation"
            kept.append(event(
                start, max(end, start + 1), category, str(raw.get("type") or "topic"),
                str(raw.get("title") or "")[:120], source="llm",
                description=str(raw.get("description") or "")[:400],
                evidence=[{"t": clock(start), "quote": str(raw["quote"])[:200], "source": source}],
                confidence=0.7,
                importance=raw.get("importance") if raw.get("importance") in IMPORTANCE else
                "minor",
                people=[str(p) for p in raw.get("people") or []][:5],
                follow_up=bool(raw.get("follow_up")),
            ))
    return kept, dropped


# ---------------------------------------------------------------------- merge + finish


def merge_events(events: list[dict]) -> list[dict]:
    """Same category + type overlapping in time -> one event with all sources and evidence."""
    out: list[dict] = []
    for item in sorted(events, key=lambda e: (e["start"], e["category"])):
        match = next(
            (e for e in out if e["category"] == item["category"]
             and (e["type"] == item["type"] or "llm" in item["sources"] or "llm" in e["sources"])
             and item["start"] <= e["end"] + MERGE_WINDOW and e["start"] <= item["end"]
             + MERGE_WINDOW and _compatible(e, item)),
            None,
        )
        if match is None:
            out.append(dict(item))
            continue
        match["sources"] = sorted(set(match["sources"]) | set(item["sources"]))
        match["evidence"] = (match["evidence"] + item["evidence"])[:6]
        match["people"] = sorted(set(match["people"]) | set(item["people"]))
        match["end"] = max(match["end"], item["end"])
        match["confidence"] = round(min(0.99, max(match["confidence"], item["confidence"]) + 0.1), 2)
        if IMPORTANCE.index(item["importance"]) > IMPORTANCE.index(match["importance"]):
            match["importance"] = item["importance"]
        if "llm" in item["sources"] and item["description"]:
            match["title"], match["description"] = item["title"], item["description"]
        match["follow_up"] = match.get("follow_up", False) or item.get("follow_up", False)
    return out


def _compatible(a: dict, b: dict) -> bool:
    # Two different LLM topics in the same minute are separate events.
    return not ("llm" in a["sources"] and "llm" in b["sources"] and a["type"] != b["type"])


def finish(events: list[dict], chapters: list[dict]) -> list[dict]:
    events = sorted(events, key=lambda e: e["start"])
    for index, item in enumerate(events):
        item["id"] = f"ev_{index + 1:04d}"
        item["chapter_id"] = next((c["id"] for c in chapters
                                   if c["start"] <= item["start"] < c["end"]), None)
        item["time"] = clock(item["start"])
    return events
