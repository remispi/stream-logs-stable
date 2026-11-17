"""Chat logs from different tools, normalised to one message list.

Supported inputs:
- TwitchDownloader chat JSON (`TwitchDownloaderCLI chatdownload`),
- yt-dlp YouTube live-chat replay (`*.live_chat.json`, one JSON object per line),
- Chatterino logs (`channel-YYYY-MM-DD.log`, wall-clock times; needs the stream start time).

Times are seconds from the start of the VOD. Subs, gifts, raids, bits and donations are tagged
with `kind` so later stages can treat them as events.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

ChatKind = Literal["message", "sub", "gift", "raid", "bits", "donation", "member", "bot"]

BOTS = {
    "streamelements", "streamlabs", "nightbot", "moobot", "fossabot", "wizebot", "sery_bot",
    "botrixoficial", "soundalerts", "kofistreambot", "streamstickers", "pokemoncommunitygame",
}

_SUB = re.compile(r"\bsubscribed (?:at tier \d|with prime|for \d+ months?)", re.IGNORECASE)
_GIFT = re.compile(r"\b(?:gifted|is gifting) (?:a |\d+ )?(?:tier \d )?subs?\b", re.IGNORECASE)
_RAID = re.compile(r"\braid(?:ed|ing)?\b.*\b\d+\b|\b\d+ raiders\b", re.IGNORECASE)
_DONATION = re.compile(r"(?:tipped|donated|donation|tip of)\b.*?[$€£]\s?\d|[$€£]\s?\d.*\b(?:tip|donation)", re.IGNORECASE)
_CHEER = re.compile(r"\bcheer\d+\b", re.IGNORECASE)


@dataclass
class ChatMessage:
    t: float
    user: str
    text: str
    kind: ChatKind = "message"
    emotes: list[str] = field(default_factory=list)
    bits: int = 0
    amount: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        # Keep chat.json compact: drop empty defaults.
        for key, empty in (("emotes", []), ("bits", 0), ("amount", None), ("kind", "message")):
            if data[key] == empty:
                del data[key]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ChatMessage:
        return cls(**data)


def classify_message(user: str, text: str, bits: int = 0) -> ChatKind:
    lowered = user.lower()
    if bits > 0 or _CHEER.search(text):
        return "bits"
    if _GIFT.search(text):
        return "gift"
    if _SUB.search(text):
        return "sub"
    if lowered in BOTS:
        if _RAID.search(text):
            return "raid"
        if _DONATION.search(text):
            return "donation"
        return "bot"
    return "message"


# ---------------------------------------------------------------------- TwitchDownloader


def parse_twitchdownloader(data: dict) -> list[ChatMessage]:
    messages = []
    for comment in data.get("comments", []):
        body = comment.get("message", {})
        commenter = comment.get("commenter") or {}
        user = commenter.get("display_name") or commenter.get("name") or "?"
        text = body.get("body", "")
        bits = int(body.get("bits_spent") or 0)
        emotes = [
            frag["text"] for frag in body.get("fragments") or [] if frag.get("emoticon")
        ]
        messages.append(
            ChatMessage(
                t=float(comment.get("content_offset_seconds", 0.0)),
                user=user,
                text=text,
                kind=classify_message(user, text, bits),
                emotes=emotes,
                bits=bits,
            )
        )
    return messages


# ---------------------------------------------------------------------- YouTube


def _runs_text(runs: list[dict]) -> tuple[str, list[str]]:
    parts, emotes = [], []
    for run in runs:
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            emoji = run["emoji"]
            shortcut = (emoji.get("shortcuts") or [emoji.get("emojiId", "")])[0]
            parts.append(shortcut)
            emotes.append(shortcut)
    return "".join(parts), emotes


def parse_youtube_live_chat(lines: list[str]) -> list[ChatMessage]:
    messages = []
    for line in lines:
        if not line.strip():
            continue
        replay = json.loads(line).get("replayChatItemAction", {})
        offset_ms = float(replay.get("videoOffsetTimeMsec", 0))
        for action in replay.get("actions", []):
            item = action.get("addChatItemAction", {}).get("item", {})
            for renderer_name, renderer in item.items():
                user = renderer.get("authorName", {}).get("simpleText", "?")
                text, emotes = _runs_text(renderer.get("message", {}).get("runs", []))
                amount = None
                if renderer_name == "liveChatTextMessageRenderer":
                    kind = classify_message(user, text)
                elif renderer_name in ("liveChatPaidMessageRenderer", "liveChatPaidStickerRenderer"):
                    kind = "donation"
                    amount = renderer.get("purchaseAmountText", {}).get("simpleText")
                elif renderer_name == "liveChatMembershipItemRenderer":
                    kind = "member"
                    if not text:
                        text, _ = _runs_text(renderer.get("headerSubtext", {}).get("runs", []))
                else:
                    continue
                messages.append(
                    ChatMessage(offset_ms / 1000, user, text, kind, emotes, amount=amount)
                )
    return messages


# ---------------------------------------------------------------------- Chatterino

_CHATTERINO_LINE = re.compile(r"^\[(\d{1,2}):(\d{2}):(\d{2})\]\s+(.*)$")
_CHATTERINO_MESSAGE = re.compile(r"^([A-Za-z0-9_]+)(?:\s*\([^)]*\))?:\s(.*)$")
_LOG_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_chatterino(paths: list[Path], vod_start: datetime) -> list[ChatMessage]:
    """`vod_start` is the local wall-clock time the VOD starts (Chatterino logs local time)."""
    if vod_start.tzinfo is not None:  # "...T19:02:11+01:00" or "...Z": as this machine's time
        vod_start = vod_start.astimezone().replace(tzinfo=None)
    messages = []
    for path in sorted(paths):
        match = _LOG_DATE.search(path.name)
        if match is None:
            raise ValueError(f"{path.name}: expected a date (YYYY-MM-DD) in the log file name")
        day = datetime.strptime(match.group(1), "%Y-%m-%d")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _CHATTERINO_LINE.match(line)
            if parsed is None:
                continue
            hours, minutes, seconds, rest = parsed.groups()
            stamp = day + timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds))
            t = (stamp - vod_start).total_seconds()
            if as_message := _CHATTERINO_MESSAGE.match(rest):
                user, text = as_message.groups()
                messages.append(ChatMessage(t, user, text, classify_message(user, text)))
            else:
                # System lines: "user subscribed at Tier 1...", "X is raiding with 34 viewers".
                user = rest.split(" ", 1)[0]
                kind = classify_message(user, rest)
                if kind == "message":
                    kind = "raid" if _RAID.search(rest) else "bot"
                messages.append(ChatMessage(t, user, rest, kind))
    return messages


# ---------------------------------------------------------------------- dispatch


def detect_format(path: Path) -> str:
    if path.is_dir() or path.suffix.lower() in (".log", ".txt"):
        return "chatterino"
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    if '"replayChatItemAction"' in first:
        return "youtube"
    return "twitchdownloader"


def load_chat(
    path: Path, *, vod_start: datetime | None = None, duration: float | None = None
) -> tuple[str, list[ChatMessage]]:
    """Parse any supported chat file (or a folder of Chatterino logs). Returns (format, messages)."""
    fmt = detect_format(path)
    if fmt == "youtube":
        messages = parse_youtube_live_chat(path.read_text(encoding="utf-8").splitlines())
    elif fmt == "twitchdownloader":
        messages = parse_twitchdownloader(json.loads(path.read_text(encoding="utf-8")))
    else:
        if vod_start is None:
            raise ValueError(
                "Chatterino logs use wall-clock times: pass the stream start time (--chat-start)"
            )
        files = sorted(path.glob("*.log")) if path.is_dir() else [path]
        messages = parse_chatterino(files, vod_start)
    limit = float("inf") if duration is None else duration + 1
    messages = sorted((m for m in messages if 0 <= m.t <= limit), key=lambda m: m.t)
    return fmt, messages


def save_chat(dest: Path, fmt: str, origin: str, messages: list[ChatMessage]) -> None:
    from stream_log.engine.project import write_json

    write_json(
        dest,
        {"format": fmt, "origin": origin, "messages": [m.to_dict() for m in messages]},
        indent=None,
    )


def read_saved_chat(path: Path) -> list[ChatMessage]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ChatMessage.from_dict(m) for m in data["messages"]]
