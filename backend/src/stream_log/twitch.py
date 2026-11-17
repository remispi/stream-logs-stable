"""Twitch: an account's past broadcasts, one VOD's details, its chat, and the video itself.

Public VODs only, without signing in. Lists, details and chat come from Twitch's GraphQL endpoint
(the one Twitch's own apps and TwitchDownloader use); the video is downloaded with yt-dlp.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

GQL_URL = "https://gql.twitch.tv/gql"
# Twitch's own web client and its chat-replay query. Twitch changes these now and then; until a
# new release catches up, the environment variables below override them without a rebuild.
CLIENT_ID = os.environ.get("STREAM_LOG_TWITCH_CLIENT_ID") or "kd1unb4b3q4t58fwlpcbzcbnm76a8fp"
COMMENTS_QUERY_HASH = os.environ.get("STREAM_LOG_TWITCH_COMMENTS_HASH") \
    or "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
LOGIN = re.compile(r"^[A-Za-z0-9_]{3,25}$")
_VOD_URL = re.compile(r"twitch\.tv/(?:[^/?#]+/)?videos?/(\d{5,20})")
_VOD_ID = re.compile(r"^v?(\d{5,20})$")

Progress = Callable[[float, str], None]


class TwitchError(RuntimeError):
    pass


@dataclass
class Vod:
    id: str
    title: str
    created_at: str  # ISO 8601, UTC
    length: int  # seconds
    channel: str = ""
    game: str = ""
    views: int = 0
    chapters: list[dict] = field(default_factory=list)  # {start, end, title}: game changes

    @property
    def url(self) -> str:
        return f"https://www.twitch.tv/videos/{self.id}"

    @property
    def started(self) -> datetime:
        return datetime.fromisoformat(self.created_at)

    def to_json(self) -> dict:
        return asdict(self) | {"url": self.url}


def vod_id(text: str) -> str | None:
    """'https://www.twitch.tv/videos/123456789', 'v123456789' or '123456789' -> '123456789'."""
    text = text.strip()
    match = _VOD_URL.search(text) or _VOD_ID.match(text)
    return match.group(1) if match else None


def _post(body: object, retries: int = 4) -> object:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            GQL_URL, data=data,
            headers={"Client-ID": CLIENT_ID, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code in (400, 401, 403) and attempt == 0:
                raise TwitchError(
                    f"Twitch refused the request ({error.code}): its client ID may have changed "
                    "(set STREAM_LOG_TWITCH_CLIENT_ID to a current one)") from error
            if error.code < 500 and error.code != 429 or attempt == retries:
                raise TwitchError(f"Twitch answered {error.code}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries:
                raise TwitchError(f"can't reach Twitch: {error}") from error
        time.sleep(1.5 * 2 ** attempt)
    raise AssertionError("unreachable")


def _query(query: str, variables: dict) -> dict:
    answer = _post({"query": query, "variables": variables})
    if not isinstance(answer, dict):
        raise TwitchError("unexpected answer from Twitch")
    if answer.get("errors") and not answer.get("data"):
        raise TwitchError(answer["errors"][0].get("message", "Twitch error"))
    return answer["data"]


_VOD_FIELDS = "id title createdAt lengthSeconds viewCount game { name } owner { login displayName }"


def _vod(node: dict) -> Vod:
    owner = node.get("owner") or {}
    return Vod(id=str(node["id"]), title=node.get("title") or f"VOD {node['id']}",
               created_at=node["createdAt"], length=int(node.get("lengthSeconds") or 0),
               channel=owner.get("displayName") or owner.get("login") or "",
               game=(node.get("game") or {}).get("name") or "",
               views=int(node.get("viewCount") or 0))


def list_vods(login: str, limit: int = 30) -> list[Vod]:
    """The account's newest past broadcasts, newest first."""
    login = login.strip().lstrip("@")
    if not LOGIN.match(login):
        raise TwitchError(f"not a Twitch account name: {login!r}")
    data = _query(
        "query($login: String!, $first: Int!) { user(login: $login) { "
        f"videos(first: $first, type: ARCHIVE, sort: TIME) {{ edges {{ node {{ {_VOD_FIELDS} }} }} }}"
        " } }", {"login": login, "first": limit})
    user = data.get("user")
    if user is None:
        raise TwitchError(f"no Twitch account called {login}")
    return [_vod(edge["node"]) for edge in user["videos"]["edges"]]


def vod_info(id_: str) -> Vod:
    data = _query(
        f"query($id: ID!) {{ video(id: $id) {{ {_VOD_FIELDS} "
        "moments(momentRequestType: VIDEO_CHAPTER_MARKERS, first: 100) { edges { node { "
        "positionMilliseconds durationMilliseconds description } } } } }", {"id": id_})
    node = data.get("video")
    if node is None:
        raise TwitchError(f"Twitch VOD {id_} not found (deleted, expired or subscriber-only)")
    vod = _vod(node)
    for edge in (node.get("moments") or {}).get("edges") or []:
        moment = edge["node"]
        start = moment["positionMilliseconds"] / 1000
        vod.chapters.append({"start": start,
                             "end": start + (moment.get("durationMilliseconds") or 0) / 1000,
                             "title": moment.get("description") or ""})
    if not vod.chapters and vod.game:
        vod.chapters = [{"start": 0, "end": vod.length, "title": vod.game}]
    return vod


# ---------------------------------------------------------------------- chat


def _comment(node: dict) -> dict:
    """One GraphQL comment in TwitchDownloader's JSON shape, which the analysis reads."""
    message = node.get("message") or {}
    fragments = []
    for fragment in message.get("fragments") or []:
        emote = fragment.get("emote")
        fragments.append({"text": fragment.get("text") or "",
                          "emoticon": {"emoticon_id": emote.get("emoteID")} if emote else None})
    commenter = node.get("commenter") or {}
    return {
        "_id": node.get("id"),
        "created_at": node.get("createdAt"),
        "content_offset_seconds": node.get("contentOffsetSeconds") or 0,
        "commenter": {"display_name": commenter.get("displayName"),
                      "name": commenter.get("login")} if commenter else None,
        "message": {"body": "".join(f["text"] for f in fragments), "bits_spent": 0,
                    "fragments": fragments},
    }


def download_chat(vod: Vod, dest: Path, progress: Progress | None = None) -> int:
    """The VOD's whole chat replay into `dest` (TwitchDownloader JSON). Returns the message count."""
    comments: list[dict] = []
    seen: set[str] = set()
    variables: dict = {"videoID": vod.id, "contentOffsetSeconds": 0}
    last_cursor = None
    while True:
        answer = _post([{"operationName": "VideoCommentsByOffsetOrCursor", "variables": variables,
                         "extensions": {"persistedQuery": {
                             "version": 1, "sha256Hash": COMMENTS_QUERY_HASH}}}])
        first = answer[0] if isinstance(answer, list) and answer else answer
        if not isinstance(first, dict):
            raise TwitchError("unexpected answer from Twitch")
        errors = " ".join(str(e.get("message", "")) for e in first.get("errors") or [])
        if "PersistedQueryNotFound" in errors:
            raise TwitchError("Twitch no longer knows the chat query: set "
                              "STREAM_LOG_TWITCH_COMMENTS_HASH to its current hash")
        video = (first.get("data") or {}).get("video")
        if video is None or video.get("comments") is None:
            if not comments:
                raise TwitchError("Twitch didn't return this VOD's chat")
            break
        page = video["comments"]
        edges = page.get("edges") or []
        for edge in edges:
            node = edge["node"]
            if node.get("id") in seen:
                continue
            seen.add(node.get("id"))
            comments.append(_comment(node))
        if progress and comments and vod.length:
            offset = comments[-1]["content_offset_seconds"]
            progress(min(1.0, offset / vod.length), f"{len(comments)} messages")
        cursor = edges[-1]["cursor"] if edges else None
        if not (page.get("pageInfo") or {}).get("hasNextPage") or not cursor or cursor == last_cursor:
            break
        last_cursor = cursor
        variables = {"videoID": vod.id, "cursor": cursor}
    comments.sort(key=lambda c: c["content_offset_seconds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_text(json.dumps({"video": {"id": vod.id, "title": vod.title,
                                         "created_at": vod.created_at, "length": vod.length},
                               "comments": comments}), encoding="utf-8")
    tmp.replace(dest)
    return len(comments)


# ---------------------------------------------------------------------- video


def download_video(vod: Vod, dest: Path, progress: Progress | None = None) -> Path:
    """Download the VOD at source quality to `dest` (.mp4). An interrupted download resumes."""
    import shutil

    import yt_dlp

    def hook(status: dict) -> None:
        if progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes") or 0
        if total:
            progress(min(done / total, 1.0), f"{done / 1e9:.1f} / {total / 1e9:.1f} GB")

    options = {
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "concurrent_fragment_downloads": 8,
        "continuedl": True,
        # Writes the VOD's link into the file's "comment" tag, so `is_same_vod` can tell later.
        "postprocessors": [{"key": "FFmpegMetadata", "add_metadata": True}],
    }
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        options["ffmpeg_location"] = str(Path(ffmpeg).parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(vod.url, download=True)
    path = Path(info["requested_downloads"][0]["filepath"])
    if path != dest and path.exists() and not dest.exists():
        path.replace(dest)
        path = dest
    return path


SAME_LENGTH = 90  # seconds: an untagged file this close to the VOD's length is taken as that VOD


def _file_facts(path: Path) -> tuple[float, str] | None:
    """(duration, the link tags yt-dlp writes) from ffprobe, or None when it can't be read."""
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
        fmt = json.loads(result.stdout).get("format") or {}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    # Only "comment" and "purl" (the VOD's link), not the description, which may link other VODs.
    tags = " ".join(str(v) for k, v in (fmt.get("tags") or {}).items()
                    if k.lower() in ("comment", "purl"))
    return float(fmt.get("duration") or 0.0), tags


def is_same_vod(path: Path, vod: Vod) -> bool | None:
    """Whether the video at `path` is this VOD: True, False, or None when ffprobe can't tell.

    Downloads made here carry the VOD's link in their tags; files from before that (or from
    other tools) are judged by length, which a different stream almost never matches closely."""
    facts = _file_facts(path)
    if facts is None:
        return None
    duration, tags = facts
    found = {m.group(1) for m in _VOD_URL.finditer(tags)}
    if found:
        return vod.id in found
    if not vod.length or not duration:
        return None
    return abs(duration - vod.length) <= SAME_LENGTH
