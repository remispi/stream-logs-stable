"""Write a StreamList as plain text (to skim anywhere) and HTML (click a time to watch it).

Both go next to the video: `<video name>.timestamps.txt` and `.timestamps.html`. In the HTML every
time opens the local video LEAD_IN seconds before the moment (browsers play MP4 files and honour
`#t=`), and a second link opens the Twitch VOD there while it still exists.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

from stream_log.listing import Block, Line, StreamList, clock, minutes

LEAD_IN = 15
TEXT_WIDTH = 58


def twitch_time(url: str, seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{url}?t={s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def _header_range(block: Block) -> tuple[float, float]:
    if block.kind == "game" and block.game_start is not None:
        return block.game_start, block.game_end or block.end
    return block.start, block.end


def _intro(sl: StreamList) -> list[str]:
    facts = [sl.title]
    if sl.channel:
        facts.append(sl.channel)
    if sl.date:
        facts.append(sl.date)
    facts.append(minutes(sl.duration))
    if sl.games:
        facts.append(f"{sl.games} game{'s' if sl.games != 1 else ''} ({minutes(sl.game_seconds)})")
    if sl.watched:
        plural = "s" if sl.watched != 1 else ""
        facts.append(f"{sl.watched} watched" if sl.games else f"{sl.watched} game{plural} watched")
    return facts


OtherTime = Callable[[float], float | None]


def _other_note(others: list[dict], t: float) -> list[tuple[str, float, str | None]]:
    out = []
    for other in others:
        mapped = other["map"](t)
        if mapped is not None:
            out.append((other["label"], mapped, other.get("url")))
    return out


def to_text(sl: StreamList) -> str:
    out = [sl.name, " · ".join(_intro(sl))]
    if sl.summary:
        out.append(sl.summary)
    for other in sl.others:
        out.append(other.get("note", ""))
    shown = sum(len(b.lines) for b in sl.blocks)
    out.append(f"{shown} moments, at most {sl.per_hour} per hour; times are in this VOD.")
    for block in sl.blocks:
        start, end = _header_range(block)
        out.append("")
        out.append(f"── {clock(start)}–{clock(end)}  {block.title} ──")
        for line in block.lines:
            text = line.text if len(line.text) <= TEXT_WIDTH else line.text[:TEXT_WIDTH - 1] + "…"
            row = f"{clock(line.t)}  {text:<{TEXT_WIDTH}}  {' · '.join(line.tags)}"
            notes = _other_note(sl.others, line.t)
            if notes:
                row += "   " + "  ".join(f"[{label} {clock(t)}]" for label, t, _ in notes)
            out.append(row.rstrip())
        if block.folded:
            out.append(f"         +{len(block.folded)} smaller")
    return "\n".join(out) + "\n"


CSS = """
:root { --bg: #fbfaf7; --fg: #1d1b18; --muted: #6b665d; --line: #e6e1d8; --accent: #b86e00;
  --chip: #f1ece2; --head: #f4efe6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #161513;
  --fg: #ece8e1; --muted: #a39d92; --line: #2c2a26; --accent: #f0a63a; --chip: #24221f;
  --head: #1e1c19; } }
:root[data-theme="dark"] { --bg: #161513; --fg: #ece8e1; --muted: #a39d92; --line: #2c2a26;
  --accent: #f0a63a; --chip: #24221f; --head: #1e1c19; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.45 "Segoe UI", system-ui, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.facts { color: var(--muted); margin: 0 0 10px; }
.summary { margin: 0 0 8px; max-width: 75ch; }
.note { color: var(--muted); margin: 0 0 4px; font-size: 13px; }
section { margin-top: 18px; }
h2 { font-size: 14px; font-weight: 600; margin: 0; padding: 6px 10px; background: var(--head);
  border-left: 3px solid var(--accent); border-radius: 3px; display: flex; gap: 10px; }
h2 .range { font-family: Consolas, ui-monospace, monospace; color: var(--muted); font-weight: 400; }
ol { list-style: none; margin: 4px 0 0; padding: 0; }
li { display: grid; grid-template-columns: 74px 1fr auto; gap: 10px; padding: 3px 10px;
  border-bottom: 1px solid var(--line); align-items: baseline; }
li:hover { background: var(--chip); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.t { font-family: Consolas, ui-monospace, monospace; }
.tags { display: flex; gap: 4px; flex-wrap: wrap; justify-content: flex-end; }
.tag { font-size: 12px; padding: 0 6px; border-radius: 8px; background: var(--chip);
  color: var(--muted); white-space: nowrap; }
.tag.strong { color: var(--accent); }
.links { font-size: 12px; color: var(--muted); white-space: nowrap; }
details { margin: 2px 10px 0; color: var(--muted); font-size: 13px; }
details li { font-size: 13px; }
@media (max-width: 640px) { li { grid-template-columns: 64px 1fr; }
  .tags, .links { grid-column: 2; justify-content: flex-start; } }
"""

STRONG = {"laugh", "yell", "loud", "raid"}


def to_html(sl: StreamList, video_href: str) -> str:
    e = html.escape

    def row(line: Line) -> str:
        local = f'<a class="t" href="{e(video_href)}#t={max(0, int(line.t) - LEAD_IN)}" ' \
                f'title="Open the video here">{clock(line.t)}</a>'
        tags = "".join(
            f'<span class="tag{" strong" if t.split(" ")[0] in STRONG or t.startswith("chat") else ""}">'
            f"{e(t)}</span>" for t in line.tags)
        links = []
        if sl.url:
            links.append(f'<a href="{e(twitch_time(sl.url, line.t - LEAD_IN))}">Twitch</a>')
        for label, t, url in _other_note(sl.others, line.t):
            target = e(twitch_time(url, t - LEAD_IN)) if url else None
            links.append(f'<a href="{target}">{e(label)} {clock(t)}</a>' if target
                         else f"{e(label)} {clock(t)}")
        link_html = f'<span class="links">{" · ".join(links)}</span>' if links else ""
        return f"<li>{local}<span>{e(line.text)}</span><span class=\"tags\">{tags}{link_html}</span></li>"

    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{e(sl.name)}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{e(sl.title)}</h1><p class=\"facts\">{e(' · '.join(_intro(sl)[1:]))}</p>",
    ]
    if sl.summary:
        parts.append(f'<p class="summary">{e(sl.summary)}</p>')
    for other in sl.others:
        if other.get("note"):
            parts.append(f'<p class="note">{e(other["note"])}</p>')
    shown = sum(len(b.lines) for b in sl.blocks)
    parts.append(f'<p class="note">{shown} moments, at most {sl.per_hour} per hour. Click a time '
                 f"to open the video {LEAD_IN} s before it.</p>")
    for block in sl.blocks:
        start, end = _header_range(block)
        parts.append(f'<section><h2><span class="range">{clock(start)}–{clock(end)}</span>'
                     f"<span>{e(block.title)}</span></h2><ol>")
        parts.extend(row(line) for line in block.lines)
        parts.append("</ol>")
        if block.folded:
            parts.append(f"<details><summary>+{len(block.folded)} smaller</summary><ol>")
            parts.extend(row(line) for line in block.folded)
            parts.append("</ol></details>")
        parts.append("</section>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write(sl: StreamList, folder: Path) -> tuple[Path, Path]:
    video = Path(sl.video)
    folder.mkdir(parents=True, exist_ok=True)
    txt = folder / f"{video.stem}.timestamps.txt"
    page = folder / f"{video.stem}.timestamps.html"
    href = quote(video.name) if video.parent == folder else video.resolve().as_uri()
    txt.write_text(to_text(sl), encoding="utf-8")
    page.write_text(to_html(sl, href), encoding="utf-8")
    return txt, page
