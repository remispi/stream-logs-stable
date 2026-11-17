"""LLM panel: several local models extract chapter titles, summaries and events, then check each
other.

Only one model fits on the GPU at a time, so they take turns. With models A, B, C (preferred
first) the order is: A extracts, B extracts, C extracts and then checks, B checks, A checks and
writes the stream summary. That is five model loads instead of six.

- Events two or more models found near the same time are merged and kept (agreement).
- An event only one model found, and every chapter title and summary sentence, goes to the other
  models with the chapter's transcript: "is this supported?". A claim is dropped when more models
  reject it than support it, counting its author as support, so with three models both checkers
  must reject it. Ties keep the claim.
- The stream summary is written last by the preferred model from the checked chapter summaries.

With one model (Fast mode, or a server the user started) it extracts and nothing is checked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from stream_log.engine.chapters import STOPWORDS, llm_titles, stream_summary, transcript_excerpt
from stream_log.engine.events import llm_events
from stream_log.engine.ingest.chat import ChatMessage
from stream_log.engine.llm import LlmClient, LlmError
from stream_log.engine.project import clock, read_cache, write_json

PROMPT_VERSION = 1  # extraction prompts (cached per model)
CHECK_VERSION = 2  # cross-check prompts and decision rules (not cached)
AGREE_WINDOW = 60.0  # seconds between two models' versions of the same event
MAX_CLAIMS_PER_CALL = 20

Log = Callable[[str], None]
Progress = Callable[[float, str], None]


@dataclass
class PanelMember:
    """One model. `session()` loads it for the duration of a `with` block and yields a client."""

    id: str
    session: Callable[[], AbstractContextManager[LlmClient]]
    identity: str = ""  # changes when the weights change (cache key)


def external_member(client: LlmClient) -> PanelMember:
    """A server the user runs themselves: always loaded, never started or stopped by us."""
    return PanelMember(client.model, lambda: nullcontext(client), client.label)


def managed_member(entry, models_root: Path, exe: Path, logs: Path) -> PanelMember:
    from stream_log.engine.llm_server import managed_server, model_identity, serve_settings

    reasons = serve_settings(entry).get("always_reasons")

    @contextmanager
    def session() -> Iterator[LlmClient]:
        with managed_server(entry, models_root, exe, logs / f"{entry.id}.server.log") as url:
            yield LlmClient(url, entry.id, reasons=reasons)

    return PanelMember(entry.id, session, model_identity(entry, models_root))


# ---------------------------------------------------------------------- pass 1: extraction


def extraction_key(member: PanelMember, chapters: list[dict], lines: list[dict]) -> str:
    """Cache key for one model's extraction, so an interrupted run resumes per model."""
    digest = hashlib.sha256()
    digest.update(json.dumps([PROMPT_VERSION, member.id, member.identity,
                              [(c["id"], c["start"], c["end"]) for c in chapters]]).encode())
    for line in lines:
        digest.update(f"{line['start']}|{line['text']}\n".encode())
    return digest.hexdigest()[:16]


def extract(llm: LlmClient, chapters: list[dict], lines: list[dict], tags: dict[int, str],
            chat: list[ChatMessage], note: str) -> dict:
    errors: list[str] = []
    titled = copy.deepcopy(chapters)
    llm_titles(llm, titled, lines, tags, note, errors)
    summary = stream_summary(llm, titled, errors)
    events, dropped = llm_events(llm, chapters, lines, tags, chat, note, errors)
    return {
        "summary": summary,
        "chapters": {c["id"]: {"title": c["title"], "summary": c.get("summary"),
                               "activity": c.get("activity")}
                     for c in titled if c.get("summary") is not None},
        "events": events,
        "grounding_dropped": dropped,
        "errors": errors,
    }


# ---------------------------------------------------------------------- merge


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def same_event(a: dict, b: dict) -> bool:
    """Two models' events describe the same thing: close in time and same kind or same words."""
    if abs(a["start"] - b["start"]) > AGREE_WINDOW:
        return False
    if a["category"] == b["category"]:
        return True
    words_a = _content_words(f"{a['title']} {a['description']}")
    words_b = _content_words(f"{b['title']} {b['description']}")
    if words_a and words_b and len(words_a & words_b) / len(words_a | words_b) >= 0.25:
        return True
    quote_a, quote_b = a["evidence"][0]["quote"].lower(), b["evidence"][0]["quote"].lower()
    return SequenceMatcher(None, quote_a, quote_b).ratio() >= 0.6


def cluster_events(per_model: dict[str, list[dict]], order: list[str]) -> list[dict]:
    """Group events across models. Each cluster: {"rep": event, "members": {model: event}}.

    The representative is the version from the earliest model in `order` (the preferred one).
    """
    clusters: list[dict] = []
    for model in order:
        for item in sorted(per_model.get(model, []), key=lambda e: e["start"]):
            match = next((c for c in clusters
                          if model not in c["members"] and same_event(c["rep"], item)), None)
            if match is None:
                clusters.append({"rep": item, "members": {model: item}})
            else:
                match["members"][model] = item
    return clusters


IMPORTANCE = ["minor", "notable", "key"]


def combine(cluster: dict) -> dict:
    rep = copy.deepcopy(cluster["rep"])
    members = cluster["members"]
    rep["models"] = list(members)
    rep["agreement"] = len(members)
    evidence = []
    for item in members.values():
        for piece in item["evidence"]:
            if piece not in evidence:
                evidence.append(piece)
    rep["evidence"] = evidence[:6]
    # Raise importance when at least two models rate it at that level or higher.
    ranks = sorted((IMPORTANCE.index(e["importance"]) for e in members.values()), reverse=True)
    if len(ranks) >= 2:
        rep["importance"] = IMPORTANCE[max(ranks[1], IMPORTANCE.index(rep["importance"]))]
    rep["follow_up"] = any(e.get("follow_up") for e in members.values())
    rep["confidence"] = round(min(0.95, 0.6 + 0.12 * len(members)), 2)
    return rep


# ---------------------------------------------------------------------- pass 2: checking

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "supported", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

VERIFY_SYSTEM = (
    "You fact-check notes about a livestream against its transcript. For each numbered claim, "
    "decide whether the transcript excerpt supports it. Supported: the transcript says or clearly "
    "shows it. Paraphrase is fine, and so are correct spellings of names the speech recognition "
    "misheard (e.g. 'Jhin' for 'Jin') and the full name of something the streamer abbreviates "
    "(e.g. 'Teamfight Tactics' for 'TFT'). Unsupported: it adds something the excerpt does not "
    "contain, such as a game, name, number or event that never comes up, or it contradicts the "
    "transcript. A claim that describes or sums up this part of the stream is supported when "
    "it fairly describes what happens in the excerpt. Give a short reason. Return one verdict "
    "for every claim id."
)


@dataclass
class Claim:
    id: int
    kind: str  # "event" | "title" | "summary"
    chapter_id: str
    author: str
    text: str
    ref: object = None  # cluster index (event) or sentence index (summary)
    yes: list[str] = field(default_factory=list)
    no: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return len(self.no) > len(self.yes) + 1  # the author counts as one "yes"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "chapter_id": self.chapter_id, "author": self.author,
                "text": self.text, "supported_by": self.yes,
                "rejected_by": [{"model": m, "reason": r} for m, r in self.no]}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def chapter_of(t: float, chapters: list[dict]) -> str | None:
    return next((c["id"] for c in chapters if c["start"] <= t < c["end"]), None)


def build_claims(chapters: list[dict], chosen: dict[str, dict], clusters: list[dict]
                 ) -> list[Claim]:
    claims: list[Claim] = []
    for index, cluster in enumerate(clusters):
        if len(cluster["members"]) > 1:
            continue
        item = cluster["rep"]
        chapter_id = chapter_of(item["start"], chapters)
        if chapter_id is None:
            continue
        text = f"At {clock(item['start'])}: {item['title']}"
        if item.get("description"):
            text += f" - {item['description']}"
        claims.append(Claim(len(claims), "event", chapter_id, next(iter(cluster["members"])),
                            text, ref=index))
    for chapter_id, choice in chosen.items():
        claims.append(Claim(len(claims), "title", chapter_id, choice["by"],
                            f"This part of the stream is fairly described as: "
                            f"\"{choice['title']}\""))
        for n, sentence in enumerate(split_sentences(choice.get("summary") or "")):
            claims.append(Claim(len(claims), "summary", chapter_id, choice["by"], sentence,
                                ref=n))
    return claims


def check(llm: LlmClient, checker: str, claims: list[Claim], chapters: list[dict],
          lines: list[dict], tags: dict[int, str], errors: list[str]) -> None:
    """Ask `checker` about every claim it didn't write, one call per chapter (in batches)."""
    by_chapter: dict[str, list[Claim]] = {}
    for claim in claims:
        if claim.author != checker:
            by_chapter.setdefault(claim.chapter_id, []).append(claim)
    for chapter in chapters:
        pending = by_chapter.get(chapter["id"], [])
        if not pending:
            continue
        excerpt = transcript_excerpt(lines, tags, chapter["start"], chapter["end"], limit=16000)
        for start in range(0, len(pending), MAX_CLAIMS_PER_CALL):
            batch = pending[start : start + MAX_CLAIMS_PER_CALL]
            numbered = "\n".join(f"{c.id}. {c.text}" for c in batch)
            prompt = (f"Transcript {clock(chapter['start'])}-{clock(chapter['end'])}:\n{excerpt}"
                      f"\n\nClaims:\n{numbered}")
            try:
                result = llm.chat_json(VERIFY_SYSTEM, prompt, VERIFY_SCHEMA,
                                       max_tokens=400 + 120 * len(batch))
            except LlmError as error:
                errors.append(f"{checker} checking {chapter['id']}: {error}")
                continue  # no verdict = abstain
            wanted = {c.id: c for c in batch}
            for verdict in result.get("verdicts", []):
                claim = wanted.pop(verdict.get("id"), None)
                if claim is None:
                    continue
                if verdict.get("supported"):
                    claim.yes.append(checker)
                else:
                    claim.no.append((checker, str(verdict.get("reason", ""))[:200]))


# ---------------------------------------------------------------------- the whole panel


def choose_chapter_text(chapters: list[dict], runs: dict[str, dict], order: list[str]
                        ) -> dict[str, dict]:
    """Per chapter, the title/summary of the first model in `order` that wrote one.

    The other models' titles are kept as fallbacks in case the chosen title is rejected.
    """
    chosen = {}
    for chapter in chapters:
        texts = [(model, runs[model]["chapters"].get(chapter["id"])) for model in order]
        texts = [(model, text) for model, text in texts if text]
        if texts:
            model, text = texts[0]
            chosen[chapter["id"]] = {**text, "by": model,
                                     "other_titles": [t["title"] for _, t in texts[1:]]}
    return chosen


def run_panel(
    members: list[PanelMember],
    chapters: list[dict],
    lines: list[dict],
    tags: dict[int, str],
    chat: list[ChatMessage],
    *,
    note: str = "",
    cache_dir: Path | None = None,
    progress: Progress | None = None,
    log: Log | None = None,
) -> dict:
    progress = progress or (lambda *_: None)
    log = log or (lambda *_: None)
    order = [m.id for m in members]
    runs: dict[str, dict] = {}
    errors: list[str] = []
    claims: list[Claim] = []
    clusters: list[dict] = []
    chosen: dict[str, dict] = {}
    summary: str | None = None
    steps = len(members) * 2
    done = 0

    def cached_run(member: PanelMember) -> dict | None:
        if cache_dir is None:
            return None
        data = read_cache(cache_dir / f"{member.id}.json")
        return data if data.get("key") == extraction_key(member, chapters, lines) else None

    def store_run(member: PanelMember, data: dict) -> None:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            data = {**data, "key": extraction_key(member, chapters, lines)}
            write_json(cache_dir / f"{member.id}.json", data, indent=None)

    def prepare_checks() -> None:
        nonlocal clusters, chosen, claims
        clusters = cluster_events({m: runs[m]["events"] for m in order}, order)
        chosen = choose_chapter_text(chapters, runs, order)
        claims = build_claims(chapters, chosen, clusters) if len(members) > 1 else []
        agreed = sum(1 for c in clusters if len(c["members"]) > 1)
        log(f"panel: {len(clusters)} distinct events, {agreed} found by 2+ models, "
            f"{len(claims)} claims to cross-check")

    # Pass 1 (extraction) for every model; the last one also starts pass 2 while loaded.
    for index, member in enumerate(members):
        last = index == len(members) - 1
        cached = cached_run(member)
        if cached is not None:
            runs[member.id] = cached
            log(f"{member.id}: extraction cached")
        if cached is not None and not (last and len(members) > 1):
            done += 1
            continue
        with member.session() as llm:
            if cached is None:
                progress(done / steps, f"{member.id}: titles, summaries and events")
                runs[member.id] = extract(llm, chapters, lines, tags, chat, note)
                store_run(member, runs[member.id])
                log(f"{member.id}: {len(runs[member.id]['events'])} events "
                    f"({runs[member.id]['grounding_dropped']} failed the quote check)")
            done += 1
            if last and len(members) > 1:
                prepare_checks()
                progress(done / steps, f"{member.id}: cross-checking")
                check(llm, member.id, claims, chapters, lines, tags, errors)
                done += 1

    if len(members) == 1:
        prepare_checks()
        summary = runs[order[0]].get("summary")

    # Pass 2 for the other models, newest first; the preferred model ends and writes the summary.
    if len(members) > 1:
        for member in reversed(members[:-1]):
            with member.session() as llm:
                progress(done / steps, f"{member.id}: cross-checking")
                check(llm, member.id, claims, chapters, lines, tags, errors)
                done += 1
                if member is members[0]:
                    chosen_chapters = apply_chapter_text(chapters, chosen, claims)
                    summary = stream_summary(llm, chosen_chapters, errors)

    chosen_chapters = apply_chapter_text(chapters, chosen, claims)
    for run in runs.values():
        errors.extend(run.get("errors", []))
    event_claims = {c.ref: c for c in claims if c.kind == "event"}
    events = []
    for index, cluster in enumerate(clusters):
        claim = event_claims.get(index)
        if claim is not None and claim.rejected:
            continue
        item = combine(cluster)
        if claim is not None:
            item["checked_by"] = claim.yes
        events.append(item)
    rejected = [c.to_dict() for c in claims if c.rejected]
    return {
        "models": order,
        "chapters": {c["id"]: {k: c[k] for k in ("title", "summary", "activity", "by")}
                     for c in chosen_chapters if c.get("by")},
        "summary": summary,
        "events": sorted(events, key=lambda e: e["start"]),
        "rejected": rejected,
        "stats": {
            "events_per_model": {m: len(runs[m]["events"]) for m in order},
            "grounding_dropped": {m: runs[m]["grounding_dropped"] for m in order},
            "distinct_events": len(clusters),
            "agreed": sum(1 for c in clusters if len(c["members"]) > 1),
            "claims_checked": len(claims),
            "claims_rejected": len(rejected),
        },
        "errors": errors,
    }


def apply_chapter_text(chapters: list[dict], chosen: dict[str, dict], claims: list[Claim]
                       ) -> list[dict]:
    """Chapters with the chosen LLM title/summary, minus rejected sentences or titles."""
    rejected_titles = {c.chapter_id for c in claims if c.kind == "title" and c.rejected}
    rejected_sentences = {(c.chapter_id, c.ref) for c in claims
                          if c.kind == "summary" and c.rejected}
    out = []
    for chapter in chapters:
        chapter = dict(chapter)
        choice = chosen.get(chapter["id"])
        if choice:
            if chapter["id"] not in rejected_titles:
                chapter["title"] = choice["title"]
            elif choice.get("other_titles"):
                chapter["title"] = choice["other_titles"][0]  # next model's (unchecked) title
            kept = [s for n, s in enumerate(split_sentences(choice.get("summary") or ""))
                    if (chapter["id"], n) not in rejected_sentences]
            chapter["summary"] = " ".join(kept) or None
            chapter["activity"] = choice.get("activity")
            chapter["by"] = choice["by"]
        out.append(chapter)
    return out
