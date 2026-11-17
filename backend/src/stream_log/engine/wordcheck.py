"""The words stage: after alignment, check that what was heard makes sense, word by word.

Speech recognition writes champion, trait and item names as whatever they sound like
("Kennan" for Kennen, "Sijuanis" for Sejuani's), and every later stage, the captions
included, inherits that. This pass looks at the words that could be game terms or names and
decides, from the sentence around them, which they are:

- `normal`: an ordinary word used normally ("set" in "the set is fun"); kept.
- `game`: a game term (champion, trait, item, augment, spell...); spelled as the lexicon does.
- `name`: a person's name or handle (players, teams, the voices named on the stream).
- `other`: something else (a name nobody listed, a word that can't be placed); kept.

Which words are looked at, and how:

- a glossary `heard -> right` rule for a single word: corrected, no question asked;
- a word or 2-3 word phrase that is exactly a game term or name: labelled; its capitals fixed
  ("galio" -> "Galio"). If the term is also an ordinary word ("Brand", "Flash", "Twitch"), only
  the hand-kept glossary's ones are checked, and the LLM says which it is in that sentence;
- a word that isn't an ordinary word and is spelled almost like a game term or name (same first
  letter, `NEAR` similar): the LLM picks one of those spellings or keeps the word.

The LLM can only choose among the listed spellings, never rewrite: free caption rewrites by an
LLM went wrong in review ("Window pay" -> "Win or pay"). Without an LLM a stricter rule decides
(one close spelling, and the word capitalised mid-sentence or heard with low confidence).

`transcript.json` keeps what was heard (`heard` on a corrected word, `text_heard` on its line),
so running the stage again starts from the recognised words; `words.json` lists every decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from stream_log.engine.lexicon import bare

PROMPT_VERSION = 2  # v2: ordinary words in any form stay normal
NEAR = 0.8  # spelling similarity to a term that makes a word worth asking about
SURE = 0.85  # without an LLM: only this close, and only one such spelling
OPTIONS = 3
MIN_LENGTH = 4  # shorter words are too easily close to something
UNSURE = 0.6  # ASR word probability under which a word may be misheard
BATCH = 20  # words per LLM call
CONTEXT_CHARS = 200
MAX_PHRASE = 3
KINDS = ("normal", "game", "name", "other")

SYSTEM = (
    "You check a speech-recognition transcript of a video game stream (often League of "
    "Legends or Teamfight Tactics). Some words may be misheard game terms or names. For each "
    "numbered word, decide from the sentence around it what the speaker meant:\n"
    "- normal: an ordinary word, used normally, in any form (plurals, past tenses, '-ing', "
    "'-er': 'I just redeemed' is normal, not Redeemer); keep it.\n"
    "- game: a game term (champion, trait, item, augment, spell, mechanic). Copy `right` "
    "exactly from that word's options.\n"
    "- name: a person's name or handle. Copy `right` exactly from that word's options.\n"
    "- other: something else (an unlisted name, a word you can't place); keep it.\n"
    "Never invent a spelling that isn't in the options. When unsure, answer normal or other: "
    "a wrong correction is worse than none."
)
SCHEMA = {
    "type": "object",
    "properties": {"words": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "kind": {"type": "string", "enum": list(KINDS)},
                       "right": {"type": "string"}},
        "required": ["n", "kind", "right"]}}},
    "required": ["words"],
}


@dataclass
class Lexicon:
    game: dict[str, str]  # lower-case term (1-3 words) -> its spelling
    names: dict[str, str]
    common: frozenset[str]  # ordinary words, lower-case
    curated: set[str]  # the hand-kept glossary's terms, lower-case
    mapping: dict[str, str]  # glossary rules for single words: heard (lower-case) -> right
    targets: dict[str, tuple[str, str]] = field(default_factory=dict)  # word -> (spelling, kind)

    def __post_init__(self) -> None:
        # Spellings a misheard word may be meant as: game terms only. Thousands of player
        # handles as targets turned ordinary words into "near" names (Narb -> Nareba).
        for low, spelled in self.game.items():
            if " " not in low and len(low) >= MIN_LENGTH and low not in self.common:
                self.targets[low] = (spelled, "game")


def build_lexicon(glossary: tuple[list[str], dict[str, str]], game: list[str],
                  names: list[str], common: frozenset[str]) -> Lexicon:
    terms, mapping = glossary
    game_terms = [*terms, *mapping.values(), *game]
    return Lexicon(
        game={t.lower(): t for t in reversed(game_terms) if len(t.split()) <= MAX_PHRASE},
        names={n.lower(): n for n in reversed(names) if len(n.split()) <= MAX_PHRASE},
        common=common, curated={t.lower() for t in [*terms, *mapping.values()]},
        mapping={h.lower(): r for h, r in mapping.items() if len(h.split()) == 1
                 and len(r.split()) == 1})


@dataclass
class Candidate:
    line: int  # line id
    index: int  # word index in the line
    heard: str  # the word as recognised, without punctuation
    options: list[tuple[str, str]]  # (spelling, kind)
    reason: str  # rule | exact | ambiguous | near


SOUND = [("ph", "f"), ("ck", "c"), ("k", "c"), ("q", "c"), ("z", "s"), ("y", "i"), ("'", "")]


def sound(word: str) -> str:
    """A rough sound-alike key: letters that sound the same made the same, doubles single
    ("Kaitlin" and "Caitlyn" -> "caitlin")."""
    out = word.lower()
    for a, b in SOUND:
        out = out.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", out)


def _similar(a: str, b: str) -> float:
    """Spelling similarity, or sound-alike similarity when that is higher."""
    return max(SequenceMatcher(None, a, b).ratio(), SequenceMatcher(None, sound(a),
                                                                     sound(b)).ratio())


def _spoken(words: list[dict]) -> list[tuple[int, dict]]:
    return [(i, w) for i, w in enumerate(words) if not w.get("filler") and bare(w["w"])]


def _same_start(a: str, b: str) -> bool:
    return sound(a)[:1] == sound(b)[:1]


CONTRACTION = re.compile(r"[a-z]+(n't|'re|'ve|'ll|'d|'m)")  # ordinary, though not one token
ENDINGS = ("ing", "ed", "es", "s", "ers", "er", "ly", "ies", "'s")  # grammatical endings
SOUND_ONLY = 0.85  # a sound-alike match with a different first letter must be this close


def ordinary(word: str, common: frozenset[str]) -> bool:
    """An ordinary word: in the vocabulary, a contraction, or a vocabulary word with a
    grammatical ending ("redeemed", "helling", "challengers": not misheard game terms)."""
    if word in common or CONTRACTION.fullmatch(word):
        return True
    for ending in ENDINGS:
        stem = word[: -len(ending)]
        if word.endswith(ending) and len(stem) >= 3 and (
                stem in common or stem + "e" in common or (stem[-1:] == stem[-2:-1]
                                                          and stem[:-1] in common)):
            return True
    return False


def _variant(word: str, term: str) -> bool:
    """The term with a grammatical ending ("challengers", "reroller"), or the term is the word
    with one ("reforge" for the item Reforger, "snipe" for Snipers): the same root in another
    form, so not a mishearing; it is labelled at most, never respelled."""
    return (word.startswith(term) and word[len(term):] in ENDINGS) or (
        term.startswith(word) and term[len(word):] in (*ENDINGS, "r", "rs"))


def _possessive(word: str) -> tuple[str, str]:
    """("Yorick", "'s") for "Yorick's": the term is looked up without it, and keeps it."""
    if word.lower().endswith("'s") and len(word) > 3:
        return word[:-2], word[-2:]
    return word, ""


def find(lines: list[dict], lex: Lexicon) -> list[Candidate]:
    """Exact matches only fix capitals of the hand-kept glossary's terms ("galio" -> "Galio");
    other lexicon terms and names are labelled and left as heard ("carousel" isn't capitalised
    mid-sentence). Phrases of ordinary words ("grow up", a team's name) are skipped unless the
    glossary has them."""
    out: list[Candidate] = []
    for line in lines:
        spoken = _spoken(line["words"])
        taken: set[int] = set()
        for size in range(MAX_PHRASE, 1, -1):  # phrases first: "Lee Sin", "Bounty Seeker"
            for k in range(len(spoken) - size + 1):
                part = spoken[k:k + size]
                if any(i in taken for i, _ in part):
                    continue
                heard = [bare(w["w"]) for _, w in part]
                phrase = " ".join(h.lower() for h in heard)
                curated = phrase in lex.curated
                if not curated and all(h.lower() in lex.common for h in heard):
                    continue
                for table, kind in ((lex.game, "game"), (lex.names, "name")):
                    if phrase in table:
                        spelled = table[phrase].split() if curated else heard
                        for (i, _), word, right in zip(part, heard, spelled, strict=True):
                            taken.add(i)
                            out.append(Candidate(line["id"], i, word, [(right, kind)], "exact"))
                        break
        for i, w in spoken:
            if i in taken:
                continue
            heard = bare(w["w"])
            base, suffix = _possessive(heard)
            low = base.lower()
            if heard.lower() in lex.mapping:
                out.append(Candidate(line["id"], i, heard, [(lex.mapping[heard.lower()], "game")],
                                     "rule"))
            elif low in lex.mapping:
                out.append(Candidate(line["id"], i, heard,
                                     [(lex.mapping[low] + suffix, "game")], "rule"))
            elif len(low) < 2 or CONTRACTION.fullmatch(heard.lower()):
                continue  # "U", "K"; "can't" isn't K'Sante
            elif low in lex.game or low in lex.names:
                kind = "game" if low in lex.game else "name"
                spelled = (lex.game if kind == "game" else lex.names)[low] + suffix
                if low in lex.curated and low not in lex.common:
                    out.append(Candidate(line["id"], i, heard, [(spelled, kind)], "exact"))
                elif low not in lex.common:
                    out.append(Candidate(line["id"], i, heard, [(heard, kind)], "exact"))
                elif kind == "game" and low in lex.curated:  # "Brand", "Twitch", "Flash"
                    out.append(Candidate(line["id"], i, heard, [(spelled, kind)], "ambiguous"))
            elif len(low) >= MIN_LENGTH and not ordinary(low, lex.common) \
                    and not ordinary(heard.lower(), lex.common):
                variant = next((t for t in lex.targets if _variant(low, t)), None)
                if variant:
                    out.append(Candidate(line["id"], i, heard, [(heard, "game")], "exact"))
                    continue
                close = sorted(((_similar(low, t), t) for t in lex.targets
                                if _same_start(t, low) and t != low), reverse=True)
                close = [(s, t) for s, t in close if s >= NEAR and (
                    t[0] == low[0] or s >= SOUND_ONLY)][:OPTIONS]
                if close:
                    out.append(Candidate(line["id"], i, heard,
                                         [(lex.targets[t][0] + suffix, lex.targets[t][1])
                                          for _, t in close], "near"))
    return out


# ---------------------------------------------------------------------- deciding


Ask = Callable[[str], dict]  # prompt -> the model's JSON answer


def _context(lines_by_id: dict[int, dict], order: list[int], c: Candidate) -> str:
    at = order.index(c.line)
    line = lines_by_id[c.line]
    words = [w["w"] for w in line["words"] if not w.get("filler")]
    spoken_index = sum(1 for j, w in enumerate(line["words"]) if j < c.index
                       and not w.get("filler"))
    marked = " ".join(f"«{w}»" if j == spoken_index else w for j, w in enumerate(words))
    before = lines_by_id[order[at - 1]]["text"][-CONTEXT_CHARS:] if at > 0 else ""
    after = lines_by_id[order[at + 1]]["text"][:CONTEXT_CHARS] if at + 1 < len(order) else ""
    return " | ".join(p for p in (before, marked[:2 * CONTEXT_CHARS], after) if p)


def prompt_for(batch: list[Candidate], lines_by_id: dict[int, dict], order: list[int],
               note: str) -> str:
    parts = [note.strip()] if note.strip() else []
    for n, c in enumerate(batch, 1):
        options = ", ".join(spelled for spelled, _ in c.options)
        parts.append(f"[{n}] word: {c.heard}\n    said: {_context(lines_by_id, order, c)}\n"
                     f"    options: {options}")
    return "\n\n".join(parts)


def _rule(c: Candidate, word: dict, capital: bool) -> tuple[str, str]:
    """(kind, right) without an LLM."""
    if c.reason in ("rule", "exact"):
        return c.options[0][1], c.options[0][0]
    if c.reason == "near" and len(c.options) == 1:
        spelled, kind = c.options[0]
        sure = _similar(c.heard.lower(), spelled.lower()) >= SURE
        if sure and (capital or word.get("p", 1.0) < UNSURE):
            return kind, spelled
    return "other" if c.reason == "near" else "normal", ""


def decide(lines: list[dict], candidates: list[Candidate], ask: Ask | None, note: str = "",
           progress: Callable[[float, str], None] = lambda *_: None,
           errors: list[str] | None = None) -> list[dict]:
    """A decision per candidate: {line, i, heard, kind, right, by, options}."""
    lines_by_id = {ln["id"]: ln for ln in lines}
    order = [ln["id"] for ln in lines]
    decisions: list[dict] = []
    asked = [c for c in candidates if c.reason in ("near", "ambiguous")]
    answers: dict[int, tuple[str, str]] = {}
    if ask is not None:
        for start in range(0, len(asked), BATCH):
            batch = asked[start:start + BATCH]
            progress(start / max(1, len(asked)), f"checking words {start + 1}-"
                     f"{start + len(batch)} of {len(asked)}")
            try:
                raw = ask(prompt_for(batch, lines_by_id, order, note))
            except Exception as error:  # noqa: BLE001 - one failed call leaves those words be
                if errors is not None:
                    errors.append(str(error))
                continue
            for item in (raw.get("words") or []) if isinstance(raw, dict) else []:
                n = item.get("n") if isinstance(item, dict) else None
                if not isinstance(n, int) or not 1 <= n <= len(batch):
                    continue
                c = batch[n - 1]
                kind = item.get("kind") if item.get("kind") in KINDS else "other"
                right = str(item.get("right") or "").strip()
                chosen = next((o for o in c.options if o[0].lower() == right.lower()), None)
                if kind in ("game", "name"):
                    answers[id(c)] = (chosen[1], chosen[0]) if chosen else ("other", "")
                else:
                    answers[id(c)] = (kind, "")
    for c in candidates:
        word = lines_by_id[c.line]["words"][c.index]
        prior = [w for w in lines_by_id[c.line]["words"][:c.index] if not w.get("filler")]
        capital = c.heard[:1].isupper() and bool(prior) and not prior[-1]["w"].endswith(
            (".", "!", "?"))
        if id(c) in answers:
            (kind, right), by = answers[id(c)], "llm"
        else:
            (kind, right), by = _rule(c, word, capital), (
                "glossary" if c.reason == "rule" else "lexicon" if c.reason == "exact"
                else "rule")
        decisions.append({"line": c.line, "i": c.index, "heard": c.heard, "kind": kind,
                          "right": right, "by": by, "reason": c.reason,
                          "options": [o[0] for o in c.options]})
    return decisions


# ---------------------------------------------------------------------- applying


def restore(lines: list[dict]) -> None:
    """Back to what was recognised (the stage's input), so a re-run starts clean."""
    for line in lines:
        for w in line["words"]:
            if "heard" in w:
                w["w"] = w.pop("heard")
            w.pop("kind", None)
        if "text_heard" in line:
            line["text"] = line.pop("text_heard")


def _respell(word: str, right: str) -> str:
    """`right` in place of the word, keeping its punctuation ("kennan," -> "Kennen,")."""
    m = re.match(r"^([^\w']*)(.*?)([^\w']*)$", word)
    return f"{m.group(1)}{right}{m.group(3)}" if m else right


def apply(lines: list[dict], decisions: list[dict]) -> int:
    """Write the decisions into the lines; returns how many words changed."""
    lines_by_id = {ln["id"]: ln for ln in lines}
    changed = 0
    cursor: dict[int, int] = {}
    for d in sorted(decisions, key=lambda d: (d["line"], d["i"])):
        if d["kind"] not in ("game", "name"):
            continue
        line = lines_by_id[d["line"]]
        word = line["words"][d["i"]]
        word["kind"] = d["kind"]
        if not d["right"] or d["right"] == d["heard"]:
            continue
        word["heard"] = word["w"]
        word["w"] = _respell(word["w"], d["right"])
        line.setdefault("text_heard", line["text"])
        pattern = re.compile(rf"(?<![\w']){re.escape(d['heard'])}(?![\w'])")
        m = pattern.search(line["text"], cursor.get(d["line"], 0))
        if m:
            line["text"] = line["text"][:m.start()] + d["right"] + line["text"][m.end():]
            cursor[d["line"]] = m.start() + len(d["right"])
        changed += 1
    return changed


def summary(decisions: list[dict]) -> dict:
    counts = {k: 0 for k in KINDS}
    for d in decisions:
        counts[d["kind"]] += 1
    counts["corrected"] = sum(1 for d in decisions if d["right"] and d["right"] != d["heard"]
                              and d["kind"] in ("game", "name"))
    return counts


def cache_key(model: str, prompt: str) -> str:
    raw = json.dumps([PROMPT_VERSION, model, SYSTEM, prompt])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""
