"""Word lists the transcript is checked against: game terms, people's names, ordinary words.

- `glossary.txt` (data home): the channel's own names and terms, hand-kept; short, since the
  word-check prompt lists it.
- `lexicon/game.txt`: every in-game name (champions, traits, items, augments, spells); long,
  never put in a prompt whole. `lexicon/names.txt`: players and teams. Both are plain lists
  (one per line, `#` comments) that any tool may fill, for example from Community Dragon,
  Data Dragon, Leaguepedia or Liquipedia.
- Ordinary English words: the speech-recognition model's own vocabulary (words that are a
  single token with a leading space), so no dictionary has to be shipped.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

from stream_log.engine.config import Paths

LEXICON_DIR = "lexicon"
GAME_FILE = "game.txt"
NAMES_FILE = "names.txt"
VOCAB_MODELS = ("whisper-large-v3-turbo", "crisperwhisper", "whisper-tiny-en")


def lexicon_path(paths: Paths, name: str) -> Path:
    return paths.home / LEXICON_DIR / name


def read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    out, seen = [], set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = " ".join(raw.split("#", 1)[0].split())
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def write_list(path: Path, header: str, items: list[str]) -> None:
    from stream_log.engine.project import write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, "".join(f"# {line}\n" for line in header.splitlines())
               + "".join(f"{item}\n" for item in items))


def bare(word: str) -> str:
    """A word without the punctuation around it ("Kennen," -> "Kennen")."""
    return re.sub(r"^[^\w']+|[^\w']+$", "", word)


def common_words(models_root: Path) -> frozenset[str]:
    """Lower-case ordinary words, from the first speech model found; empty if none."""
    for model in VOCAB_MODELS:
        found = _vocabulary(str(models_root / model / "tokenizer.json"))
        if found:
            return found
    return frozenset()


@cache
def _vocabulary(path: str) -> frozenset[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    vocab = (data.get("model") or {}).get("vocab") or {}
    return frozenset(t[1:].lower() for t in vocab if t.startswith("Ġ") and t[1:].isalpha()
                     and len(t) > 2)


GLOSSARY = "glossary.txt"  # in the data home


def read_glossary(path: Path) -> tuple[list[str], dict[str, str]]:
    """Terms (hints for the LLM) and `heard -> right` fixes from a glossary file."""
    terms, mapping = [], {}
    if not path.exists():
        return terms, mapping
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "->" in line:
            heard, right = (part.strip() for part in line.split("->", 1))
            if heard and right:
                mapping[heard] = right
                terms.append(right)
        elif line:
            terms.append(line)
    return list(dict.fromkeys(terms)), mapping
