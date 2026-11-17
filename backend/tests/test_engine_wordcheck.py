import json

from stream_log.engine import wordcheck as W
from stream_log.engine.config import Paths
from stream_log.engine.lexicon import bare, read_list, write_list
from stream_log.engine.pipeline import STAGES, Settings, process
from stream_log.engine.project import create_project, read_json, write_json

COMMON = frozenset({"we", "the", "and", "set", "sign", "brand", "is", "fun", "play", "on",
                    "took", "with", "then", "ult", "into", "so", "his", "really", "milk"})


def words(text: str, start: float = 0.0, p: float = 0.95) -> list[dict]:
    return [{"w": w, "start": start + n * 0.4, "end": start + n * 0.4 + 0.3, "p": p,
             "filler": False} for n, w in enumerate(text.split())]


def line(n: int, text: str, p: float = 0.95) -> dict:
    return {"id": n, "start": n * 10.0, "end": n * 10.0 + 5, "text": text,
            "words": words(text, n * 10.0, p)}


def lexicon() -> W.Lexicon:
    return W.build_lexicon((["Kennen", "Brand", "Sett", "Galio", "Lee Sin", "Yorick", "Caitlyn"],
                            {"Kennan": "Kennen"}),
                           ["Sejuani", "Bounty Seeker", "Carousel", "Grow Up"],
                           ["Dishsoap", "Milk", "Narbus"], COMMON | {"grow", "up"})


def by_word(candidates):
    return {c.heard: c for c in candidates}


def test_finds_what_could_be_a_game_term_or_name():
    lines = [line(1, "we took lee sin and galio then Kennan really"),
             line(2, "the set is fun so we play brand with Sejuanni"),
             line(3, "Dishsoap and milk on Bounty Seeker sign"),
             line(4, "the carousel so grow up Narbs into Yoricks Yorick's Kaitlin")]
    found = by_word(W.find(lines, lexicon()))
    # The glossary's terms get their capitals; other lexicon terms and names are only labelled.
    assert found["lee"].options == [("Lee", "game")] and found["sin"].reason == "exact"
    assert found["galio"].options == [("Galio", "game")]
    assert found["carousel"].options == [("carousel", "game")]
    assert found["Bounty"].reason == "exact" and found["Bounty"].options == [("Bounty", "game")]
    assert found["Dishsoap"].options == [("Dishsoap", "name")]
    assert found["Kennan"].reason == "rule" and found["Kennan"].options == [("Kennen", "game")]
    assert found["brand"].reason == "ambiguous"  # a champion, and an ordinary word
    assert found["Sejuanni"].reason == "near" and found["Sejuanni"].options[0][0] == "Sejuani"
    assert found["Yorick's"].options == [("Yorick's", "game")]  # keeps its 's
    assert found["Kaitlin"].options[0][0] == "Caitlyn"  # K and C sound alike
    for ordinary in ("set", "sign", "milk", "we", "the", "grow", "up"):
        assert ordinary not in found  # "set" isn't "Sett"; "grow up" isn't a team here
    assert "Narbs" not in found  # player names are never spelling targets


def test_the_llm_only_picks_listed_spellings():
    lines = [line(1, "we play brand with Sejuanni and Kenen")]
    candidates = W.find(lines, lexicon())
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return {"words": [{"n": 1, "kind": "normal", "right": ""},          # brand: the word
                          {"n": 2, "kind": "game", "right": "Sejuani"},
                          {"n": 3, "kind": "game", "right": "Kennen the Heart"}]}  # not listed

    decisions = {d["heard"]: d for d in W.decide(lines, candidates, ask)}
    assert len(asked) == 1 and "«Sejuanni»" in asked[0] and "options: Sejuani" in asked[0]
    assert decisions["brand"]["kind"] == "normal" and decisions["brand"]["by"] == "llm"
    assert decisions["Sejuanni"]["right"] == "Sejuani"
    assert decisions["Kenen"]["kind"] == "other" and decisions["Kenen"]["right"] == ""


def test_apply_keeps_punctuation_and_what_was_heard_and_restores():
    lines = [line(1, "we took galio, then Kennan really.")]
    original = json.loads(json.dumps(lines))
    decisions = W.decide(lines, W.find(lines, lexicon()), None)
    assert W.apply(lines, decisions) == 2
    ws = [w["w"] for w in lines[0]["words"]]
    assert ws == ["we", "took", "Galio,", "then", "Kennen", "really."]
    assert lines[0]["text"] == "we took Galio, then Kennen really."
    assert lines[0]["words"][2]["heard"] == "galio," and lines[0]["words"][2]["kind"] == "game"
    W.restore(lines)
    assert lines == original
    # Running again from the restored words gives the same result.
    W.apply(lines, W.decide(lines, W.find(lines, lexicon()), None))
    assert [w["w"] for w in lines[0]["words"]] == ws


def test_without_an_llm_only_sure_capitalised_or_unsure_words_change():
    lines = [line(1, "then Sejuanni took it"), line(2, "then sejuanni took it"),
             line(3, "then sejuanni took it", p=0.3)]
    decisions = W.decide(lines, W.find(lines, lexicon()), None)
    rights = [(d["line"], d["right"], d["by"]) for d in decisions]
    assert (1, "Sejuani", "rule") in rights and (3, "Sejuani", "rule") in rights
    assert (2, "", "rule") in rights  # lowercase and heard clearly: left alone


def test_a_failed_llm_call_leaves_the_words(tmp_path):
    lines = [line(1, "we play with Sejuanni")]

    def broken(prompt):
        raise RuntimeError("server gone")

    errors = []
    decisions = W.decide(lines, W.find(lines, lexicon()), broken, errors=errors)
    assert errors == ["server gone"]
    assert decisions[0]["by"] == "rule"


def test_the_stage_runs_after_align_and_again_from_scratch(tmp_path):
    paths = Paths(tmp_path / "home").ensure()
    (paths.home / "glossary.txt").write_text("Kennen\nGalio\nKennan -> Kennen\n",
                                             encoding="utf-8")
    write_list(paths.home / "lexicon" / "game.txt", "game terms", ["Carousel"])
    project = create_project(paths, "p", {"source_path": "x.mp4"})
    transcript = {"lines": [line(1, "we took galio and Kennan")], "duration": 20.0}
    write_json(project.path("transcript.json"), transcript)
    stage = next(s for s in STAGES if s.name == "words")
    assert [s.name for s in STAGES].index("words") == [s.name for s in STAGES].index("align") + 1
    process(paths, project, Settings(use_llm=False), stages=[stage])
    data = read_json(project.path("transcript.json"))
    assert data["lines"][0]["text"] == "we took Galio and Kennen"
    report = read_json(project.path("words.json"))
    assert report["counts"]["corrected"] == 2 and report["model"] is None
    process(paths, project, Settings(use_llm=False, force={"words"}), stages=[stage])
    assert read_json(project.path("transcript.json")) == data


def test_lists_and_bare():
    assert bare("«Kennen,»") == "Kennen"
    assert bare("Kai'Sa!") == "Kai'Sa"


def test_read_list_skips_comments_and_repeats(tmp_path):
    path = tmp_path / "l.txt"
    path.write_text("# header\nGalio\ngalio  # again\n\nLee  Sin\n", encoding="utf-8")
    assert read_list(path) == ["Galio", "Lee Sin"]


def test_word_forms_are_not_mishearings():
    lex = W.build_lexicon((["Galio"], {}), ["Reforger", "Redeemer", "Challenger", "Snipers"],
                          [], COMMON | {"redeem", "challenge", "hell"})
    lines = [line(1, "let me reforge this then I redeemed it and challengers helling Snipe")]
    found = by_word(W.find(lines, lex))
    assert "redeemed" not in found and "helling" not in found  # ordinary words, other forms
    for form in ("reforge", "challengers", "Snipe"):
        assert form not in found or found[form].options == [(form, "game")]  # labelled only
    assert W.ordinary("can't", COMMON) and W.ordinary("redeemed", frozenset({"redeem"}))
    assert not W.ordinary("galileo", COMMON)
