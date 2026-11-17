from datetime import UTC, datetime
from itertools import pairwise

import numpy as np

from stream_log import games, listing, naming, sync, twitch
from stream_log.listing import Block, Evidence, Line


def test_title_part_keeps_caps_and_drops_commands():
    assert naming.title_part("BBBC day 12") == "BBBCDay12"
    assert naming.title_part("superbrawl winner POV Kappa -> soloq afterwards") == \
        "SuperbrawlWinnerPOVKappaSoloqAfterwards"
    assert naming.title_part("Regionals Weekend 2 Costream - fighting for worlds spots !scores") \
        == "RegionalsWeekend2CostreamFightingForWorldsSpots"
    assert naming.title_part("tac crown day 2 costream with @Hmoon76") == \
        "TacCrownDay2CostreamWithHmoon76"
    assert naming.title_part("don't stop") == "DontStop"
    assert naming.title_part("🔥🔥") == ""


def test_file_stem_uses_the_local_start_day():
    started = datetime(2026, 9, 6, 15, 0, tzinfo=UTC)
    stem = naming.file_stem(started, "BBBC Day 12")
    assert stem == f"TwitchStream{started.astimezone():%Y%m%d}-BBBCDay12"
    assert naming.file_stem(datetime(2026, 11, 2, 20), "") == "TwitchStream20261102"


def test_free_path_adds_a_number(tmp_path):
    (tmp_path / "A.mp4").write_bytes(b"")
    (tmp_path / "A-2.mp4").write_bytes(b"")
    assert naming.free_path(tmp_path, "A").name == "A-3.mp4"
    assert naming.date_in_name("TwitchStream20261114-X.mp4") == "2026-11-14"


def test_vod_id_from_links():
    assert twitch.vod_id("https://www.twitch.tv/videos/2928525824") == "2928525824"
    assert twitch.vod_id("https://www.twitch.tv/justremikun/video/2928525824?t=1h") == "2928525824"
    assert twitch.vod_id("v2928525824") == "2928525824"
    assert twitch.vod_id("hello") is None


def test_comment_matches_twitchdownloader_shape():
    node = {"id": "x", "contentOffsetSeconds": 12, "commenter": {"login": "a", "displayName": "A"},
            "message": {"fragments": [{"text": "hi ", "emote": None},
                                      {"text": "Kappa", "emote": {"emoteID": "25"}}]}}
    c = twitch._comment(node)
    assert c["content_offset_seconds"] == 12
    assert c["message"]["body"] == "hi Kappa"
    assert c["message"]["fragments"][1]["emoticon"] == {"emoticon_id": "25"}


def _path(spec: str, per: int = 6) -> np.ndarray:
    """'G20 O1 G10' -> states, each unit = `per` frames (1 min at 10 s steps)."""
    out = []
    for part in spec.split():
        state = "GLHO".index(part[0])
        out += [state] * int(part[1:]) * per
    return np.array(out)


def test_games_split_on_lobby_and_merge_short_breaks():
    found = games.games_from_path(_path("L1 G30 H2 O1 L1 G20 O2 G15 H1"))
    assert [(g["start"] // 60, g["end"] // 60) for g in found] == [(1, 31), (35, 72)]
    assert found[0]["loading"] == 0


def test_games_drop_short_runs_and_long_breaks_split():
    found = games.games_from_path(_path("G5 O10 G20 O4 G20"))
    assert [(g["start"] // 60, g["end"] // 60) for g in found] == [(15, 35), (39, 59)]


def test_cluster_merges_nearby_evidence_and_prefers_llm_text():
    lines = [{"start": 99, "end": 103, "text": "oh my god what was that"}]
    ev = [Evidence(100, 2.0, ["topic"], "Big story", 3), Evidence(110, 1.2, ["laugh"]),
          Evidence(400, 1.0, ["laugh"]), Evidence(1000, 0.6, ["loud"])]
    out = listing.cluster(ev, lines + [{"start": 398, "end": 402, "text": "ha ha ha"}])
    assert [(line.t, line.text, line.tags) for line in out] == [
        (100, "Big story", ["topic", "laugh"]), (400, "“ha ha ha”", ["laugh"])]


def test_fill_keeps_the_strongest_per_hour():
    block = Block(0, 1800, "Game 1", "game")
    lines = [Line(t, f"l{t}", ["topic"], score) for t, score in
             [(10, 1), (20, 5), (30, 2), (40, 4), (50, 3), (60, 0.5), (70, 6), (80, 7), (90, 1.5)]]
    listing.fill([block], lines, per_hour=12)
    assert [line.t for line in block.lines] == [20, 30, 40, 50, 70, 80]  # in time order
    assert len(block.folded) == 3


def test_long_runs_are_not_called_games():
    chapters = [{"start": s, "end": s + 600, "title": f"c{s}"} for s in range(0, 10800, 600)]
    blocks = listing.make_blocks(10800, [{"start": 600, "end": 2400}, {"start": 3000, "end": 10800}],
                                 chapters, [])
    assert blocks[0].title.startswith("Before the first game")
    assert blocks[1].title == "Game 1 · 30 min"
    assert blocks[2].title.startswith("Between games")
    assert len(blocks) > 4 and all(b.title.startswith("TFT on screen") for b in blocks[3:])


def test_games_watched_from_pronouns():
    played = [{"start": t, "text": "I need to roll, my board is weak"} for t in range(0, 600, 10)]
    watched = [{"start": t, "text": "he's rolling, I think his board is weak"}
               for t in range(600, 1200, 10)]
    quiet = [{"start": 1300, "text": "he's he's he's"}]
    found = [{"start": 0, "end": 600}, {"start": 600, "end": 1200}, {"start": 1200, "end": 1800}]
    marked = games.mark_watched(found, {"language": "en", "lines": played + watched + quiet})
    assert [g["watched"] for g in marked] == [False, True, False]
    french = games.mark_watched(found, {"language": "fr", "lines": played + watched})
    assert not any(g["watched"] for g in french)


def test_watched_games_get_no_number():
    found = [{"start": 0, "end": 1800, "watched": True}, {"start": 2000, "end": 3800},
             {"start": 4000, "end": 9000, "watched": True}]
    blocks = listing.make_blocks(9000, found, [], [])
    assert [b.title for b in blocks] == ["Watched game · 30 min", "Game 1 · 30 min",
                                         "Watching TFT"]


def test_chapter_blocks_tile_without_overlap():
    # The 11-14 costream's first hour: the short last block folds into the one before it.
    edges = [0, 420, 960, 1380, 1680, 2280, 3000, 3540, 4190]
    chapters = [{"start": a, "end": b, "title": f"c{a}"} for a, b in pairwise(edges)]
    blocks = listing.chapter_blocks(chapters, 0, 3920)
    assert [(b.start, b.end) for b in blocks] == [(0, 1680), (1680, 3920)]


def _windows(lags):
    return [{"t": i * sync.STEP, "lag": lag, "r": 0.5 if lag is not None else 0.1,
             "margin": 0.4 if lag is not None else 0.0} for i, lag in enumerate(lags)]


def test_sync_pieces_follow_a_drop_in_b():
    found = _windows([100.0] * 10 + [70.0] * 10)
    pieces = sync.pieces([w | {"lag": w["lag"] or 0} for w in found])
    assert len(pieces) == 2
    first, second = pieces
    assert second["a_start"] - first["a_end"] == 30.0  # B misses 30 s: A has nothing there
    to_b = sync.mapper([{"start": p["a_start"], "end": p["a_end"], "lag": p["lag"]}
                        for p in pieces])
    assert to_b(first["a_start"] + 10) == first["a_start"] + 110
    assert to_b(first["a_end"] + 10) is None


def test_ncc_finds_the_offset():
    rng = np.random.default_rng(1)
    b = rng.normal(size=5000).astype(np.float32)
    a = b[1234:1234 + 600]
    r = sync.ncc(a, b)
    assert int(np.argmax(r)) == 1234
    assert r[1234] > 0.99


def test_free_path_reuses_the_same_video(tmp_path):
    (tmp_path / "A.mp4").write_bytes(b"other")
    (tmp_path / "A-2.mp4").write_bytes(b"mine")
    def mine(p):
        return p.read_bytes() == b"mine"

    assert naming.free_path(tmp_path, "A", reuse=mine).name == "A-2.mp4"
    assert naming.free_path(tmp_path, "A", reuse=lambda p: False).name == "A-3.mp4"


def test_is_same_vod_reads_the_link_tag_then_the_length(monkeypatch, tmp_path):
    vod = twitch.Vod(id="2928525824", title="t", created_at="2026-11-14T18:00:00Z", length=7200)
    facts = {}
    monkeypatch.setattr(twitch, "_file_facts", lambda path: facts.get("v"))
    assert twitch.is_same_vod(tmp_path, vod) is None  # ffprobe can't tell
    facts["v"] = (7200.0, "https://www.twitch.tv/videos/2928525824")
    assert twitch.is_same_vod(tmp_path, vod) is True
    facts["v"] = (7200.0, "https://www.twitch.tv/videos/1111111111")
    assert twitch.is_same_vod(tmp_path, vod) is False  # same length, but tagged as another VOD
    facts["v"] = (7230.0, "")
    assert twitch.is_same_vod(tmp_path, vod) is True
    facts["v"] = (3600.0, "")
    assert twitch.is_same_vod(tmp_path, vod) is False


def test_relist_goes_to_the_folder_the_list_was_written_to(tmp_path):
    class Project:
        def path(self, name):
            return tmp_path / name

    project, video = Project(), tmp_path / "v.mp4"
    assert sync.list_dir_for(project, video) == video.parent
    sync.remember_video(project, video, tmp_path / "lists")
    sync.remember_video(project, video)  # relist: no folder given, the old one stays
    assert sync.list_dir_for(project, video) == (tmp_path / "lists").resolve()
    assert sync._list_videos(project) == [str(video.resolve())]


def test_bad_stem_rejects_what_windows_or_a_path_would_misread():
    assert naming.bad_stem("TwitchStream20261114-BBBCDay12") is None
    for name in ("", "a:b", r"..\x", "a/b", r"C:\x", "x.", "x ", "CON", "lpt1.x", "a\x01"):
        assert naming.bad_stem(name), name


def test_cli_passes_no_games(monkeypatch):
    from stream_log import cli, run

    seen = {}
    monkeypatch.setattr(run, "run", lambda job, emit: seen.setdefault("job", job))
    assert cli.main(["run", "--file", "x.mp4", "--no-games"]) == 0
    assert seen["job"].find_games is False
    seen.clear()
    assert cli.main(["run", "--file", "x.mp4"]) == 0
    assert seen["job"].find_games is True
