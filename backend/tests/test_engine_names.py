from types import SimpleNamespace

import pytest

from stream_log.engine import analysis as A
from stream_log.engine.config import Paths
from stream_log.engine.project import create_project, write_json
from stream_log.engine.speakers.names import (
    NAME_MAX,
    clean_name,
    set_name,
    speaker_labels,
    voice_names,
    voices_note,
)
from stream_log.engine.speakers.profiles import create_profile, load_profile

# Voice 0 is the streamer, voice 1 a guest (one of its lines matched the enrolled voice, one is a
# repeated song line), voice 2 a media voice by override.
SPEAKERS = {
    "streamer_cluster": 0,
    "clusters": [{"id": 0, "tag": "streamer", "lines": 2, "seconds": 9.0, "voice_sim": 0.9},
                 {"id": 1, "tag": "guest", "lines": 4, "seconds": 12.0, "voice_sim": 0.4},
                 {"id": 2, "tag": "media", "lines": 1, "seconds": 3.0, "voice_sim": 0.1}],
    "lines": [{"id": 0, "cluster": 0, "tag": "streamer"},
              {"id": 1, "cluster": 1, "tag": "guest"},
              {"id": 2, "cluster": 1, "tag": "streamer", "reason": "matches enrolled voice"},
              {"id": 3, "cluster": 1, "tag": "media"},
              {"id": 4, "cluster": 2, "tag": "media"},
              {"id": 5, "cluster": 1, "tag": "guest"}],
}
TEXTS = ["hi chat", "we met in California", "yeah", "never gonna give", "welcome to summoners",
         "hold the case for a second while I screw this in"]


@pytest.fixture
def project(tmp_path):
    paths = Paths(tmp_path)
    create_profile(paths, "streamer")
    project = create_project(paths, "p1", {"source_path": "vod.mp4", "profile": "streamer"})
    write_json(project.path("speakers.json"), SPEAKERS)
    write_json(project.path("transcript.json"), {"duration": 60.0, "lines": [
        {"id": i, "start": 10.0 * i, "end": 10.0 * i + 1 + len(t) / 10, "text": t}
        for i, t in enumerate(TEXTS)]})
    return paths, project


def test_clean_name_keeps_transcript_lines_parsable():
    assert clean_name(' guest\'s "parent":\n [guest] ') == "guest's parent guest"
    assert len(clean_name("x" * 100)) == NAME_MAX
    assert clean_name(None) == ""


def test_no_names_changes_nothing():
    assert speaker_labels(SPEAKERS, {}) == {0: "streamer", 1: "guest", 2: "streamer",
                                            3: "media", 4: "media", 5: "guest"}
    assert voices_note(SPEAKERS, {}) == ""


def test_names_go_on_the_lines_that_kept_their_voice_tag():
    names = {"streamer": {"name": "Creator"},
             "1": {"name": "guest's parent", "about": "helps with the project"}}
    labels = speaker_labels(SPEAKERS, names)
    assert labels == {0: "Creator", 1: "guest's parent", 2: "Creator", 3: "media", 4: "media",
                      5: "guest's parent"}
    note = voices_note(SPEAKERS, names)
    assert '"Creator" is the streamer.' in note
    assert "\"guest's parent\" is someone else on the stream (helps with the project)." in note


def test_names_of_missing_or_streamer_voices_are_ignored():
    names = {"0": {"name": "Not used"}, "7": {"name": "Gone"}, "2": {"name": "The song"}}
    labels = speaker_labels(SPEAKERS, names)
    assert labels[0] == "streamer" and labels[4] == "The song" and labels[3] == "media"
    note = voices_note(SPEAKERS, names)
    assert "Gone" not in note and "Not used" not in note
    assert '"The song" is audio from the game or a video.' in note


def test_streamer_name_lives_in_the_profile(project):
    paths, p = project
    assert "profile streamer" in set_name(paths, p, "streamer", "Creator")
    assert load_profile(paths, "streamer")["streamer_name"] == "Creator"
    assert voice_names(paths, p)["streamer"] == {"name": "Creator", "about": "",
                                                 "channel": "streamer"}
    other = create_project(paths, "p2", {"source_path": "b.mp4", "profile": "streamer"})
    assert voice_names(paths, other)["streamer"]["name"] == "Creator"
    # A project's own name wins; naming the channel again drops it.
    set_name(paths, p, "streamer", "Creator Local", channel=False)
    assert voice_names(paths, p)["streamer"]["name"] == "Creator Local"
    assert voice_names(paths, other)["streamer"]["name"] == "Creator"
    set_name(paths, p, "0", "Creator Alias")  # the streamer's voice by its number
    assert "streamer" not in p.meta["voices"]
    assert voice_names(paths, other)["streamer"]["name"] == "Creator Alias"


def test_guest_names_are_per_project_and_removable(project):
    paths, p = project
    assert set_name(paths, p, "1", "guest's parent", "helps with the project") == "this project"
    assert p.meta["voices"] == {
        "1": {"name": "guest's parent", "about": "helps with the project"}
    }
    set_name(paths, p, "1", "  ")
    assert p.meta["voices"] == {}


def test_stage_keys_only_change_once_a_voice_is_named(project):
    paths, p = project
    ctx = SimpleNamespace(paths=paths, project=p)
    config = {"prompt": 1}
    assert A._with_voices(ctx, config) == config
    set_name(paths, p, "1", "guest's parent")
    labels, note = A._voices(ctx)
    assert A._with_voices(ctx, config) == {"prompt": 1, "voices": note}
    assert labels[1] == "guest's parent" and labels[0] == "streamer"
