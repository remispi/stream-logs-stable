import pytest

from stream_log.engine.config import Paths
from stream_log.engine.ingest.sources import classify, fingerprint
from stream_log.engine.pipeline import Settings, Stage, process
from stream_log.engine.project import (
    ProjectError,
    create_project,
    open_project,
    read_cache,
    write_json,
)


def test_classify_sources(tmp_path):
    video = tmp_path / "My Stream (Oct 30).mp4"
    video.write_bytes(b"x")
    source = classify(str(video))
    assert source.ref == str(video.resolve())
    assert source.project_id.startswith("my-stream-oct-30-")
    with pytest.raises(ValueError):
        classify(str(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError):  # links are downloaded by stream_log.twitch, not here
        classify("https://www.twitch.tv/videos/2283445566")


def test_fingerprint_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"1" * 100)
    first = fingerprint(a)
    a.write_bytes(b"2" * 100)
    assert fingerprint(a) != first


def test_open_project_by_prefix(tmp_path):
    paths = Paths(tmp_path)
    create_project(paths, "twitch-111", {})
    create_project(paths, "twitch-112", {})
    assert open_project(paths, "twitch-111").id == "twitch-111"
    with pytest.raises(ProjectError, match="several"):
        open_project(paths, "twitch-11")
    with pytest.raises(ProjectError, match="no project"):
        open_project(paths, "yt-")


def test_stage_cache_and_downstream_invalidation(tmp_path):
    paths = Paths(tmp_path)
    project = create_project(paths, "p", {})
    runs = []
    setting = {"value": 1}

    def stage(name):
        def run(ctx):
            runs.append(name)
            ctx.project.path(f"{name}.txt").write_text("ok")

        return Stage(name, 1, lambda ctx: [f"{name}.txt"],
                     (lambda ctx: dict(setting)) if name == "a" else (lambda ctx: {}), run)

    stages = [stage("a"), stage("b")]
    process(paths, project, stages=stages)
    assert runs == ["a", "b"]
    process(paths, project, stages=stages)
    assert runs == ["a", "b"]  # all cached
    setting["value"] = 2
    process(paths, project, stages=stages)
    assert runs == ["a", "b", "a", "b"]  # config change cascades
    process(paths, project, Settings(force={"a"}), stages=stages)
    assert runs[-2:] == ["a", "b"]  # forcing a stage re-runs what follows
    project.path("b.txt").unlink()
    report = process(paths, project, stages=stages)
    assert [status for _, status, _ in report] == ["cached", "ran"]


def test_json_writes_are_whole_and_caches_forgiving(tmp_path):
    path = tmp_path / "cache.json"
    assert read_cache(path) == {} and read_cache(None) == {}
    write_json(path, {"a": 1})
    assert read_cache(path) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]  # no temporary file left
    path.write_text('{"a": 1, "b"', encoding="utf-8")  # a run stopped mid-write (old writer)
    assert read_cache(path) == {}
    path.write_text("[1]", encoding="utf-8")
    assert read_cache(path) == {}
