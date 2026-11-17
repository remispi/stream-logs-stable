import json
import os
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stream_log.engine.llm import LlmClient
from stream_log.engine.llm_server import find_exe, free_port, runtime_builds
from stream_log.engine.panel import PanelMember, cluster_events, run_panel, same_event

LINES = [
    {"id": 0, "start": 10.0, "end": 14.0, "text": "welcome back everyone, today we play elden ring"},
    {"id": 1, "start": 70.0, "end": 75.0, "text": "I'll be streaming tomorrow at 6 with my friend"},
    {"id": 2, "start": 130.0, "end": 136.0, "text": "my cat knocked over the plant this morning"},
]
CHAPTERS = [{"id": "ch_01", "start": 0.0, "end": 200.0, "title": "Chatting: elden",
             "mode": "chatting", "game": None}]


def _event(time_, quote, title, category="conversation"):
    return {"time": time_, "end_time": "", "category": category, "type": "topic", "title": title,
            "description": "", "quote": quote, "importance": "notable", "people": [],
            "follow_up": False}


EVENTS = {
    "a": [_event("0:01:10", "streaming tomorrow at 6", "Next stream tomorrow", "announcement"),
          _event("0:02:10", "my cat knocked over the plant", "Cat story")],
    "b": [_event("0:01:15", "I'll be streaming tomorrow", "Tomorrow's stream", "announcement")],
    "c": [_event("0:00:10", "today we play elden ring", "Elden Ring today", "gameplay")],
}


class _Panel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, content):
        body = json.dumps({"choices": [{"finish_reason": "stop",
                                        "message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        props = (body.get("response_format", {}).get("json_schema", {})
                 .get("schema", {}).get("properties", {}))
        user = body["messages"][-1]["content"]
        if "events" in props:
            self._send(json.dumps({"events": EVENTS[model]}))
        elif "title" in props:
            summary = "Welcome back everyone. They play Auto Chess." if model == "a" else "Hi."
            self._send(json.dumps({"title": f"Title by {model}", "summary": summary,
                                   "activity": "banter"}))
        elif "verdicts" in props:
            claims = re.findall(r"^(\d+)\. (.*)$", user.split("Claims:")[1], re.MULTILINE)
            verdicts = [{"id": int(i), "reason": "checked",
                         "supported": not ("Auto Chess" in text or "Cat" in text)}
                        for i, text in claims]
            self._send(json.dumps({"verdicts": verdicts}))
        else:
            self._send(f"Summary by {model}.")


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Panel)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def _members(url, loads):
    def member(name):
        @contextmanager
        def session():
            loads.append(name)
            yield LlmClient(url, name)

        return PanelMember(name, session, name)

    return [member(m) for m in ("a", "b", "c")]


def test_panel_agrees_checks_and_rejects(server, tmp_path):
    loads = []
    result = run_panel(_members(server, loads), CHAPTERS, LINES, {}, [], cache_dir=tmp_path)
    assert loads == ["a", "b", "c", "b", "a"]  # five loads, the last extractor checks right away

    titles = {e["title"]: e for e in result["events"]}
    assert set(titles) == {"Next stream tomorrow", "Elden Ring today"}
    assert titles["Next stream tomorrow"]["models"] == ["a", "b"]  # agreement, no check needed
    assert titles["Elden Ring today"]["checked_by"] == ["b", "a"]  # single-model, confirmed
    rejected = {r["text"] for r in result["rejected"]}
    assert any("Cat story" in text for text in rejected)  # single-model, both checkers said no
    assert "They play Auto Chess." in rejected

    chapter = result["chapters"]["ch_01"]
    assert chapter["title"] == "Title by a" and chapter["by"] == "a"
    assert chapter["summary"] == "Welcome back everyone."  # invented sentence removed
    assert result["summary"] == "Summary by a."  # written by the preferred model at the end
    assert result["stats"]["agreed"] == 1 and result["stats"]["claims_rejected"] == 2

    # A rerun reuses every model's extraction; only checking loads models again.
    loads.clear()
    again = run_panel(_members(server, loads), CHAPTERS, LINES, {}, [], cache_dir=tmp_path)
    assert loads == ["c", "b", "a"]
    assert again["events"] == result["events"]


def test_single_model_is_not_checked(server):
    loads = []
    result = run_panel(_members(server, loads)[:1], CHAPTERS, LINES, {}, [])
    assert loads == ["a"]
    assert result["rejected"] == [] and result["stats"]["claims_checked"] == 0
    assert result["chapters"]["ch_01"]["summary"] == "Welcome back everyone. They play Auto Chess."


def _ev(start, category, title, quote="x"):
    return {"start": start, "end": start + 5, "category": category, "type": "t", "title": title,
            "description": "", "evidence": [{"quote": quote}], "importance": "minor"}


def test_same_event_rules():
    assert same_event(_ev(10, "gameplay", "Won"), _ev(50, "gameplay", "Victory"))
    assert not same_event(_ev(10, "gameplay", "Won"), _ev(100, "gameplay", "Won"))
    assert same_event(_ev(10, "gameplay", "Lilia item build"),
                      _ev(20, "conversation", "Item build for Lilia"))
    assert not same_event(_ev(10, "gameplay", "Won round", "we won that round"),
                          _ev(20, "conversation", "Cat story", "my cat knocked it over"))


def test_cluster_events_one_per_model():
    clusters = cluster_events({"a": [_ev(10, "gameplay", "Won"), _ev(20, "gameplay", "Lost")],
                               "b": [_ev(12, "gameplay", "Won it")]}, ["a", "b"])
    assert [sorted(c["members"]) for c in clusters] == [["a", "b"], ["a"]]
    assert clusters[0]["rep"]["title"] == "Won"  # preferred model's wording


def test_runtime_discovery_prefers_aged_builds(tmp_path, monkeypatch):
    monkeypatch.delenv("STREAM_LOG_LLAMA_SERVER", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    root = tmp_path / "runtimes" / "llama.cpp"
    for build, age_days in (("b100", 30), ("b200", 1), ("notes", 99)):
        (root / build).mkdir(parents=True)
        exe = root / build / "llama-server.exe"
        exe.write_bytes(b"")
        old = time.time() - age_days * 86400
        os.utime(exe, (old, old))
    assert [p.parent.name for p in runtime_builds(tmp_path)] == ["b200", "b100"]
    assert find_exe(home=tmp_path).parent.name == "b100"  # b200 is too new for Smart App Control


def test_free_port_avoids_busy_port():
    httpd = HTTPServer(("127.0.0.1", 0), _Panel)
    busy = httpd.server_port
    try:
        assert free_port(busy) != busy
    finally:
        httpd.server_close()


def test_rejected_title_falls_back_to_next_model():
    from stream_log.engine.panel import Claim, apply_chapter_text

    chosen = {"ch_01": {"title": "Bad title", "summary": "One. Two.", "activity": "banter",
                        "by": "a", "other_titles": ["Good title"]}}
    title = Claim(0, "title", "ch_01", "a", "x", no=[("b", "no"), ("c", "no")])
    sentence = Claim(1, "summary", "ch_01", "a", "Two.", ref=1, no=[("b", "no"), ("c", "no")])
    (chapter,) = apply_chapter_text(CHAPTERS, chosen, [title, sentence])
    assert chapter["title"] == "Good title" and chapter["summary"] == "One."
