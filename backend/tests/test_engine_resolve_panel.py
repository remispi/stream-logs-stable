from stream_log.engine import llm_server, pipeline
from stream_log.engine.config import Paths
from stream_log.engine.hardware import Tier
from stream_log.engine.models.manifest import parse_manifest
from stream_log.engine.project import create_project

_LLM = {"kind": "llm", "runtime": "llama.cpp", "size_gb": 1.0, "license": "MIT",
        "license_verified": True, "commercial_ok": True, "phase": 2, "purpose": "test",
        "files": ["*.gguf"]}
_ROLES = {"events": ["llm-a", "llm-b"]}


def test_a_panel_model_without_its_gguf_is_skipped(tmp_path, monkeypatch):
    manifest = parse_manifest({
        "model": [{**_LLM, "id": i, "source": f"hf:example/{i}"} for i in ("llm-a", "llm-b")],
        "tiers": {tier.value: {"asr": "llm-a", "verbatim_asr": "llm-a", "llm_default": "llm-a",
                               "llm_ensemble": ["llm-a", "llm-b"], "modes": ["fast"],
                               "roles": {"fast": _ROLES}} for tier in Tier},
    })
    paths = Paths(tmp_path).ensure()
    (paths.models / "llm-a").mkdir()
    (paths.models / "llm-a" / "A-Q4.gguf").write_bytes(b"GGUF")
    (paths.models / "llm-b").mkdir()
    (paths.models / "llm-b" / "mmproj-F16.gguf").write_bytes(b"GGUF")  # matches, but no model
    monkeypatch.setattr(pipeline, "load_manifest", lambda: manifest)
    monkeypatch.setattr(pipeline, "detect", lambda home: None)
    monkeypatch.setattr(pipeline, "choose_tier", lambda info: Tier.HIGH)
    monkeypatch.setattr(llm_server, "find_exe", lambda exe, home: tmp_path / "llama-server.exe")
    said = []

    panel = pipeline.resolve_panel(paths, create_project(paths, "p", {}), pipeline.Settings(),
                                   said.append)

    assert [m.id for m in panel.events] == ["llm-a"]
    assert any("llm-b" in line and "skipped" in line for line in said)
