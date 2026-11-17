import copy

import pytest

from stream_log.engine.hardware import Tier
from stream_log.engine.models import load_manifest
from stream_log.engine.models.manifest import ManifestError, parse_manifest

_ENTRY = {
    "id": "asr-a",
    "kind": "asr",
    "runtime": "ctranslate2",
    "source": "hf:example/asr-a",
    "size_gb": 1.0,
    "license": "MIT",
    "license_verified": False,
    "commercial_ok": True,
    "phase": 1,
    "purpose": "test",
}
_LLM = {**_ENTRY, "id": "llm-a", "kind": "llm", "runtime": "llama.cpp"}
_PRESET = {
    "asr": "asr-a",
    "verbatim_asr": "asr-a",
    "llm_default": "llm-a",
    "llm_ensemble": ["llm-a"],
    "modes": ["fast"],
}


def _data() -> dict:
    return {
        "model": [copy.deepcopy(_ENTRY), copy.deepcopy(_LLM)],
        "tiers": {tier.value: copy.deepcopy(_PRESET) for tier in Tier},
    }


def test_bundled_manifest_is_valid():
    manifest = load_manifest()
    assert set(manifest.tiers) == set(Tier)


def test_minimal_manifest_parses():
    manifest = parse_manifest(_data())
    assert manifest.tiers[Tier.HIGH].llm_default == "llm-a"


def test_duplicate_ids_rejected():
    data = _data()
    data["model"].append(copy.deepcopy(_ENTRY))
    with pytest.raises(ManifestError, match="duplicate"):
        parse_manifest(data)


def test_unknown_kind_rejected():
    data = _data()
    data["model"][0]["kind"] = "telepathy"
    with pytest.raises(ManifestError, match="unknown kind"):
        parse_manifest(data)


def test_tier_referencing_unknown_model_rejected():
    data = _data()
    data["tiers"]["high"]["llm_ensemble"] = ["llm-a", "missing-llm"]
    with pytest.raises(ManifestError, match="missing-llm"):
        parse_manifest(data)


def test_mode_roles_fall_back_to_llm_default():
    data = _data()
    data["model"].append({**copy.deepcopy(_LLM), "id": "llm-b"})
    data["tiers"]["high"].update(modes=["fast", "best"], default_mode="best",
                                 roles={"best": {"events": ["llm-a", "llm-b"]}})
    preset = parse_manifest(data).tiers[Tier.HIGH]
    assert preset.mode_roles("best") == {"events": ["llm-a", "llm-b"]}
    assert preset.mode_roles("fast") == {"events": ["llm-a"]}


def test_roles_of_older_model_files_are_ignored():
    data = _data()
    data["tiers"]["high"].update(roles={"fast": {"events": ["llm-a"], "judges": ["llm-a"]}})
    assert parse_manifest(data).tiers[Tier.HIGH].mode_roles("fast") == {"events": ["llm-a"]}


@pytest.mark.parametrize(("change", "message"), [
    ({"default_mode": "best"}, "default_mode"),
    ({"roles": {"fast": {"events": ["llm-a", "llm-a"]}}}, "twice"),
    ({"roles": {"fast": {"events": ["nope"]}}}, "nope"),
    ({"roles": {"turbo": {}}}, "unknown modes"),
])
def test_bad_roles_rejected(change, message):
    data = _data()
    data["tiers"]["high"].update(change)
    with pytest.raises(ManifestError, match=message):
        parse_manifest(data)


def test_bundled_modes_have_their_models():
    manifest = load_manifest()
    for preset in manifest.tiers.values():
        assert preset.default_mode in preset.modes
        for mode in preset.modes:
            for ids in preset.mode_roles(mode).values():
                assert ids and all(manifest.get(i).kind == "llm" for i in ids)


def test_missing_tier_rejected():
    data = _data()
    del data["tiers"]["cpu"]
    with pytest.raises(ManifestError, match="missing tier"):
        parse_manifest(data)


def test_is_downloaded(tmp_path):
    entry = parse_manifest(_data()).get("asr-a")
    assert not entry.is_downloaded(tmp_path)
    entry.local_dir(tmp_path).mkdir()
    assert not entry.is_downloaded(tmp_path)
    (entry.local_dir(tmp_path) / "model.bin").write_bytes(b"x")
    assert entry.is_downloaded(tmp_path)


def test_an_interrupted_download_is_not_downloaded(tmp_path):
    data = _data()
    data["model"][1]["files"] = ["Model-Q4.gguf", "mmproj-F16.gguf"]
    entry = parse_manifest(data).get("llm-a")
    folder = entry.local_dir(tmp_path)
    # Hugging Face's own files, and a part file still waiting in .cache
    (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (folder / ".cache" / "huggingface" / "download" / "Model-Q4.gguf.incomplete").write_bytes(b"x")
    assert not entry.is_downloaded(tmp_path)
    (folder / "mmproj-F16.gguf").write_bytes(b"x")
    assert not entry.is_downloaded(tmp_path)  # the model itself is still missing
    (folder / "Model-Q4.gguf").write_bytes(b"x")
    assert entry.is_downloaded(tmp_path)


def test_file_patterns_match_like_the_download(tmp_path):
    data = _data()
    data["model"][0]["files"] = ["*.onnx", "onnx/config.json"]
    entry = parse_manifest(data).get("asr-a")
    folder = entry.local_dir(tmp_path)
    (folder / "onnx").mkdir(parents=True)
    (folder / "onnx" / "config.json").write_bytes(b"x")
    assert not entry.is_downloaded(tmp_path)
    (folder / "onnx" / "model.onnx").write_bytes(b"x")  # `*` crosses folders, as in fnmatch
    assert entry.is_downloaded(tmp_path)
