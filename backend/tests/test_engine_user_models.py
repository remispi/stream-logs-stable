import pytest

from stream_log.engine.hardware import Tier
from stream_log.engine.llm_server import model_identity, server_args
from stream_log.engine.models import load_manifest
from stream_log.engine.models.manifest import ManifestError
from stream_log.engine.models.user import add_model, read_user, remove_model, set_roles, user_file


def test_no_file_is_the_bundled_manifest(tmp_path):
    manifest = load_manifest(tmp_path)
    assert manifest.user_error is None
    assert manifest.get("gemma4-12b").origin == "bundled"
    assert manifest.get("gemma4-12b").train_base == "unsloth/gemma-4-12b-it"
    with pytest.raises(ManifestError, match="unknown model"):
        manifest.get("nope")


def test_add_tune_and_remove(tmp_path):
    gguf = tmp_path / "mine.gguf"
    gguf.write_bytes(b"GGUF")
    add_model(tmp_path, {"id": "my-llm", "source": f"file:{gguf}", "serve": {"context": 8192,
                                                                          "always_reasons": True}})
    add_model(tmp_path, {"id": "gemma4-12b", "serve": {"context": 12288}, "files": ["x.gguf"]})
    manifest = load_manifest(tmp_path)
    mine = manifest.get("my-llm")
    assert mine.origin == "user" and mine.kind == "llm" and mine.local_file == gguf
    assert mine.is_downloaded(tmp_path / "models")
    args = server_args(mine, tmp_path / "models")
    assert args[args.index("-m") + 1] == str(gguf) and args[args.index("-c") + 1] == "8192"
    tuned = manifest.get("gemma4-12b")
    assert tuned.serve == {"context": 12288} and tuned.files != ["x.gguf"]  # weights stay
    set_roles(tmp_path, "high", "fast", {"events": ["my-llm"]})
    roles = load_manifest(tmp_path).tiers[Tier.HIGH].mode_roles("fast")
    assert roles == {"events": ["my-llm"]}
    assert remove_model(tmp_path, "my-llm")
    assert "my-llm" not in load_manifest(tmp_path).models
    assert "tiers" not in read_user(tmp_path)  # the mode lost its only change
    assert not remove_model(tmp_path, "my-llm")


def test_a_bad_file_is_reported_not_fatal(tmp_path):
    user_file(tmp_path).write_text('[[model]]\nid = "x"\n', encoding="utf-8")  # no source
    manifest = load_manifest(tmp_path)
    assert manifest.user_error and "source" in manifest.user_error
    assert "gemma4-12b" in manifest.models


def test_writing_checks_the_result(tmp_path):
    with pytest.raises(ManifestError):
        set_roles(tmp_path, "high", "fast", {"events": ["not-a-model"]})
    with pytest.raises(ValueError):
        set_roles(tmp_path, "high", "fast", {"events": []})
    assert not user_file(tmp_path).exists()


def test_adapter_runs_on_its_base(tmp_path):
    models = tmp_path / "models"
    (models / "gemma4-12b").mkdir(parents=True)
    (models / "gemma4-12b" / "gemma-4-12b-it-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    (models / "gemma4-12b" / "mmproj-F16.gguf").write_bytes(b"GGUF")
    (models / "gemma4-12b-lora-1").mkdir()
    add_model(tmp_path, {"id": "gemma4-12b-lora-1", "source": "trained:run", "base": "gemma4-12b"})
    adapter = load_manifest(tmp_path).get("gemma4-12b-lora-1")
    assert adapter.origin == "trained" and not adapter.is_downloaded(models)  # no adapter yet
    (models / "gemma4-12b-lora-1" / "adapter.gguf").write_bytes(b"GGUF-LORA")
    assert adapter.is_downloaded(models)
    args = server_args(adapter, models)
    assert args[args.index("-m") + 1].endswith("gemma-4-12b-it-UD-Q4_K_XL.gguf")
    assert args[args.index("--lora") + 1].endswith("adapter.gguf")
    assert args[args.index("--alias") + 1] == "gemma4-12b-lora-1"
    assert "gemma4-12b-lora-1" in model_identity(adapter, models)


def test_an_adapter_needs_an_llm_base(tmp_path):
    user_file(tmp_path).write_text('[[model]]\nid = "a"\nsource = "trained:x"\nbase = "yunet"\n',
                                   encoding="utf-8")
    assert "base" in load_manifest(tmp_path).user_error
