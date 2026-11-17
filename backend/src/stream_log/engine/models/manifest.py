"""The model manifest: what the app may download or run, with licence status.

The bundled `manifest.toml` is the base. `<data home>/models.toml` (`models/user.py`, written
by Settings > Models and the training screen) adds the user's own models, LoRA adapters trained
in the app, and their choice of models per mode; `load_manifest()` returns both merged.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from importlib import resources
from pathlib import Path

from stream_log.engine.hardware import Tier

KNOWN_KINDS = {
    "asr",
    "align",
    "vad",
    "speaker",
    "audio-emotion",
    "audio-events",
    "denoise",
    "separation",
    "text-emotion",
    "text-embedding",
    "vision-text",
    "ocr",
    "face",
    "llm",
}
KNOWN_MODES = {"fast", "balanced", "best"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
ROLES = ("events",)  # the LLMs that extract events and cross-check each other


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ModelEntry:
    id: str
    kind: str
    runtime: str
    source: str
    size_gb: float
    license: str
    license_verified: bool
    commercial_ok: bool
    phase: int
    purpose: str
    # Glob patterns to fetch from the source repo; empty means everything.
    files: list[str] = field(default_factory=list)
    # llama-server overrides for LLMs, e.g. {n_cpu_moe = 16} when automatic fitting is slower;
    # also `context` (tokens), `always_reasons` (thinks before every answer, like gpt-oss) and
    # `extra_args` (more llama-server flags). See docs/adding-models.md.
    serve: dict = field(default_factory=dict)
    # A LoRA adapter (source "trained:<run>"): the id of the LLM whose GGUF it applies to.
    base: str = ""
    # Hugging Face repo of the full-precision weights, for training a LoRA on this LLM.
    train_base: str = ""
    # Where the entry comes from: "bundled" (manifest.toml), "user" (added in Settings) or
    # "trained" (a LoRA adapter made in the app). Set when loading, not written in TOML.
    origin: str = "bundled"
    base_entry: ModelEntry | None = field(default=None, compare=False, repr=False)

    @property
    def source_chosen(self) -> bool:
        return bool(self.source)

    @property
    def is_adapter(self) -> bool:
        return bool(self.base)

    @property
    def local_file(self) -> Path | None:
        """A model used where it lies (source "file:<path>"), not copied into the models folder."""
        return Path(self.source[len("file:"):]) if self.source.startswith("file:") else None

    def local_dir(self, models_root: Path) -> Path:
        return models_root / self.id

    def is_downloaded(self, models_root: Path) -> bool:
        """True when every pattern in `files` matches a finished file (any file when `files`
        is empty). A Hugging Face download keeps its partial files in the folder's `.cache`
        and moves each one out when it is complete, so an interrupted download, which can leave
        nothing but `.cache`, doesn't count."""
        if self.local_file is not None:
            return self.local_file.is_file()
        directory = self.local_dir(models_root)
        present = _finished_files(directory) if directory.is_dir() else []
        if not present or not all(any(fnmatch(name, pattern) for name in present)
                                  for pattern in self.files):
            return False
        return self.base_entry is None or self.base_entry.is_downloaded(models_root)


def _finished_files(directory: Path) -> list[str]:
    """The folder's files as relative POSIX paths (how `files` patterns name them), leaving out
    the download's `.cache`."""
    names = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if relative.parts[0] != ".cache" and path.is_file():
            names.append(relative.as_posix())
    return names


@dataclass(frozen=True)
class TierPreset:
    asr: str
    verbatim_asr: str
    llm_default: str
    llm_ensemble: list[str]
    modes: list[str]
    default_mode: str = "fast"
    # Per mode, the models that extract events (and cross-check each other), preferred first.
    # A mode not listed runs llm_default.
    roles: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def mode_roles(self, mode: str) -> dict[str, list[str]]:
        given = self.roles.get(mode, {})
        return {role: list(given.get(role) or [self.llm_default]) for role in ROLES}


@dataclass(frozen=True)
class Manifest:
    models: dict[str, ModelEntry]
    tiers: dict[Tier, TierPreset]
    # Why <data home>/models.toml was ignored (the bundled manifest is used alone), if it was.
    user_error: str | None = None

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self.models[model_id]
        except KeyError:
            raise ManifestError(f"unknown model {model_id!r} (not in the manifest or "
                                "models.toml)") from None


def parse_manifest(data: dict) -> Manifest:
    models: dict[str, ModelEntry] = {}
    for raw in data.get("model", []):
        entry = ModelEntry(**raw)
        if entry.id in models:
            raise ManifestError(f"duplicate model id: {entry.id}")
        if entry.kind not in KNOWN_KINDS:
            raise ManifestError(f"{entry.id}: unknown kind {entry.kind!r}")
        if not ID.fullmatch(entry.id):
            raise ManifestError(f"{entry.id!r}: ids are letters, digits, '.', '_' and '-'")
        models[entry.id] = entry
    for entry in list(models.values()):
        if not entry.base:
            continue
        base = models.get(entry.base)
        if base is None or base.kind != "llm" or entry.kind != "llm" or base.base:
            raise ManifestError(f"{entry.id}: base {entry.base!r} must be an LLM (not an adapter)")
        models[entry.id] = replace(entry, base_entry=base)

    tiers: dict[Tier, TierPreset] = {}
    for name, raw in data.get("tiers", {}).items():
        preset = TierPreset(**raw)
        role_ids = [i for mode in preset.roles.values() for ids in mode.values() for i in ids]
        referenced = [preset.asr, preset.verbatim_asr, preset.llm_default, *preset.llm_ensemble,
                      *role_ids]
        missing = [model_id for model_id in referenced if model_id not in models]
        if missing:
            raise ManifestError(f"tier {name}: unknown model ids {missing}")
        if preset.llm_default not in preset.llm_ensemble:
            raise ManifestError(f"tier {name}: llm_default must be part of llm_ensemble")
        unknown_modes = (set(preset.modes) | set(preset.roles)) - KNOWN_MODES
        if unknown_modes:
            raise ManifestError(f"tier {name}: unknown modes {sorted(unknown_modes)}")
        if preset.default_mode not in preset.modes:
            raise ManifestError(f"tier {name}: default_mode {preset.default_mode!r} not in modes")
        for mode, roles in preset.roles.items():
            for other in set(roles) - set(ROLES):  # roles of older model files: not used here
                del roles[other]
            if any(len(set(ids)) != len(ids) for ids in roles.values()):
                raise ManifestError(f"tier {name} {mode}: a role lists a model twice")
        tiers[Tier(name)] = preset

    missing_tiers = set(Tier) - set(tiers)
    if missing_tiers:
        raise ManifestError(f"missing tier presets: {sorted(missing_tiers)}")
    return Manifest(models=models, tiers=tiers)


def bundled_data() -> dict:
    text = resources.files("stream_log.engine.models").joinpath("manifest.toml").read_text("utf-8")
    return tomllib.loads(text)


def load_manifest(home: Path | None = None) -> Manifest:
    """The bundled manifest with the user's models.toml merged in (`home`: the data home,
    default the configured one). A broken models.toml is reported, never fatal."""
    from stream_log.engine.config import get_paths
    from stream_log.engine.models.user import merge, read_user

    bundled = bundled_data()
    home = home if home is not None else get_paths().home
    try:
        user = read_user(home)
        return parse_manifest(merge(bundled, user)) if user else parse_manifest(bundled)
    except (ManifestError, ValueError, TypeError, OSError) as error:
        return replace(parse_manifest(bundled), user_error=f"models.toml ignored: {error}")
