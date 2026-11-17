"""The user's own model file, `<data home>/models.toml`: models added in Settings > Models, LoRA
adapters trained in the app, and the models each mode uses on this machine's tier.

It has the bundled manifest's shape, so the guide (docs/adding-models.md) can show one format:

    [[model]]                      # a new LLM: a GGUF from Hugging Face or a file on disk
    id = "qwen3-14b"
    source = "hf:unsloth/Qwen3-14B-GGUF"
    files = ["Qwen3-14B-UD-Q4_K_XL.gguf"]

    [[model]]                      # same id as a bundled model: only the keys given change
    id = "gemma4-12b"
    serve = { context = 12288 }

    [tiers.high.roles.best]        # the models of one mode
    events = ["gemma4-26b-a4b", "qwen3-14b"]

A new LLM only needs `id` and `source`; the rest has defaults (kind "llm", runtime llama.cpp).
"""

from __future__ import annotations

import copy
import json
import os
import tomllib
from pathlib import Path

USER_FILE = "models.toml"
HEADER = (
    "# Your models and your choice of models per mode (Settings > Models; docs/adding-models.md).\n"
    "# Same format as the bundled manifest. The app rewrites this file when you change models\n"
    "# in it, so comments you add here are not kept.\n"
)
LLM_DEFAULTS = {
    "kind": "llm", "runtime": "llama.cpp", "size_gb": 0.0, "license": "unknown",
    "license_verified": False, "commercial_ok": False, "phase": 3, "purpose": "",
}
# Keys of a bundled model the user may change (its weights and licence stay as shipped).
TUNABLE = {"serve", "train_base", "purpose"}
ROLE_KEYS = ("events",)


def user_file(home: Path) -> Path:
    return home / USER_FILE


def read_user(home: Path) -> dict:
    path = user_file(home)
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def merge(bundled: dict, user: dict) -> dict:
    """Bundled manifest data + the user's file -> data for `parse_manifest`."""
    data = copy.deepcopy(bundled)
    by_id = {m["id"]: m for m in data.get("model", [])}
    for raw in user.get("model", []):
        if not isinstance(raw, dict) or not raw.get("id"):
            raise ValueError("every [[model]] needs an id")
        known = by_id.get(raw["id"])
        if known is not None:  # a bundled model: tune it
            for key in TUNABLE & set(raw):
                known[key] = copy.deepcopy(raw[key])
            continue
        if not raw.get("source"):
            raise ValueError(f"{raw['id']}: needs a source (hf:<repo>, file:<path>)")
        entry = {**LLM_DEFAULTS, **copy.deepcopy(raw)}
        entry.pop("base_entry", None)
        entry["origin"] = "trained" if entry["source"].startswith("trained:") else "user"
        data.setdefault("model", []).append(entry)
        by_id[entry["id"]] = entry
    for tier, given in (user.get("tiers") or {}).items():
        preset = data.get("tiers", {}).get(tier)
        if preset is None or not isinstance(given, dict):
            raise ValueError(f"unknown tier {tier!r}")
        if "default_mode" in given:
            preset["default_mode"] = given["default_mode"]
        for mode, roles in (given.get("roles") or {}).items():
            current = preset.setdefault("roles", {}).setdefault(mode, {})
            for role, ids in roles.items():
                current[role] = list(ids)
            # Models a mode uses are part of the tier.
            for ids in roles.values():
                preset["llm_ensemble"] = list(dict.fromkeys([*preset["llm_ensemble"], *ids]))
    return data


# ---------------------------------------------------------------------- writing


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a valid TOML basic string
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"can't write {type(value).__name__} to TOML")


def _key(key: str) -> str:
    return key if key.replace("_", "").replace("-", "").isalnum() else json.dumps(key)


def dumps(data: dict) -> str:
    out = [HEADER]
    for model in data.get("model", []):
        out.append("[[model]]")
        out += [f"{_key(k)} = {_value(v)}" for k, v in model.items() if v is not None]
        out.append("")
    for tier, given in (data.get("tiers") or {}).items():
        scalars = {k: v for k, v in given.items() if k != "roles"}
        if scalars:
            out.append(f"[tiers.{tier}]")
            out += [f"{_key(k)} = {_value(v)}" for k, v in scalars.items()]
            out.append("")
        for mode, roles in (given.get("roles") or {}).items():
            out.append(f"[tiers.{tier}.roles.{mode}]")
            out += [f"{role} = {_value(ids)}" for role, ids in roles.items()]
            out.append("")
    return "\n".join(out)


def write_user(home: Path, data: dict) -> None:
    """Write the file, checked first: the merged manifest must still be valid."""
    from stream_log.engine.models.manifest import bundled_data, parse_manifest

    parse_manifest(merge(bundled_data(), data))
    path = user_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(USER_FILE + ".part")
    tmp.write_text(dumps(data), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------- changes


def add_model(home: Path, entry: dict) -> None:
    """Add or replace one of the user's models (or tune a bundled one)."""
    data = read_user(home)
    models = [m for m in data.get("model", []) if m.get("id") != entry["id"]]
    data["model"] = [*models, {k: v for k, v in entry.items() if v not in (None, "", {}, [])}]
    write_user(home, data)


def remove_model(home: Path, model_id: str) -> bool:
    """Forget a model the user added (and every mode choice that names it). Its files stay."""
    data = read_user(home)
    models = data.get("model", [])
    kept = [m for m in models if m.get("id") != model_id]
    if len(kept) == len(models):
        return False
    data["model"] = kept
    for given in (data.get("tiers") or {}).values():
        for mode, roles in list((given.get("roles") or {}).items()):
            for role, ids in list(roles.items()):
                roles[role] = [i for i in ids if i != model_id]
                if not roles[role]:
                    del roles[role]
            if not roles:
                del given["roles"][mode]
    write_user(home, data)
    return True


def set_roles(home: Path, tier: str, mode: str, roles: dict[str, list[str]] | None) -> None:
    """The models of one mode on one tier; None goes back to the bundled choice."""
    data = read_user(home)
    tiers = data.setdefault("tiers", {})
    given = tiers.setdefault(tier, {})
    modes = given.setdefault("roles", {})
    if roles is None:
        modes.pop(mode, None)
    else:
        unknown = set(roles) - set(ROLE_KEYS)
        if unknown:
            raise ValueError(f"roles are {list(ROLE_KEYS)}, not {sorted(unknown)}")
        empty = [role for role, ids in roles.items() if not ids]
        if empty:
            raise ValueError(f"a role needs at least one model: {empty}")
        modes[mode] = {role: list(dict.fromkeys(ids)) for role, ids in roles.items()}
    if not modes:
        given.pop("roles")
    if not given:
        tiers.pop(tier)
    write_user(home, data)
