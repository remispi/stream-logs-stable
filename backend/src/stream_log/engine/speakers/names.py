"""Names for VOD voices, such as "guest's parent".

With voices named, transcript excerpts given to the LLM can label lines with the names and the
prompt can say who each one is.

Storage: project.json `voices` maps "streamer" or a voice number (the speakers table's, like
`speaker_labels.json`) to {"name", "about"}. The streamer's name can live in the channel's
profile instead (`streamer_name`, `streamer_about`), so every VOD of the channel gets it; a
project's own "streamer" entry wins.
"""

from __future__ import annotations

import re

from stream_log.engine.config import Paths
from stream_log.engine.project import Project, read_json
from stream_log.engine.speakers.profiles import ProfileError, create_profile, load_profile

NAME_MAX = 40
ABOUT_MAX = 200
ROLES = {
    "streamer": "the streamer",
    "guest": "someone else on the stream",
    "media": "audio from the game or a video",
    "tts": "a text-to-speech voice reading chat messages",
}


def _squash(text: object, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit].strip()


def clean_name(text: object) -> str:
    """One line, no characters that would break `name: text` transcript lines or the quotes."""
    return _squash(re.sub(r"[:\"\[\]{}<>]", "", str(text or "")), NAME_MAX)


def clean_about(text: object) -> str:
    return _squash(str(text or "").replace('"', "'"), ABOUT_MAX)


def _profile(paths: Paths, project: Project) -> tuple[str | None, dict]:
    name = project.meta.get("profile")
    if not name:
        return None, {}
    try:
        return name, load_profile(paths, name)
    except ProfileError:
        return name, {}


def voice_names(paths: Paths, project: Project) -> dict[str, dict]:
    """{"streamer" | voice number: {"name", "about", "channel"}}; channel is the profile's name
    when the name comes from it."""
    out = {key: {"name": clean_name(v.get("name")), "about": clean_about(v.get("about")),
                 "channel": None}
           for key, v in (project.meta.get("voices") or {}).items()
           if isinstance(v, dict) and clean_name(v.get("name"))}
    profile, meta = _profile(paths, project)
    if "streamer" not in out and clean_name(meta.get("streamer_name")):
        out["streamer"] = {"name": clean_name(meta["streamer_name"]),
                           "about": clean_about(meta.get("streamer_about")), "channel": profile}
    return out


def set_name(paths: Paths, project: Project, voice: str, name: str, about: str = "",
             *, channel: bool = True) -> str:
    """Name a voice ("streamer" or a voice number); an empty name removes it. The streamer's name
    goes to the project's profile when it has one and `channel` is set. Returns where it went."""
    name, about = clean_name(name), clean_about(about)
    speakers = project.path("speakers.json")
    if voice != "streamer" and speakers.exists() and _cluster_tags(read_json(speakers)).get(
            voice) == "streamer":
        voice = "streamer"  # the streamer's voice by its number
    voices = dict(project.meta.get("voices") or {})
    profile, _ = _profile(paths, project)
    if voice == "streamer" and channel and profile:
        create_profile(paths, profile, streamer_name=name, streamer_about=about)
        voices.pop("streamer", None)  # the channel's name is the one now in use
        where = f"profile {profile} (every VOD with it)"
    else:
        if name:
            voices[voice] = {"name": name, "about": about}
        else:
            voices.pop(voice, None)
        where = "this project"
    if voices != (project.meta.get("voices") or {}):
        project.update_meta(voices=voices)
    return where


def _cluster_tags(speakers: dict) -> dict[str, str]:
    return {str(c["id"]): c["tag"] for c in speakers.get("clusters", [])}


def _named_voices(speakers: dict, names: dict[str, dict]) -> dict[str, dict]:
    """Names of voices that exist in this speakers.json and aren't the streamer's cluster."""
    tags = _cluster_tags(speakers)
    return {k: v for k, v in names.items() if tags.get(k) not in (None, "streamer")}


def speaker_labels(speakers: dict, names: dict[str, dict]) -> dict[int, str]:
    """Each line's label in the prompts: its voice's name when that voice has one, else its tag.
    A voice's name only goes on the lines that kept the voice's tag (a guest's line that matched
    the enrolled voice is the streamer's; a repeated song line inside it is media)."""
    tags = _cluster_tags(speakers)
    voices = _named_voices(speakers, names)
    streamer = names.get("streamer")
    out = {}
    for entry in speakers.get("lines", []):
        tag = entry["tag"]
        k = None if entry.get("cluster") is None else str(entry["cluster"])
        if tag == "streamer":
            named = streamer
        else:
            named = voices.get(k) if k is not None and tags.get(k) == tag else None
        out[entry["id"]] = named["name"] if named else tag
    return out


def voices_note(speakers: dict, names: dict[str, dict]) -> str:
    """Who is who, for the prompt; empty when no voice is named (so stage keys don't change)."""
    tags = _cluster_tags(speakers)
    parts = []

    def describe(entry: dict, role: str) -> str:
        about = f" ({entry['about']})" if entry.get("about") else ""
        return f"\"{entry['name']}\" is {role}{about}."

    if streamer := names.get("streamer"):
        parts.append(describe(streamer, ROLES["streamer"]))
    voices = _named_voices(speakers, names)
    for k in sorted(voices, key=lambda x: int(x) if x.isdigit() else -1):
        parts.append(describe(voices[k], ROLES.get(tags[k], ROLES["guest"])))
    if not parts:
        return ""
    return ("Who is speaking (transcript lines are labelled with these names): " + " ".join(parts)
            + " Call people by these names and say how they relate in what you write, so a "
            "viewer who doesn't know the stream knows who they are.")
