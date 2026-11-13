"""Page destinations and shared navigation rules for the managed Core."""

import re
from urllib.parse import urlencode

from exmekh_agent.exmekh_bridge.providers.core_management import Connection
from exmekh_agent.ui.resources import ui_text

SETTINGS = {
    "server": ui_text("system_settings.server"),
    "connection": ui_text("system_settings.connection"),
    "models": ui_text("system_settings.models"),
    "features": ui_text("system_settings.features"),
    "activity": ui_text("system_settings.activity"),
    "conversations": ui_text("system_settings.conversations"),
    "storage": ui_text("system_settings.temporary_files"),
}
SETTINGS_GROUPS = {
    "server": ("server", "connection", "models", "features", "activity"),
    "client": ("conversations", "storage"),
}
SETTINGS_GROUP_LABELS = {
    group: ui_text(f"system_settings.{group}") for group in SETTINGS_GROUPS
}
STEPS = {
    "model": ui_text("system_settings.chat_model"),
    "runtime": ui_text("system_settings.memory_and_context"),
    "features": ui_text("system_settings.optional_features"),
    "review": ui_text("system_settings.review"),
}
FEATURES = {
    "vision": ui_text("system_settings.analyze_images_in_chat"),
    "images": ui_text("system_settings.generate_and_edit_images"),
    "embeddings": ui_text("system_settings.semantic_search_and_embeddings"),
}
CHAT_PATH = re.compile(r"/chat/[^/?#]+/[^/?#]+")


def chat_destination(value: str) -> str:
    return (
        value
        if value == "/" or (CHAT_PATH.fullmatch(value) and "\\" not in value)
        else "/"
    )


def route(path: str, destination: str = "/", **parameters) -> str:
    query = {key: value for key, value in parameters.items() if value not in ("", None)}
    if destination != "/":
        query["return_to"] = chat_destination(destination)
    return path + ("?" + urlencode(query) if query else "")


def gate(connection: Connection, destination: str) -> str | None:
    if connection.kind == "manual":
        return None
    if not connection.status:
        return route("/settings/connection", destination, setup=1)
    status = connection.status
    if status.ready_profile:
        return None
    if status.state in {"starting", "error"}:
        return route("/startup", destination)
    return route("/setup/model", destination)


def feature_matches_mode(
    capability: dict, mode: str, *, legacy: bool | None = None
) -> bool:
    modes = capability.get("available_modes")
    if modes is not None:
        return mode in modes
    return bool(capability.get("available")) and (
        mode == "vram" if legacy is None else legacy
    )


def profile_link(profile: dict, destination: str, step: str = "model") -> str:
    return route(
        f"/setup/{step}",
        destination,
        model=profile.get("model_id"),
        mode=profile.get("mode_id"),
        tokens=profile.get("context_tokens"),
        features=",".join(profile.get("features", [])),
    )


def settings_group(section: str) -> str:
    return next(
        group for group, sections in SETTINGS_GROUPS.items() if section in sections
    )
