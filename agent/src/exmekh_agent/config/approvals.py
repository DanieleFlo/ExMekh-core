"""Approval mode identifiers and compatibility with saved legacy settings."""

TOOL_APPROVAL_MANUAL = "Manual"

TOOL_APPROVAL_ALL = "Approve All"

TOOL_APPROVAL_LLM = "Approve via LLM"

DEFAULT_TOOL_APPROVAL_MODE = TOOL_APPROVAL_MANUAL
TOOL_APPROVAL_MODES: list[str] = [
    TOOL_APPROVAL_MANUAL,
    TOOL_APPROVAL_ALL,
    TOOL_APPROVAL_LLM,
]

LEGACY_TOOL_APPROVAL_MODES: dict[str, str] = {
    "Manuale": TOOL_APPROVAL_MANUAL,
    "Approva tutto": TOOL_APPROVAL_ALL,
    "Approva da un LLM": TOOL_APPROVAL_LLM,
}


def normalize_tool_approval_mode(mode: str | None) -> str:
    """Resolve saved legacy identifiers and use the manual default for unknown modes."""
    if mode in TOOL_APPROVAL_MODES:
        return mode
    return LEGACY_TOOL_APPROVAL_MODES.get(mode, DEFAULT_TOOL_APPROVAL_MODE)
