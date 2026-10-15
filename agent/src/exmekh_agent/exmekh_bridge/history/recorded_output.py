"""Parse persisted ExMekh-Lib output headers at the frontend boundary."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from exmekh_lib.core.models.tool_output_reference import ToolOutputReference


@dataclass(frozen=True, slots=True)
class RecordedToolOutput:
    name: str
    source: str
    id: str
    body: str
    used_tools: tuple[ToolOutputReference, ...] | None
    thinking: str = ""
    summary_mode: str | None = None

    @staticmethod
    def cancelled_before_execution(text: str) -> bool:
        """Recognize a transcript receipt for a generated, undispatched call."""
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return False
        return (
            root.tag == "tool-result"
            and root.findtext("status") == "cancelled"
            and root.findtext("executed") == "false"
        )

    @staticmethod
    def split_header(text: object) -> tuple[str, str] | None:
        """Separate the recorded envelope, leaving body interpretation to callers."""
        if not isinstance(text, str) or not text.lstrip().startswith("<header>"):
            return None
        header, separator, body = text.partition("</header>")
        if not separator:
            return None
        return header + separator, body.lstrip("\r\n")

    @classmethod
    def parse(cls, text: object) -> RecordedToolOutput | None:
        parts = cls.split_header(text)
        if parts is None:
            return None
        header, body = parts
        try:
            root = ET.fromstring(header)
        except ET.ParseError:
            return None
        output_id = (root.findtext("id") or "").strip()
        if not output_id:
            return None
        return cls(
            name=(root.findtext("name") or "").strip(),
            source=(root.findtext("type") or "Tool").strip(),
            id=output_id,
            body=body,
            used_tools=cls._references(root),
            thinking=root.findtext("thinking") or "",
            summary_mode=root.findtext("summary-mode"),
        )

    @staticmethod
    def _references(root: ET.Element) -> tuple[ToolOutputReference, ...] | None:
        used_tools = root.find("used-tools")
        if used_tools is None:
            return None
        try:
            return tuple(
                ToolOutputReference(
                    name=item.findtext("name") or "", id=item.findtext("id") or ""
                )
                for item in used_tools.findall("item")
            )
        except (TypeError, ValueError):
            return None
