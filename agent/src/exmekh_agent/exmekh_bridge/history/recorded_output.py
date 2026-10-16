"""Parse persisted output through ExMekh-Lib's public XML metadata interface."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.xml_header import XmlHeader, XmlHeaderContent


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
    def _content(text: object) -> XmlHeaderContent | None:
        if not isinstance(text, str):
            return None
        return XmlHeader.parse_content(text.lstrip())

    @classmethod
    def split_header(cls, text: object) -> tuple[str, str] | None:
        """Separate the recorded envelope without changing tool-authored text."""
        parsed = cls._content(text)
        return None if parsed is None else (parsed.header, parsed.body)

    @classmethod
    def parse(cls, text: object) -> RecordedToolOutput | None:
        parsed = cls._content(text)
        if parsed is None:
            return None
        try:
            output_id = (parsed.field("id") or "").strip()
            if not output_id:
                return None
            name = (parsed.field("name") or "").strip()
            source = (parsed.field("type") or "Tool").strip()
            thinking = parsed.field("thinking") or ""
            summary_mode = parsed.field("summary-mode")
        except (TypeError, ValueError):
            return None
        try:
            references = parsed.output_references()
        except (TypeError, ValueError):
            references = None
        return cls(
            name=name,
            source=source,
            id=output_id,
            body=parsed.body,
            used_tools=references,
            thinking=thinking,
            summary_mode=summary_mode,
        )
