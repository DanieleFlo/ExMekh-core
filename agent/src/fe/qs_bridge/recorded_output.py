"""Parse persisted QS output headers at the frontend boundary."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from qs.core.models.tool_output_reference import ToolOutputReference


@dataclass(frozen=True, slots=True)
class RecordedToolOutput:
    name: str
    source: str
    id: str
    body: str
    used_tools: tuple[ToolOutputReference, ...] | None

    @classmethod
    def parse(cls, text: object) -> RecordedToolOutput | None:
        if not isinstance(text, str) or not text.lstrip().startswith("<header>"):
            return None
        header, separator, body = text.partition("</header>")
        if not separator:
            return None
        try:
            root = ET.fromstring(header + separator)
        except ET.ParseError:
            return None
        output_id = (root.findtext("id") or "").strip()
        if not output_id:
            return None
        return cls(
            name=(root.findtext("name") or "").strip(),
            source=(root.findtext("type") or "Tool").strip(),
            id=output_id,
            body=body.lstrip("\r\n"),
            used_tools=cls._references(root),
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
