"""Format recorded output data independently of its HTML presentation."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

from exmekh_agent.exmekh_bridge.history.recorded_output import RecordedToolOutput
from exmekh_agent.ui.resources import render_html, ui_text


def _format_xml(body: str) -> str:
    try:
        document = ET.fromstring(body)
        root = ET.Element("output")
        root.append(document)
    except ET.ParseError:
        root = ET.fromstring("<output>" + body + "</output>")
    for child in root:
        ET.indent(child, space="  ")
    return (root.text or "") + "\n".join(
        ET.tostring(child, encoding="unicode").rstrip() for child in root
    )


def format_output_text(body: str) -> str:
    """Indent JSON and XML when valid, preserving unstructured output verbatim."""
    try:
        return json.dumps(json.loads(body), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        if body.lstrip().startswith("<"):
            try:
                return _format_xml(body)
            except ET.ParseError:
                pass
        return body


def format_output_html(text: str | None) -> str:
    unavailable = ""
    recorded = None
    body = ""
    if text is None:
        unavailable = ui_text("shared.output_format.unavailable")
    elif not text:
        unavailable = ui_text("shared.output_format.empty")
    else:
        recorded = RecordedToolOutput.parse(text)
        body = format_output_text(recorded.body if recorded else text)
    return render_html(
        "shared/output.html", unavailable=unavailable, recorded=recorded, body=body
    )
