"""A JSON inspector block owns formatting, updates and inline image copying."""

from __future__ import annotations

from typing import Any

from markupsafe import Markup
from nicegui import ui
from nicegui.events import GenericEventArguments

from exmekh_agent.ui.history.json_format import format_json_to_highlighted_html
from exmekh_agent.ui.resources import RESOURCE_DIR, render_html, ui_text

IMAGE_COPY_JS = (RESOURCE_DIR / "js" / "raw_json_copy.js").read_text(encoding="utf-8")


class RawJsonBlock(ui.html):
    def __init__(self, data: Any) -> None:
        super().__init__(self._formatted_content(data))
        self.on("click", self._handle_image_copy, js_handler=IMAGE_COPY_JS)

    def set_data(self, data: Any) -> None:
        self.set_content(self._formatted_content(data))

    @staticmethod
    def _formatted_content(data: Any) -> str:
        return render_html(
            "shared/raw_json.html",
            content=Markup(format_json_to_highlighted_html(data)),
            extra_classes="",
        )

    def _handle_image_copy(self, event: GenericEventArguments) -> None:
        copied = isinstance(event.args, dict) and event.args.get("copied") is True
        ui.notify(
            ui_text("history.raw_history_view.image_data_copied")
            if copied
            else ui_text("history.raw_history_view.unable_to_copy_image_data"),
            type="positive" if copied else "negative",
        )
