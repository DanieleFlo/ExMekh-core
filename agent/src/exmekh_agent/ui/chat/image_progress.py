"""Small in-place progress overlay shared by all native image models."""

from __future__ import annotations

import time

from exmekh_lib.interfaces.operation_progress import OperationProgress
from nicegui import ui

from exmekh_agent.exmekh_bridge.models import ToolEvent
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.utils.formatting import format_duration

STYLE = presentation_config("controls/image_progress")


class ImageProgressView(ui.column):
    def __init__(self, event: ToolEvent, item_index: int) -> None:
        super().__init__()
        self.classes(STYLE["classes.overlay"])
        self.item_index = item_index
        self._progress: OperationProgress | None = None
        self._elapsed = 0.0
        self._received = time.monotonic()
        self._running = False
        with self:
            self.label = ui.label().classes(STYLE["classes.label"])
        self.show(event)

    def show(self, event: ToolEvent) -> None:
        progress = event.image_progress
        if progress is not None and progress.follows(self._progress):
            self._progress = progress
        elif progress != self._progress:
            return
        elapsed = progress.elapsed_seconds if progress else event.elapsed_seconds()
        now = time.monotonic()
        self._elapsed = max(elapsed, self._elapsed_at(now))
        self._received = now
        self._running = event.image_state == "running" and (
            progress is None or progress.state == "running"
        )
        self.refresh()

    def _elapsed_at(self, now: float) -> float:
        return self._elapsed + (max(0.0, now - self._received) if self._running else 0)

    def refresh(self) -> None:
        progress = self._progress
        fraction = progress.item_fraction(self.item_index) if progress else None
        estimated = (
            progress is not None
            and progress.state != "completed"
            and progress.is_active_item(self.item_index)
            and progress.overall_kind == "estimated"
        )
        if fraction is not None:
            percent = ui_text(
                "chat.image_progress.estimated"
                if estimated
                else "chat.image_progress.percent",
                percent=min(99, int(round(fraction * 100, 8))),
            )
        else:
            percent = ui_text("chat.image_progress.unknown_percent")
        self.label.text = ui_text(
            "chat.image_progress.summary",
            percent=percent,
            elapsed=format_duration(self._elapsed_at(time.monotonic())),
        )
