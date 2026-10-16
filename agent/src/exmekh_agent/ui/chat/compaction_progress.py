"""Transient compaction bars in the streaming response, updated by its timer."""

from exmekh_lib.core.agents.deep_agent import DeepAgentCompactionProgressEvent
from nicegui import ui

from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.utils.formatting import format_duration

PRESENTATION = presentation_config("controls/compaction_progress")


class CompactionProgressView(ui.column):
    def __init__(self) -> None:
        super().__init__()
        self.classes(PRESENTATION["container"])
        self._events: dict[str, DeepAgentCompactionProgressEvent] = {}
        self._bars: dict[str, ui.linear_progress] = {}
        self._labels: dict[str, ui.label] = {}
        self.set_visibility(False)

    def show(self, events: list[DeepAgentCompactionProgressEvent]) -> None:
        incoming = {
            event.progress.operation_id: event
            for event in events
            if not event.progress.finished
        }
        if incoming.keys() != self._events.keys():
            self.clear()
            self._bars.clear()
            self._labels.clear()
            with self:
                for operation_id, event in incoming.items():
                    with ui.column().classes(PRESENTATION["row"]):
                        ui.label(
                            ui_text("chat.compaction.title", scope=event.skill_path)
                        ).classes(PRESENTATION["title"])
                        self._bars[operation_id] = ui.linear_progress(
                            value=0, show_value=False, color=PRESENTATION["color"]
                        ).props(
                            PRESENTATION["bar_props_template"].format(
                                label=ui_text("chat.compaction.accessible_label")
                            )
                        )
                        self._labels[operation_id] = ui.label().classes(
                            PRESENTATION["details"]
                        )
        self._events = incoming
        self.set_visibility(bool(incoming))
        self.refresh()

    def refresh(self) -> None:
        for operation_id, event in self._events.items():
            progress = event.progress
            bar = self._bars[operation_id]
            bar.set_value(progress.completed_outputs / max(1, progress.total_outputs))
            if progress.phase == "scoring":
                bar.props(PRESENTATION["indeterminate_props"])
            else:
                bar.props(remove=PRESENTATION["indeterminate_props"])
            phase = (
                "scoring"
                if progress.phase == "scoring"
                else (
                    "finalizing"
                    if progress.completed_outputs == progress.total_outputs
                    else "summarizing"
                )
            )
            remaining = progress.remaining_seconds()
            self._labels[operation_id].set_text(
                ui_text(
                    "chat.compaction.details",
                    counter=ui_text(
                        "chat.compaction.counter",
                        phase=ui_text(f"chat.compaction.{phase}"),
                        completed=progress.completed_outputs,
                        total=progress.total_outputs,
                    ),
                    elapsed=ui_text(
                        "chat.compaction.elapsed",
                        elapsed=format_duration(progress.elapsed_seconds()),
                    ),
                    remaining=ui_text("chat.compaction.estimating")
                    if remaining is None
                    else ui_text(
                        "chat.compaction.remaining",
                        remaining=format_duration(remaining),
                    ),
                )
            )
