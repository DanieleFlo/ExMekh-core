"""Render running and applied context compaction in the chat timeline."""

from exmekh_lib.core.agents.deep_agent import DeepAgentCompactionProgressEvent
from nicegui import ui

from exmekh_agent.exmekh_bridge.history.compaction import RecordedCompaction
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.utils.formatting import format_duration

PRESENTATION = presentation_config("controls/compaction_progress")
CHAT_PRESENTATION = presentation_config("controls/chat_view")


def _render_compaction_heading(text: str) -> None:
    with ui.row().classes(CHAT_PRESENTATION["execution_row.main"]):
        ui.icon(
            PRESENTATION["icon"], size=CHAT_PRESENTATION["execution_row.icon_size"]
        ).classes(PRESENTATION["icon_classes"])
        ui.label(text).classes(CHAT_PRESENTATION["execution_row.title"])


def render_applied_compaction(record: RecordedCompaction) -> None:
    details = []
    duration = record.duration_seconds
    if duration is not None:
        elapsed = f"{duration:.2f}s" if duration < 1 else format_duration(duration)
        details.append(ui_text("chat.compaction.elapsed", elapsed=elapsed))
    count = len(record.replacement_call_ids)
    details.append(
        ui_text("chat.compaction.one_block")
        if count == 1
        else ui_text("chat.compaction.block_count", count=count)
    )
    if record.completed_summaries is not None:
        details.append(
            ui_text("chat.compaction.one_summary")
            if record.completed_summaries == 1
            else ui_text(
                "chat.compaction.summary_count", count=record.completed_summaries
            )
        )
    with ui.row().classes(PRESENTATION["applied"]):
        _render_compaction_heading(
            ui_text("chat.compaction.applied", scope=record.skill_path)
        )
        ui.label(" · ".join(details)).classes(PRESENTATION["applied_details"])


class CompactionProgressView(ui.column):
    def __init__(self) -> None:
        super().__init__()
        self.classes(PRESENTATION["container"])
        self._events: dict[str, DeepAgentCompactionProgressEvent] = {}
        self._bars: dict[str, ui.linear_progress] = {}
        self._labels: dict[str, ui.label] = {}
        self._elapsed: dict[str, ui.label] = {}
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
            self._elapsed.clear()
            with self:
                for operation_id, event in incoming.items():
                    with ui.column().classes(PRESENTATION["row"]):
                        with ui.row().classes(PRESENTATION["heading"]):
                            _render_compaction_heading(
                                ui_text("chat.compaction.title", scope=event.skill_path)
                            )
                            self._elapsed[operation_id] = ui.label().classes(
                                PRESENTATION["elapsed"]
                            )
                        with ui.column().classes(PRESENTATION["body"]):
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
            self._elapsed[operation_id].set_text(
                ui_text(
                    "chat.compaction.elapsed",
                    elapsed=format_duration(progress.elapsed_seconds()),
                )
            )
            self._labels[operation_id].set_text(
                ui_text(
                    "chat.compaction.details",
                    counter=ui_text(
                        "chat.compaction.counter",
                        phase=ui_text(f"chat.compaction.{phase}"),
                        completed=progress.completed_outputs,
                        total=progress.total_outputs,
                    ),
                    remaining=ui_text("chat.compaction.estimating")
                    if remaining is None
                    else ui_text(
                        "chat.compaction.remaining",
                        remaining=format_duration(remaining),
                    ),
                )
            )
