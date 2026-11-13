"""Show active Core operations and a bounded page of operation history."""

from datetime import UTC, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nicegui import ui

from exmekh_agent.exmekh_bridge.providers.core_management import (
    CoreClient,
    CoreJob,
    CoreProblem,
)
from exmekh_agent.ui.resources import ui_text
from exmekh_agent.ui.settings.core_views import (
    STYLES,
    action_button,
    job_controls,
    job_poll,
    notify_error,
    progress,
    transfer_queue,
)

HISTORY_PAGE_SIZE = 10
OPERATION_LABELS = {
    "apply": "operation_apply",
    "restore": "operation_restore",
    "download": "download",
    "import": "verify_local_file",
    "remove": "operation_remove",
    "discard": "discard_partial_download",
}


def activity_view(client: CoreClient) -> None:
    active_jobs: list[CoreJob] = []
    history: list[CoreJob] | None = None
    page = 0
    file_details: dict[str, bool] = {}
    timezone: tzinfo = UTC

    def render_job(job: CoreJob) -> None:
        operation_key = OPERATION_LABELS.get(job.kind)
        operation = (
            ui_text(f"system_settings.{operation_key}")
            if operation_key
            else job.kind.replace("_", " ").capitalize()
        )
        with ui.column().classes(STYLES["classes.item"]):
            with ui.row().classes(
                STYLES["classes.w_full_items_center_justify_between_gap_3"]
            ):
                ui.label(f"{operation} · {job.model_name or job.model_id}").classes(
                    STYLES["classes.item_title"]
                )
                if job.created_at is not None:
                    date = job.created_at.astimezone(timezone).strftime(
                        "%d %b %Y, %H:%M %Z"
                    )
                    timestamp = ui_text(
                        "system_settings.operation_requested_at", date=date
                    )
                else:
                    timestamp = ui_text("system_settings.operation_date_unavailable")
                ui.label(timestamp).classes(STYLES["classes.activity_date"])
            progress(job, details=True)
            if job.files:
                with ui.expansion(
                    ui_text("system_settings.model_files_activity"),
                    value=file_details.get(job.id, False),
                    on_value_change=lambda event: file_details.update(
                        {job.id: event.value}
                    ),
                ).classes(STYLES["classes.w_full"]):
                    transfer_queue(job)
            job_controls(client, job)
            if job.retryable:

                async def retry() -> None:
                    try:
                        await client.start_job(job.kind, model_id=job.model_id)
                    except CoreProblem as error:
                        notify_error(error)

                action_button(ui_text("system_settings.retry"), retry, tone="accent")

    def show_page(target: int) -> None:
        nonlocal page
        page = max(0, min(target, last_history_page()))
        render_history.refresh()

    def last_history_page() -> int:
        return max(0, len(history or ()) - 1) // HISTORY_PAGE_SIZE

    @ui.refreshable
    def render_active() -> None:
        if active_jobs:
            ui.label(ui_text("system_settings.active_operations")).props(
                STYLES["props.subheading"]
            ).classes(STYLES["classes.font_medium"])
            with ui.column().classes(STYLES["classes.surface"]):
                for job in active_jobs:
                    render_job(job)

    @ui.refreshable
    def render_history() -> None:
        if history is None:
            return
        ui.label(ui_text("system_settings.operation_history")).props(
            STYLES["props.subheading"]
        ).classes(STYLES["classes.font_medium"])
        if not history:
            key = "no_completed_operations" if active_jobs else "no_recent_activity"
            ui.label(ui_text(f"system_settings.{key}"))
            return
        offset = page * HISTORY_PAGE_SIZE
        end = min(offset + HISTORY_PAGE_SIZE, len(history))
        if len(history) > HISTORY_PAGE_SIZE:
            with ui.row().classes(
                STYLES["classes.w_full_items_center_justify_between_gap_3"]
            ):
                ui.label(
                    ui_text(
                        "system_settings.operation_history_range",
                        first=offset + 1,
                        last=end,
                        total=len(history),
                    )
                ).props(STYLES["props.role_status"])
                with ui.row().classes(STYLES["classes.gap_2"]):
                    newer = action_button(
                        ui_text("system_settings.operation_newer"),
                        lambda: show_page(page - 1),
                    )
                    newer.set_enabled(page > 0)
                    older = action_button(
                        ui_text("system_settings.operation_older"),
                        lambda: show_page(page + 1),
                    )
                    older.set_enabled(end < len(history))
        with ui.column().classes(STYLES["classes.surface"]):
            for job in history[offset:end]:
                render_job(job)

    def render(jobs: list[CoreJob], _inventory: dict | None) -> None:
        nonlocal active_jobs, history, page
        active = [job for job in jobs if job.active]
        completed = sorted(
            (job for job in jobs if not job.active),
            key=lambda job: (
                job.created_at.timestamp() if job.created_at else float("-inf")
            ),
            reverse=True,
        )
        for job_id in file_details.keys() - {job.id for job in jobs}:
            del file_details[job_id]
        had_active_jobs = bool(active_jobs)
        if active != active_jobs:
            active_jobs = active
            render_active.refresh()
        if completed != history or (not completed and bool(active) != had_active_jobs):
            history = completed
            page = min(page, last_history_page())
            render_history.refresh()

    async def set_browser_timezone() -> None:
        nonlocal timezone
        try:
            name = await ui.run_javascript(
                "Intl.DateTimeFormat().resolvedOptions().timeZone"
            )
            timezone = ZoneInfo(name)
        except (TimeoutError, TypeError, ValueError, ZoneInfoNotFoundError):
            return
        render_active.refresh()
        render_history.refresh()

    render_active()
    render_history()
    ui.timer(0, set_browser_timezone, once=True)
    job_poll(client, render)
