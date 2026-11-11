"""Shared settings layout, model rows and activity rendering."""

from contextlib import contextmanager
from functools import partial

from nicegui import ui

from exmekh_agent.exmekh_bridge.providers.core_management import (
    CoreClient,
    CoreJob,
    CoreProblem,
)
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.settings.core_navigation import SETTINGS, route
from exmekh_agent.ui.styles import load_css
from exmekh_agent.utils.formatting import format_file_size

STYLES = presentation_config("controls/system_pages")
PHASES = {
    "queued": ui_text("system_settings.queued"),
    "downloading": ui_text("system_settings.downloading"),
    "verifying": ui_text("system_settings.verifying_checksum"),
    "preparing_assets": ui_text("system_settings.preparing_model_assets"),
    "building_worker": ui_text("system_settings.building_the_native_server"),
    "draining": ui_text("system_settings.waiting_for_active_requests"),
    "opening_model": ui_text("system_settings.opening_model_weights"),
    "initializing_runtime": ui_text(
        "system_settings.initializing_the_inference_session"
    ),
    "rolling_back": ui_text("system_settings.restoring_the_previous_profile"),
    "completed": ui_text("system_settings.completed"),
}


@contextmanager
def page(title: str, subtitle: str, destination: str, section: str | None = None):
    load_css()
    ui.dark_mode().enable()
    with ui.column().classes(
        STYLES["classes.w_full_max_w_6xl_mx_auto_px_4_sm_px_8_py_6_gap_5"]
    ):
        with ui.row().classes(
            STYLES["classes.w_full_items_center_justify_between_gap_3"]
        ):
            ui.label(ui_text("system_settings.exmekh_agent")).classes(
                STYLES["classes.text_lg_font_semibold"]
            )
            if section is None:
                ui.link(
                    ui_text("system_settings.system_settings"),
                    route("/settings/server", destination),
                ).classes(STYLES["classes.text_primary"])
            else:
                ui.link(ui_text("system_settings.back_to_chat"), destination).classes(
                    STYLES["classes.text_primary"]
                )
        layout = (
            STYLES["classes.settings_layout"] if section else STYLES["classes.w_full"]
        )
        with ui.element("div").classes(layout):
            if section:
                with ui.element("nav").classes(STYLES["classes.settings_navigation"]):
                    for key, label in SETTINGS.items():
                        link = ui.link(
                            label, route(f"/settings/{key}", destination)
                        ).classes(STYLES["classes.rounded_px_3_py_2"])
                        if key == section:
                            link.classes(
                                STYLES[
                                    "classes.font_semibold_bg_primary_15_text_primary"
                                ]
                            ).props(STYLES["props.aria_current_page"])
            with ui.column().classes(STYLES["classes.settings_body"]):
                with ui.column().classes(STYLES["classes.w_full_gap_1"]):
                    ui.label(title).classes(STYLES["classes.text_3xl_font_semibold"])
                    if subtitle:
                        ui.label(subtitle).classes(STYLES["classes.opacity_75"])
                yield


def block(title: str, detail: str = "") -> None:
    with ui.column().classes(STYLES["classes.gap_0"]):
        ui.label(title).classes(STYLES["classes.font_medium"])
        if detail:
            ui.label(detail).classes(
                STYLES["classes.text_sm_opacity_75_whitespace_normal_break_words"]
            )


def fact(title: str, value: str) -> None:
    with ui.row().classes(
        STYLES[
            "classes.w_full_items_baseline_justify_between_gap_2_border_b_border_zinc_600"
        ]
    ):
        ui.label(title).classes(STYLES["classes.opacity_75"])
        ui.label(value).classes(STYLES["classes.font_medium_break_all"])


def notify_error(error: Exception) -> None:
    ui.notify(str(error), type="negative", close_button=True)


def timespan(seconds: float | int | None) -> str:
    value = max(0, int(seconds or 0))
    mins, secs = divmod(value, 60)
    hours, mins = divmod(mins, 60)
    return (
        f"{hours} h {mins} min"
        if hours
        else f"{mins} min {secs} s"
        if mins
        else f"{secs} s"
    )


def progress(job: CoreJob, *, details: bool = False) -> None:
    state = job.state
    phase = job.phase
    label = PHASES.get(phase, phase.replace("_", " ").capitalize())
    if state in {
        "paused",
        "pausing",
        "cancelling",
        "failed",
        "interrupted",
        "cancelled",
    }:
        label = state.capitalize()
    block(label, job.file if details else "")
    data = job.progress
    elapsed = timespan(job.elapsed_seconds)
    metrics = [f"{elapsed} elapsed"] if state != "queued" else []
    if job.active:
        percent = data.percent if data else None
        if percent is None:
            ui.linear_progress(show_value=False).props(
                STYLES["props.indeterminate_aria_label_progress_unavailable"]
            )
        else:
            ui.linear_progress(percent / 100, show_value=False).props(
                f'aria-label="{label}"'
            )
            suffix = " of milestones" if data.unit == "milestones" else ""
            metrics.insert(0, f"{percent:g}%{suffix}")
        if (
            data
            and data.unit == "bytes"
            and data.total is not None
            and data.completed is not None
        ):
            metrics.append(
                f"{format_file_size(data.completed)} / {format_file_size(data.total)}"
            )
        if data and data.bytes_per_second and state == "running":
            metrics.append(f"{format_file_size(data.bytes_per_second)}/s")
        if data and data.eta_seconds is not None and state == "running":
            metrics.append(f"About {timespan(data.eta_seconds)} remaining")
    if metrics:
        ui.label(" · ".join(metrics)).classes(
            STYLES["classes.text_sm_tabular_nums_opacity_75"]
        )
    if job.error:
        ui.label(job.error["message"]).classes(STYLES["classes.text_negative"])


async def job_action(client: CoreClient, job: CoreJob, action: str) -> None:
    try:
        await client.change_job(job.id, action)
    except CoreProblem as error:
        notify_error(error)


def job_controls(client: CoreClient, job: CoreJob) -> None:
    labels = {
        "pause": "system_settings.pause",
        "resume": "system_settings.resume",
        "cancel": "system_settings.cancel_keep_partial_files",
    }
    if not job.allowed_actions:
        return
    with ui.row().classes(STYLES["classes.gap_2"]):
        for action in job.allowed_actions:
            ui.button(
                ui_text(labels[action]),
                on_click=partial(job_action, client, job, action),
            ).props(
                STYLES["props.flat_no_caps"]
                if action == "cancel"
                else STYLES["props.outline_no_caps"]
            )


def job_poll(client: CoreClient, render, *, catalog: bool = False) -> None:
    view = ui.column().classes(STYLES["classes.w_full_gap_4"])
    message = ui.label("").classes(STYLES["classes.text_sm_opacity_75"])
    previous = None

    async def refresh() -> None:
        nonlocal previous
        try:
            jobs = await client.jobs()
            data = await client.catalog() if catalog else None
            state = (jobs, data)
            if state != previous:
                view.clear()
                with view:
                    render(jobs, data)
                previous = state
            message.set_text("")
        except CoreProblem as error:
            message.set_text(f"Reconnecting · {error}")

    ui.timer(1.5, refresh)


def row_for_model(
    client: CoreClient, model: dict, jobs: list[CoreJob], choose=None
) -> None:
    model_id = model["id"]
    job = next(
        (
            j
            for j in jobs
            if j.model_id == model_id and j.kind in {"download", "import"}
        ),
        None,
    )

    async def operation(kind: str) -> None:
        try:
            await client.start_job(kind, model_id=model_id)
        except CoreProblem as error:
            notify_error(error)

    with ui.card().classes(STYLES["classes.w_full_p_4_gap_3"]):
        with ui.row().classes(
            STYLES["classes.w_full_items_center_justify_between_gap_3"]
        ):
            block(model["name"], format_file_size(model["size_bytes"]))
            with ui.row().classes(STYLES["classes.items_center_gap_2"]):
                if model["installed"]:
                    ui.label(ui_text("system_settings.already_downloaded")).classes(
                        STYLES["classes.text_positive_text_sm"]
                    )
                elif not job or not job.active:
                    kind = "import" if model["importable"] else "download"
                    label = "Verify local file" if model["importable"] else "Download"
                    ui.button(label, on_click=lambda: operation(kind)).props(
                        STYLES["props.outline_no_caps"]
                    )
                if choose:
                    enabled = any(mode["available"] for mode in model["modes"])
                    ui.button(
                        ui_text("system_settings.configure"), on_click=choose
                    ).props(STYLES["props.unelevated_no_caps"]).set_enabled(enabled)
                elif not job or not job.active:
                    if (model["installed"] or model["importable"]) and model.get(
                        "removable", False
                    ):
                        with ui.dialog() as confirm, ui.card():
                            ui.label(f"Remove {model['filename']} from Core?")
                            with ui.row():
                                ui.button(
                                    ui_text("system_settings.keep_model"),
                                    on_click=confirm.close,
                                ).props(STYLES["props.flat_no_caps"])

                                async def remove() -> None:
                                    confirm.close()
                                    await operation("remove")

                                ui.button(
                                    ui_text("system_settings.remove"),
                                    on_click=remove,
                                    color="negative",
                                ).props(STYLES["props.no_caps"])
                        ui.button(
                            ui_text("system_settings.remove"), on_click=confirm.open
                        ).props(STYLES["props.flat_no_caps"])
        if job and job.state != "completed":
            progress(job)
            job_controls(client, job)
            if (job.error or {}).get("code") in {
                "RESUME_UNAVAILABLE",
                "CHECKSUM_MISMATCH",
            } and not model["importable"]:
                ui.button(
                    ui_text("system_settings.discard_partial_download"),
                    on_click=lambda: operation("discard"),
                ).props(STYLES["props.flat_no_caps"])
        if choose and not any(mode["available"] for mode in model["modes"]):
            ui.label(
                ui_text("system_settings.no_qualified_memory_mode_for_this_hardware")
            ).classes(STYLES["classes.opacity_75_text_sm"])


def hardware_summary(host: dict) -> str:
    memory = "Unified memory" if host["unified_memory"] else "RAM"
    values = [host["backend"].upper()]
    if host["ram_bytes"]:
        values.append(f"{format_file_size(host['ram_bytes'])} {memory}")
    values.extend(
        f"{device['name']} · {format_file_size(device['free_bytes'])} VRAM free"
        for device in host["devices"]
    )
    values.append(f"{format_file_size(host['disk_free_bytes'])} free on model drive")
    return " · ".join(values)
