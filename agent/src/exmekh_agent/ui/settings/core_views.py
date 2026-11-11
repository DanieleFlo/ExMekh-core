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
from exmekh_agent.ui.settings.core_navigation import SETTINGS, STEPS, route
from exmekh_agent.ui.styles import load_css
from exmekh_agent.utils.formatting import format_file_size

STYLES = presentation_config("controls/system_pages")
PHASES = {
    "checking_profile": ui_text("system_settings.checking_saved_profile"),
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
def page(
    title: str,
    subtitle: str,
    destination: str,
    section: str | None = None,
    *,
    setup_step: str | None = None,
    setup_links: dict[str, str] | None = None,
    connected: bool | None = None,
):
    load_css()
    ui.dark_mode().enable()
    with ui.element("section").classes(STYLES["classes.console"]):
        with ui.element("header").classes(STYLES["classes.topbar"]):
            with ui.row().classes(STYLES["classes.brand"]):
                ui.label(ui_text("system_settings.exmekh_agent"))
                badge = None
                if connected is not None:
                    badge = (
                        ui.label("")
                        .classes(STYLES["classes.status"])
                        .props(STYLES["props.role_status"])
                    )
                    show_connection(badge, connected)
            key = "back_to_chat" if section else "system_settings"
            target = destination if section else route("/settings/server", destination)
            ui.link(ui_text(f"system_settings.{key}"), target).classes(
                STYLES["classes.header_link"]
            )
        has_navigation = section or setup_step
        layout = (
            STYLES["classes.settings_layout"]
            if has_navigation
            else STYLES["classes.w_full"]
        )
        with ui.element("div").classes(layout):
            if has_navigation:
                labels = (
                    SETTINGS
                    if section
                    else {**STEPS, "startup": ui_text("system_settings.startup_step")}
                )
                navigation_label = ui_text(
                    "system_settings.settings_sections"
                    if section
                    else "system_settings.setup_steps"
                )
                with (
                    ui.element("nav")
                    .classes(STYLES["classes.settings_navigation"])
                    .props(f'aria-label="{navigation_label}"')
                ):
                    for key, label in labels.items():
                        if key == (section or setup_step):
                            ui.label(label).props(
                                STYLES[
                                    "props.aria_current_page"
                                    if section
                                    else "props.aria_current_step"
                                ]
                            )
                            continue
                        target = (
                            route(f"/settings/{key}", destination)
                            if section
                            else (setup_links or {}).get(key)
                        )
                        if target:
                            ui.link(label, target)
                        else:
                            ui.label(label).props(STYLES["props.aria_disabled"])
            with ui.column().classes(STYLES["classes.settings_body"]):
                with ui.column().classes(STYLES["classes.w_full_gap_1"]):
                    ui.label(title).classes(STYLES["classes.heading"]).props(
                        STYLES["props.heading"]
                    )
                    if subtitle:
                        ui.label(subtitle).classes(STYLES["classes.subtitle"])
                yield badge


def show_connection(badge, connected: bool) -> None:
    key = "core_connected" if connected else "connecting_to_core"
    badge.set_text(ui_text(f"system_settings.{key}"))
    badge.classes(
        add=STYLES["classes.connected"] if connected else None,
        remove=None if connected else STYLES["classes.connected"],
    )


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
    metrics = (
        [ui_text("system_settings.elapsed_time", duration=elapsed)]
        if state != "queued"
        else []
    )
    if job.active:
        percent = data.percent if data else None
        if percent is None:
            ui.linear_progress(show_value=False).classes(
                STYLES["classes.progress"]
            ).props(STYLES["props.indeterminate_aria_label_progress_unavailable"])
        else:
            ui.linear_progress(percent / 100, show_value=False).classes(
                STYLES["classes.progress"]
            ).props(f'aria-label="{label}"')
            metrics.insert(
                0,
                ui_text("system_settings.milestone_progress", percent=percent)
                if data.unit == "milestones"
                else f"{percent:g}%",
            )
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
            key = (
                "estimated_load_time"
                if data.unit == "milestones"
                else "estimated_transfer_time"
            )
            metrics.append(
                ui_text(f"system_settings.{key}", duration=timespan(data.eta_seconds))
            )
        elif data and data.unit == "milestones" and state == "running":
            metrics.append(ui_text("system_settings.load_estimate_unavailable"))
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
    message = (
        ui.label("")
        .classes(STYLES["classes.text_sm_opacity_75"])
        .props(STYLES["props.role_status"])
    )
    message.set_visibility(False)
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
            message.set_visibility(False)
        except CoreProblem as error:
            message.set_text(
                ui_text("system_settings.reconnecting_with_detail", detail=error)
            )
            message.set_visibility(True)

    ui.timer(1.5, refresh)


def row_for_model(
    client: CoreClient, model: dict, jobs: list[CoreJob], *, choose=None, selected=False
) -> None:
    model_id = model["id"]
    model_jobs = [
        item
        for item in jobs
        if item.model_id == model_id and item.kind in {"download", "import"}
    ]
    latest = model_jobs[0] if model_jobs else None
    job = next((item for item in model_jobs if item.active), latest)

    async def operation(kind: str) -> None:
        try:
            await client.start_job(kind, model_id=model_id)
        except CoreProblem as error:
            notify_error(error)

    enabled = any(mode["available"] for mode in model["modes"])
    with ui.column().classes(STYLES["classes.item"]):
        with ui.row().classes(
            STYLES["classes.w_full_items_center_justify_between_gap_3"]
        ):
            if choose is not None:
                with ui.element("label").classes(STYLES["classes.choice"]):
                    radio = (
                        ui.element("input")
                        .props(STYLES["props.model_radio"])
                        .classes(STYLES["classes.radio"])
                    )
                    if selected:
                        radio.props(STYLES["props.checked"])
                    if not enabled:
                        radio.props(STYLES["props.disabled"])
                    radio.on("change", choose)
                    model_identity(model)
            else:
                model_identity(model)
            with ui.row().classes(STYLES["classes.items_center_gap_2"]):
                if model["installed"]:
                    ui.label(ui_text("system_settings.already_downloaded")).classes(
                        STYLES["classes.installed_status"]
                    )
                elif (choose is None or enabled) and (not job or not job.active):
                    kind = "import" if model["importable"] else "download"
                    label = ui_text(
                        "system_settings.verify_local_file"
                        if kind == "import"
                        else "system_settings.download"
                    )
                    ui.button(label, on_click=lambda: operation(kind)).props(
                        STYLES["props.outline_no_caps"]
                    )
                if choose is None and (not job or not job.active):
                    if (model["installed"] or model["importable"]) and model.get(
                        "removable", False
                    ):
                        with ui.dialog() as confirm, ui.card():
                            ui.label(
                                ui_text(
                                    "system_settings.confirm_model_removal",
                                    filename=model["filename"],
                                )
                            )
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
        if choose is not None and not enabled:
            ui.label(
                ui_text("system_settings.no_qualified_memory_mode_for_this_hardware")
            ).classes(STYLES["classes.opacity_75_text_sm"])


def model_identity(model: dict) -> None:
    details = [format_file_size(model["size_bytes"])]
    if any(
        feature["id"] == "vision" and feature["available"]
        for feature in model.get("features", ())
    ):
        details.append(ui_text("system_settings.vision_with_encoder"))
    with ui.column().classes(STYLES["classes.gap_0"]):
        ui.label(model["name"]).classes(STYLES["classes.item_title"])
        ui.label(" · ".join(details)).classes(STYLES["classes.item_detail"])


def hardware_summary(host: dict) -> str:
    memory = ui_text(
        "system_settings.unified_memory"
        if host["unified_memory"]
        else "system_settings.ram"
    )
    values = [host["backend"].upper()]
    if host["ram_bytes"]:
        values.append(f"{format_file_size(host['ram_bytes'])} {memory}")
    values.extend(
        ui_text(
            "system_settings.device_memory_free",
            device=device["name"],
            size=format_file_size(device["free_bytes"]),
        )
        for device in host["devices"]
    )
    values.append(
        ui_text(
            "system_settings.model_disk_free",
            size=format_file_size(host["disk_free_bytes"]),
        )
    )
    return " · ".join(values)
