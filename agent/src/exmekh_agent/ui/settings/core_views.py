"""Shared settings layout, model rows and activity rendering."""

from contextlib import contextmanager
from functools import partial
from itertools import groupby

from nicegui import ui

from exmekh_agent.exmekh_bridge.providers.core_management import (
    CoreClient,
    CoreJob,
    CoreProblem,
    CoreStatus,
)
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.settings.core_navigation import (
    FEATURES,
    SETTINGS,
    SETTINGS_GROUP_LABELS,
    SETTINGS_GROUPS,
    STEPS,
    profile_link,
    route,
    settings_group,
)
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
def core_frame(
    destination: str,
    *,
    settings: bool = False,
    group: str | None = None,
    show_back: bool = True,
    connected: bool | None = None,
    extra_classes: str = "",
):
    load_css()
    ui.dark_mode().enable()
    with (
        ui.element("section").classes(STYLES["classes.console"]).classes(extra_classes)
    ):
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
            if show_back:
                key = "back_to_chat" if settings else "system_settings"
                target = (
                    destination if settings else route("/settings/server", destination)
                )
                ui.link(ui_text(f"system_settings.{key}"), target).classes(
                    STYLES["classes.header_link"]
                )
        if group:
            label = ui_text("system_settings.settings_groups")
            with (
                ui.element("nav")
                .classes(STYLES["classes.settings_tabs"])
                .props(f'aria-label="{label}"')
            ):
                for key, sections in SETTINGS_GROUPS.items():
                    link = ui.link(
                        SETTINGS_GROUP_LABELS[key],
                        route(f"/settings/{sections[0]}", destination),
                    )
                    if key == group:
                        link.props(STYLES["props.aria_current_page"])
        yield badge


@contextmanager
def page(
    title: str,
    subtitle: str,
    destination: str,
    section: str | None = None,
    *,
    setup_step: str | None = None,
    setup_links: dict[str, str] | None = None,
    configured_setup: bool = False,
    show_back: bool = True,
    connected: bool | None = None,
):
    group = (
        settings_group(section) if section else "server" if configured_setup else None
    )
    links = {}
    with core_frame(
        destination,
        settings=group is not None,
        group=group,
        show_back=show_back,
        connected=connected,
    ):
        has_navigation = section or setup_step
        layout = (
            STYLES["classes.settings_layout"]
            if has_navigation
            else STYLES["classes.w_full"]
        )
        with ui.element("div").classes(layout):
            if has_navigation:
                labels = (
                    {key: SETTINGS[key] for key in SETTINGS_GROUPS[group]}
                    if section
                    else STEPS
                    if configured_setup
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
                            links[key] = ui.link(label, target)
                        else:
                            ui.label(label).props(STYLES["props.aria_disabled"])
            with ui.column().classes(STYLES["classes.settings_body"]):
                with ui.column().classes(STYLES["classes.w_full_gap_1"]):
                    ui.label(title).classes(STYLES["classes.heading"]).props(
                        STYLES["props.heading"]
                    )
                    if subtitle:
                        ui.label(subtitle).classes(STYLES["classes.subtitle"])
                yield links


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


def whole_number(value: str | None) -> int | None:
    try:
        number = float(value or "")
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def inference_activity(
    client: CoreClient, initial: CoreStatus, *, on_busy=None
) -> None:
    with ui.column().classes(STYLES["classes.w_full_gap_1"]) as panel:
        message = ui.label("").props(STYLES["props.role_status"])

        async def stop() -> None:
            button.disable()
            try:
                await client.stop_inference()
                await refresh()
            except CoreProblem as error:
                notify_error(error)
                button.enable()

        button = ui.button(
            ui_text("system_settings.stop_active_requests"), on_click=stop
        ).props(STYLES["props.outline_no_caps"])

    def render(status: CoreStatus) -> None:
        busy = status.active_requests > 0
        panel.set_visibility(busy)
        if busy:
            key = "stopping_requests" if status.stopping_requests else "active_requests"
            message.set_text(
                ui_text(f"system_settings.{key}", count=status.active_requests)
            )
            button.set_enabled(status.stopping_requests < status.active_requests)
        if on_busy:
            on_busy(busy)

    async def refresh() -> None:
        try:
            render(await client.status())
        except CoreProblem:
            pass

    render(initial)
    ui.timer(1.5, refresh, immediate=False)


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


def progress(job: CoreJob, *, details: bool = False, show_error: bool = True) -> None:
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
    elapsed = timespan(
        data.elapsed_seconds
        if data and data.elapsed_seconds is not None
        else job.elapsed_seconds
    )
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
    if show_error and job.error:
        ui.label(job.error["message"]).classes(STYLES["classes.text_negative"])


FILE_STATES = {
    "queued": ("schedule", "queued"),
    "downloading": ("file_download", "downloading"),
    "verifying": ("fact_check", "verifying_checksum"),
    "verified": ("check_circle", "file_verified"),
    "failed": ("error_outline", "file_failed"),
    "interrupted": ("pause_circle_outline", "file_interrupted"),
    "skipped": ("remove_circle_outline", "file_skipped"),
}


def transfer_queue(job: CoreJob) -> None:
    verified = sum(file.state == "verified" for file in job.files)
    ui.label(
        ui_text(
            "system_settings.files_verified_count",
            verified=verified,
            total=len(job.files),
        )
    ).classes(STYLES["classes.transfer_count"])
    for feature, files in groupby(job.files, key=lambda file: file.feature):
        ui.label(FEATURES.get(feature, ui_text("system_settings.chat_model"))).classes(
            STYLES["classes.transfer_group"]
        ).props(STYLES["props.subheading"])
        with ui.element("ul").classes(STYLES["classes.transfer_list"]):
            for file in files:
                icon, key = FILE_STATES.get(file.state, FILE_STATES["queued"])
                label = ui_text(f"system_settings.{key}")
                with (
                    ui.element("li")
                    .classes(STYLES["classes.transfer_file"])
                    .props(f'data-state="{file.state}"')
                ):
                    with ui.row().classes(STYLES["classes.transfer_row"]):
                        ui.icon(icon).props(STYLES["props.decorative_icon"])
                        ui.label(file.name).classes(STYLES["classes.transfer_name"])
                        ui.label(label).classes(STYLES["classes.transfer_state"])
                    data = file.progress
                    details = [format_file_size(file.size_bytes)]
                    if data and data.percent is not None:
                        ui.linear_progress(
                            data.percent / 100, show_value=False
                        ).classes(STYLES["classes.progress"]).props(
                            f'aria-label="{label}"'
                        )
                        if data.completed is not None:
                            details[0] = (
                                f"{format_file_size(data.completed)} / {details[0]}"
                            )
                        details.insert(0, f"{data.percent:g}%")
                        if data.bytes_per_second:
                            details.append(
                                f"{format_file_size(data.bytes_per_second)}/s"
                            )
                        if data.eta_seconds is not None:
                            details.append(
                                ui_text(
                                    "system_settings.estimated_transfer_time",
                                    duration=timespan(data.eta_seconds),
                                )
                            )
                    ui.label(" · ".join(details)).classes(
                        STYLES["classes.transfer_details"]
                    )


def activation_view(job: CoreJob | None, error: dict | None) -> None:
    files = bool(job and job.preparing_files)
    failure = job.error if job else None
    if not failure and (job is None or not job.active):
        failure = error
    if failure or (job and job.state == "failed"):
        icon, title = "error_outline", "core_startup_failed"
    elif job and job.state in {"cancelled", "interrupted"}:
        icon, title = "pause_circle_outline", f"core_startup_{job.state}"
    elif files:
        icon, title = "file_download", "model_files"
    else:
        icon, title = "memory", "core_startup"
    subtitle = "model_files_detail" if files else "loading_model_detail"
    classes = STYLES[
        "classes.activation_files" if files else "classes.activation_loading"
    ]
    with ui.column().classes(classes):
        with ui.row().classes(STYLES["classes.activation_heading"]):
            with ui.element("span").classes(STYLES["classes.activation_icon"]):
                ui.icon(icon).props(STYLES["props.decorative_icon"])
            with ui.column().classes(STYLES["classes.gap_0"]):
                ui.label(ui_text(f"system_settings.{title}")).classes(
                    STYLES["classes.heading"]
                ).props(STYLES["props.heading"])
                if job:
                    ui.label(job.model_name or job.model_id).classes(
                        STYLES["classes.activation_model"]
                    )
                if title in {"core_startup", "model_files"}:
                    ui.label(ui_text(f"system_settings.{subtitle}")).classes(
                        STYLES["classes.subtitle"]
                    )
        if files and job.files:
            transfer_queue(job)
        elif job:
            with ui.column().classes(STYLES["classes.activation_status"]):
                progress(job, details=files, show_error=False)
        if failure:
            ui.label(failure["message"]).classes(STYLES["classes.text_negative"])


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
            inventory = await client.catalog() if catalog else None
            if inventory:
                inventory.pop("hardware", None)
            state = (jobs, inventory)
            if state != previous:
                view.clear()
                with view:
                    render(jobs, inventory)
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
                        removal_button(
                            client,
                            model_id,
                            ui_text("system_settings.remove"),
                            ui_text(
                                "system_settings.confirm_model_removal",
                                filename=model["filename"],
                            ),
                        )
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


def removal_button(
    client: CoreClient, model_id: str, label: str, confirmation: str
) -> None:
    async def remove() -> None:
        confirm.close()
        try:
            await client.start_job("remove", model_id=model_id)
        except CoreProblem as error:
            notify_error(error)

    with ui.dialog() as confirm, ui.card():
        ui.label(confirmation)
        with ui.row():
            ui.button(
                ui_text("system_settings.keep_model"), on_click=confirm.close
            ).props(STYLES["props.flat_no_caps"])
            ui.button(label, on_click=remove, color="negative").props(
                STYLES["props.no_caps"]
            )
    ui.button(label, on_click=confirm.open).props(STYLES["props.flat_no_caps"])


def row_for_optional_model(
    client: CoreClient,
    model: dict,
    jobs: list[CoreJob],
    profile: dict,
    capability: dict,
    destination: str,
) -> None:
    model_id = model["id"]
    features = set(profile.get("features", ()))
    enabled = model_id in features
    compatible = profile.get("mode_id") in capability.get("available_modes", ())
    job = next((item for item in jobs if item.model_id == model_id), None)
    with ui.column().classes(STYLES["classes.item"]):
        with ui.row().classes(
            STYLES["classes.w_full_items_center_justify_between_gap_3"]
        ):
            block(model["name"], FEATURES[model_id])
            with ui.row().classes(STYLES["classes.items_center_gap_2"]):
                if enabled or compatible:
                    next_features = (
                        features - {model_id} if enabled else features | {model_id}
                    )
                    ui.link(
                        ui_text(
                            "system_settings.disable_optional_model"
                            if enabled
                            else "system_settings.enable_optional_model"
                        ),
                        profile_link(
                            {**profile, "features": sorted(next_features)},
                            destination,
                            "review",
                        ),
                    )
                if model["removable"] and not (job and job.active):
                    removal_button(
                        client,
                        model_id,
                        ui_text("system_settings.remove_downloaded_files"),
                        ui_text(
                            "system_settings.confirm_optional_model_removal",
                            name=model["name"],
                        ),
                    )
        details = []
        if model.get("installed"):
            details.append(ui_text("system_settings.already_downloaded"))
        elif model.get("present"):
            details.append(ui_text("system_settings.available_for_verification"))
        elif not model.get("error"):
            details.append(
                ui_text(
                    "system_settings.feature_download_remaining",
                    size=format_file_size(model["missing_bytes"]),
                )
            )
        if model.get("present") and not model["managed"]:
            details.append(ui_text("system_settings.optional_model_local_source"))
        if model["active"]:
            details.append(ui_text("system_settings.optional_model_active"))
        elif enabled:
            details.append(ui_text("system_settings.optional_model_enabled"))
        if details:
            ui.label(" · ".join(details)).classes(STYLES["classes.item_detail"])
        if (
            reason := model.get("activation_error")
            or model.get("error")
            or (
                None
                if compatible
                else capability.get("reason")
                or ui_text("system_settings.optional_model_needs_profile")
            )
        ):
            ui.label(reason).classes(STYLES["classes.item_detail"])
        if job and job.state != "completed":
            progress(job)
            job_controls(client, job)


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
    devices = host["devices"]
    if "execution_device_uuid" in host:
        devices = [
            device
            for device in devices
            if device["uuid"] == host["execution_device_uuid"]
        ]
    values.extend(
        ui_text(
            "system_settings.device_memory_free",
            device=device["name"],
            size=format_file_size(device["free_bytes"]),
        )
        for device in devices
    )
    values.append(
        ui_text(
            "system_settings.model_disk_free",
            size=format_file_size(host["disk_free_bytes"]),
        )
    )
    return " · ".join(values)
