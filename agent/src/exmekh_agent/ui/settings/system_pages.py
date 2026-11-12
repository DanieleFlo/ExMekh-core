"""System pages for a connection, model inventory and Agent preferences."""

import asyncio
import json

from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMX
from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.config.provider import DEFAULTS as PROVIDER_DEFAULTS
from exmekh_agent.exmekh_bridge.providers.core_management import (
    CoreClient,
    CoreProblem,
    connect_core,
    validated_url,
)
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from exmekh_agent.ui.resources import ui_text
from exmekh_agent.ui.settings.core_navigation import (
    FEATURES,
    SETTINGS,
    STEPS,
    chat_destination,
    gate,
    profile_link,
    route,
)
from exmekh_agent.ui.settings.core_views import (
    STYLES,
    block,
    fact,
    hardware_summary,
    inference_activity,
    job_controls,
    job_poll,
    notify_error,
    page,
    progress,
    row_for_model,
    row_for_optional_model,
    transfer_queue,
    whole_number,
)
from exmekh_agent.utils.formatting import format_file_size


def settings_connection(destination: str, initial_setup: bool) -> None:
    view = ui.context.client
    with page(
        ui_text("system_settings.connect_to_core"),
        ui_text("system_settings.choose_the_server_that_agent_uses_for_inference"),
        destination,
        "connection",
        show_back=not initial_setup,
    ):
        address = (
            ui.input(ui_text("system_settings.api_url"), value=config.base_url)
            .props(STYLES["props.outlined"])
            .classes(STYLES["classes.w_full_max_w_xl"])
        )
        key = (
            ui.input(
                ui_text("system_settings.api_key_or_core_management_token"),
                value=config.api_key,
                password=True,
                password_toggle_button=True,
            )
            .props(STYLES["props.outlined"])
            .classes(STYLES["classes.w_full_max_w_xl"])
        )
        protocol = ui.select(
            PROVIDER_DEFAULTS["api_types"],
            label=ui_text("system_settings.inference_protocol"),
            value=config.api_type,
        ).classes(STYLES["classes.w_full_max_w_xl"])
        status = ui.label("").props(STYLES["props.role_status"])
        if initial_setup:
            ui.label(
                ui_text(
                    "system_settings.start_core_in_its_checkout_then_verify_the_connection"
                )
            )
            with ui.row().classes(STYLES["classes.items_center"]):
                ui.code("./start-ds4-server.sh", language="bash")
                ui.button(
                    ui_text("system_settings.copy_command"),
                    on_click=lambda: ui.clipboard.write("./start-ds4-server.sh"),
                ).props(STYLES["props.flat_no_caps"])

            def using_saved_connection() -> bool:
                return (
                    verify_button.enabled
                    and address.value == config.base_url
                    and key.value == config.api_key
                    and protocol.value == config.api_type
                )

            async def reconnect() -> None:
                if not using_saved_connection():
                    return
                connection = await connect_core()
                if view.is_deleted or not using_saved_connection():
                    return
                if connection.kind == "unavailable":
                    status.set_text(connection.detail)
                    return
                retry.cancel()
                ui.navigate.to(gate(connection, destination) or destination)

        async def verify() -> None:
            if not verify_button.enabled:
                return
            controls = (verify_button, address, key, protocol)
            for control in controls:
                control.disable()
            status.set_text(ui_text("system_settings.checking_connection"))
            try:
                url = validated_url(address.value)
                api_key, api_type = key.value, protocol.value
                if any(char in api_key for char in "\r\n"):
                    raise CoreProblem(ui_text("system_settings.single_line_key"))
                client = CoreClient(url, api_key)
                connection = await client.connect()
                if view.is_deleted:
                    return
                status.set_text(
                    connection.detail or ui_text("system_settings.core_reachable")
                )
                if connection.kind == "unavailable":
                    return
                saved = await config.update_async(
                    base_url=url, api_key=api_key, api_type=api_type
                )
                client.follow(connection)
                if view.is_deleted:
                    return
                if not saved:
                    status.set_text(ui_text("system_settings.connection_save_failed"))
                    return
                ui.navigate.to(
                    gate(connection, destination)
                    or (
                        destination
                        if initial_setup
                        else route("/settings/server", destination)
                    )
                )
            except CoreProblem as error:
                if not view.is_deleted:
                    status.set_text(str(error))
            finally:
                if not view.is_deleted:
                    for control in controls:
                        control.enable()

        verify_button = (
            ui.button(
                ui_text("system_settings.verify_and_save_connection"), on_click=verify
            )
            .props(STYLES["props.unelevated_no_caps"])
            .classes(STYLES["classes.primary"])
        )
        if initial_setup:
            retry = ui.timer(1.5, reconnect, immediate=False)


def client_preferences(destination: str) -> None:
    view = ui.context.client
    with page(
        ui_text("system_settings.conversations"),
        ui_text("system_settings.defaults_for_conversations_in_agent"),
        destination,
        "conversations",
    ):
        reasoning = ui.select(
            [effort.value for effort in ReasoningEffortLMX],
            value=config.reasoning_effort,
            label=ui_text("system_settings.reasoning_effort"),
        ).classes(STYLES["classes.w_full_max_w_lg"])
        threshold = ui.input(
            ui_text("system_settings.automatic_compaction"),
            value=str(config.auto_compaction_threshold_percent),
        ).props(STYLES["props.compaction_threshold"])
        images = ui.switch(
            ui_text("system_settings.send_generated_images_to_the_agent"),
            value=config.include_generated_images_in_context,
        )

        async def save() -> None:
            if not save_button.enabled:
                return
            value = whole_number(threshold.value)
            if value is None or not 1 <= value <= 99:
                notify_error(
                    ValueError(ui_text("system_settings.invalid_compaction_threshold"))
                )
                return
            save_button.disable()
            try:
                saved = await config.update_async(
                    reasoning_effort=reasoning.value,
                    auto_compaction_threshold_percent=value,
                    include_generated_images_in_context=images.value,
                )
                if not view.is_deleted:
                    ui.notify(
                        ui_text(
                            "system_settings.preferences_saved"
                            if saved
                            else "system_settings.preferences_save_failed"
                        ),
                        type="positive" if saved else "negative",
                    )
            finally:
                if not view.is_deleted:
                    save_button.enable()

        save_button = ui.button(
            ui_text("system_settings.save_preferences"), on_click=save
        ).props(STYLES["props.unelevated_no_caps"])


def client_storage(destination: str) -> None:
    session = ChatRoutes(session_manager).resolve(destination)
    storage = session_manager.temp_storage(
        session or session_manager.get_current_session()
    )
    with page(
        ui_text("system_settings.temporary_files"),
        ui_text("system_settings.local_agent_cache_and_staged_attachments"),
        destination,
        "storage",
    ):
        stats = ui.label(ui_text("system_settings.checking_temporary_files"))

        async def refresh() -> None:
            result = await asyncio.to_thread(storage.stats)
            stats.set_text(
                f"{result.total_files} files · {format_file_size(result.total_size)} · {result.folder_path}"
            )
            clear.set_enabled(result.total_files > 0)

        async def clean() -> None:
            confirm.close()
            count, size = await asyncio.to_thread(storage.clean)
            ui.notify(f"Removed {count} files · {format_file_size(size)} freed.")
            await refresh()

        with ui.dialog() as confirm, ui.card():
            ui.label(
                ui_text(
                    "system_settings.clear_temporary_files_this_removes_staged_attachments_from_the_local"
                )
            )
            with ui.row():
                ui.button(
                    ui_text("system_settings.keep_files"), on_click=confirm.close
                ).props(STYLES["props.flat_no_caps"])
                ui.button(
                    ui_text("system_settings.clear_files"),
                    color="negative",
                    on_click=clean,
                ).props(STYLES["props.no_caps"])
        with ui.row():
            ui.button(ui_text("system_settings.refresh"), on_click=refresh).props(
                STYLES["props.outline_no_caps"]
            )
            clear = ui.button(
                ui_text("system_settings.clear_cache"), on_click=confirm.open
            ).props(STYLES["props.outline_no_caps"])
            clear.disable()
        ui.timer(0.1, refresh, once=True)


async def system_settings(section: str, return_to: str, setup: int = 0) -> None:
    destination = chat_destination(return_to)
    if section == "connection":
        settings_connection(destination, bool(setup))
        return
    if section == "conversations":
        client_preferences(destination)
        return
    if section == "storage":
        client_storage(destination)
        return
    connection = await connect_core()
    with page(
        SETTINGS[section],
        ui_text("system_settings.manage_core_server") if section == "server" else "",
        destination,
        section,
        connected=True if connection.status else None,
        show_back=not connection.status or connection.status.chat_ready,
    ):
        status = connection.status
        if not status:
            ui.label(connection.detail or ui_text("system_settings.core_unavailable"))
            if section == "server" and connection.kind == "manual":
                fact(ui_text("system_settings.chat_model"), config.default_model)
                fact(
                    ui_text("system_settings.context"),
                    f"{config.get_context_length():,} tokens",
                )
            ui.link(
                ui_text("system_settings.connection_settings"),
                route("/settings/connection", destination),
            )
            return
        client = CoreClient()
        active = status.active_profile
        saved = status.saved_profile or {}
        profile = saved or (active.model_dump() if active else {})
        if section == "server":
            fact(ui_text("system_settings.connection"), config.base_url)
            fact(
                ui_text("system_settings.chat"),
                active.api_model
                if active and status.chat_ready
                else status.state.capitalize(),
            )
            if active:
                fact(
                    ui_text("system_settings.context"),
                    f"{active.context_tokens:,} tokens",
                )
                fact(ui_text("system_settings.memory_mode"), active.mode_id.upper())
            fact(
                ui_text("system_settings.profile_at_next_startup"),
                saved.get("model_id") or ui_text("system_settings.no_saved_profile"),
            )
            inference_activity(client, status)
            try:
                hardware = await client.hardware()
                block(
                    ui_text("system_settings.server_hardware"),
                    hardware_summary(hardware),
                )
            except CoreProblem as error:
                ui.label(str(error))
            with ui.column().classes(STYLES["classes.w_full_gap_1"]):
                block(ui_text("system_settings.startup_profile"))
                with ui.row():
                    ui.link(
                        ui_text("system_settings.reconfigure_startup"),
                        profile_link(profile, destination),
                    )
                    if profile:
                        for step in ("runtime", "features"):
                            ui.link(
                                STEPS[step], profile_link(profile, destination, step)
                            )
            with ui.row():
                diagnostic = json.dumps(
                    status.model_dump(
                        include={
                            "version",
                            "boot_id",
                            "revision",
                            "state",
                            "saved_profile",
                            "job_id",
                            "error",
                        }
                    ),
                    indent=2,
                )
                ui.button(
                    ui_text("system_settings.copy_diagnostics"),
                    on_click=lambda: ui.clipboard.write(diagnostic),
                ).props(STYLES["props.flat_no_caps"])
                if status.job_id:
                    ui.link(
                        ui_text("system_settings.view_startup"),
                        route("/startup", destination),
                    )
                elif status.error:
                    ui.link(
                        ui_text("system_settings.review_startup_error"),
                        route("/startup", destination),
                    )
        elif section == "models":
            current_tab = None

            def remember_tab(event) -> None:
                nonlocal current_tab
                current_tab = event.value

            def render(jobs, inventory):
                nonlocal current_tab
                models = inventory["models"]
                optional_models = inventory.get("optional_models", ())
                installed_models = [model for model in models if model["installed"]]
                installed_optional = [
                    model for model in optional_models if model.get("present")
                ]
                configured_profile = inventory.get("profile", profile) or {}
                selected = next(
                    (
                        model
                        for model in models
                        if model["id"] == configured_profile.get("model_id")
                    ),
                    {},
                )
                capabilities = {
                    item["id"]: item for item in selected.get("features", ())
                }
                if current_tab is None:
                    current_tab = (
                        "installed"
                        if installed_models or installed_optional
                        else "catalog"
                    )
                with ui.tabs() as tabs:
                    installed = ui.tab(
                        "installed", label=ui_text("system_settings.installed")
                    )
                    catalog = ui.tab(
                        "catalog", label=ui_text("system_settings.catalog")
                    )
                with ui.tab_panels(
                    tabs, value=current_tab, on_change=remember_tab, animated=False
                ).classes(STYLES["classes.w_full_bg_transparent"]):
                    for tab, chat_models, extras in (
                        (installed, installed_models, installed_optional),
                        (catalog, models, optional_models),
                    ):
                        with ui.tab_panel(tab).classes(STYLES["classes.p_0_gap_3"]):
                            if not chat_models and not extras:
                                ui.label(ui_text("system_settings.no_verified_models"))
                            if chat_models:
                                block(ui_text("system_settings.chat_models"))
                                with ui.column().classes(STYLES["classes.surface"]):
                                    for model in chat_models:
                                        row_for_model(client, model, jobs)
                                        if tab is installed:
                                            ui.link(
                                                ui_text(
                                                    "system_settings.configure_as_chat_model"
                                                ),
                                                profile_link(
                                                    configured_profile
                                                    if configured_profile.get(
                                                        "model_id"
                                                    )
                                                    == model["id"]
                                                    else {"model_id": model["id"]},
                                                    destination,
                                                    "runtime",
                                                ),
                                            )
                            if extras:
                                block(ui_text("system_settings.optional_models"))
                                with ui.column().classes(STYLES["classes.surface"]):
                                    for model in extras:
                                        row_for_optional_model(
                                            client,
                                            model,
                                            jobs,
                                            configured_profile,
                                            capabilities.get(model["id"], {}),
                                            destination,
                                        )

            job_poll(client, render, catalog=True)
        elif section == "features":
            ready = status.ready_profile
            for key, title in FEATURES.items():
                available = ready and key in ready.available_features
                detail = status.feature_errors.get(key) or ui_text(
                    "system_settings.feature_available"
                    if available
                    else "system_settings.feature_unavailable"
                )
                block(title, detail)
            if profile:
                ui.link(
                    ui_text("system_settings.configure_optional_features"),
                    profile_link(profile, destination, "features"),
                )
        elif section == "activity":

            def render(jobs, _):
                if not jobs:
                    ui.label(ui_text("system_settings.no_recent_activity"))
                for job in jobs:
                    with ui.card().classes(STYLES["classes.w_full_p_4"]):
                        block(
                            f"{job.kind.capitalize()} · {job.model_name or job.model_id}"
                        )
                        progress(job, details=True)
                        if job.files:
                            with ui.expansion(
                                ui_text("system_settings.model_files_activity"),
                                value=job.active,
                            ).classes(STYLES["classes.w_full"]):
                                transfer_queue(job)
                        job_controls(client, job)
                        if job.retryable:

                            async def retry(item=job) -> None:
                                try:
                                    await client.start_job(
                                        item.kind, model_id=item.model_id
                                    )
                                except CoreProblem as error:
                                    notify_error(error)

                            ui.button(
                                ui_text("system_settings.retry"), on_click=retry
                            ).props(STYLES["props.outline_no_caps"])

            job_poll(client, render)
