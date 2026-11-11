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
    job_controls,
    job_poll,
    notify_error,
    page,
    progress,
    row_for_model,
)
from exmekh_agent.utils.formatting import format_file_size


def settings_connection(destination: str, initial_setup: bool) -> None:
    view = ui.context.client
    with page(
        ui_text("system_settings.connect_to_core"),
        ui_text("system_settings.choose_the_server_that_agent_uses_for_inference"),
        destination,
        "connection",
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
            label="Inference protocol",
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

        async def verify() -> None:
            if not verify_button.enabled:
                return
            verify_button.disable()
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
                    verify_button.enable()

        verify_button = ui.button(
            ui_text("system_settings.verify_and_save_connection"), on_click=verify
        ).props(STYLES["props.unelevated_no_caps"])


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
            label="Reasoning effort",
        ).classes(STYLES["classes.w_full_max_w_lg"])
        threshold = ui.number(
            ui_text("system_settings.automatic_compaction"),
            value=config.auto_compaction_threshold_percent,
            min=1,
            max=99,
            step=1,
        )
        images = ui.switch(
            ui_text("system_settings.send_generated_images_to_the_agent"),
            value=config.include_generated_images_in_context,
        )

        async def save() -> None:
            if not save_button.enabled:
                return
            value = threshold.value
            if value is None or not 1 <= value <= 99 or int(value) != value:
                notify_error(
                    ValueError(
                        "Enter a whole-number compaction threshold from 1 to 99."
                    )
                )
                return
            save_button.disable()
            try:
                saved = await config.update_async(
                    reasoning_effort=reasoning.value,
                    auto_compaction_threshold_percent=int(value),
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
        "Manage this Core server." if section == "server" else "",
        destination,
        section,
    ):
        status = connection.status
        if not status:
            ui.label(connection.detail or "Core is unavailable.")
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
                saved.get("model_id", "No saved profile"),
            )
            try:
                hardware = await client.hardware()
                block(
                    ui_text("system_settings.server_hardware"),
                    hardware_summary(hardware),
                )
            except CoreProblem as error:
                ui.label(str(error))
            with ui.row():
                ui.link(
                    ui_text("system_settings.reconfigure_startup"),
                    profile_link(saved, destination),
                ).classes(STYLES["classes.text_primary"])
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

            def render(jobs, data):
                with ui.tabs() as tabs:
                    installed = ui.tab(ui_text("system_settings.installed"))
                    catalog = ui.tab(ui_text("system_settings.catalog"))
                selected_tab = (
                    catalog
                    if not any(m["installed"] for m in data["models"])
                    else installed
                )
                with ui.tab_panels(
                    tabs, value=current_tab or selected_tab, on_change=remember_tab
                ).classes(STYLES["classes.w_full_bg_transparent"]):
                    with ui.tab_panel(installed).classes(STYLES["classes.p_0_gap_3"]):
                        if not any(m["installed"] for m in data["models"]):
                            ui.label(ui_text("system_settings.no_verified_models"))
                        for model in data["models"]:
                            if model["installed"]:
                                row_for_model(client, model, jobs)
                                ui.link(
                                    ui_text("system_settings.configure_as_chat_model"),
                                    profile_link(
                                        {"model_id": model["id"]},
                                        destination,
                                        "runtime",
                                    ),
                                )
                    with ui.tab_panel(catalog).classes(STYLES["classes.p_0_gap_3"]):
                        for model in data["models"]:
                            row_for_model(client, model, jobs)

            job_poll(client, render, catalog=True)
        elif section == "features":
            for key, title in FEATURES.items():
                available = active and key in active.available_features
                detail = status.feature_errors.get(key) or (
                    "Available for this profile"
                    if available
                    else "Unavailable for the active profile"
                )
                block(title, detail)
            if saved:
                ui.link(
                    ui_text("system_settings.configure_optional_features"),
                    profile_link(saved, destination, "features"),
                )
        elif section == "activity":

            def render(jobs, _):
                if not jobs:
                    ui.label(ui_text("system_settings.no_recent_activity"))
                for job in jobs:
                    with ui.card().classes(STYLES["classes.w_full_p_4"]):
                        block(f"{job.kind.capitalize()} · {job.model_id}")
                        progress(job, details=True)
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
