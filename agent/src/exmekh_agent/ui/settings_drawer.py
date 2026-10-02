from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMX
from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agent_runner import check_provider_health
from exmekh_agent.state.temp_storage import TempStorage, TempStorageStats
from exmekh_agent.ui.file_icons import format_file_size, get_file_icon


class SettingsDrawerComponent:
    """Global provider settings shared by every conversation."""

    def __init__(
        self,
        on_settings_saved: Callable[[], Awaitable[None] | None],
        on_models_refreshed: Callable[[], None] | None = None,
        temp_storage_getter: Callable[[], TempStorage] | None = None,
    ) -> None:
        self.on_settings_saved = on_settings_saved
        self.on_models_refreshed = on_models_refreshed
        self.temp_storage_getter = temp_storage_getter
        self.drawer: ui.right_drawer | None = None
        self.save_button: ui.button | None = None
        self.input_base_url: ui.input | None = None
        self.input_api_key: ui.input | None = None
        self.select_api_type: ui.select | None = None
        self.select_reasoning: ui.select | None = None
        self.model_select: ui.select | None = None
        self.status_badge: ui.badge | None = None
        self.status_spinner: ui.spinner | None = None
        self.refresh_btn: ui.button | None = None
        self.tmp_badge: ui.badge | None = None
        self.tmp_path_label: ui.label | None = None
        self.tmp_files_container: ui.column | None = None
        self.clean_tmp_btn: ui.button | None = None
        self.refresh_tmp_btn: ui.button | None = None

    def build(self) -> ui.right_drawer:
        with ui.right_drawer(value=False).classes(
            "bg-zinc-900 border-l border-zinc-800 p-4 w-80"
        ) as drawer:
            self.drawer = drawer
            with ui.row().classes("items-center justify-between w-full mb-4"):
                ui.label("Global Settings").classes("font-bold text-zinc-100 text-sm")
                ui.button(icon="close", on_click=self.close).props(
                    "flat round dense size=sm"
                )

            with ui.column().classes(
                "w-full gap-4 overflow-y-auto max-h-[calc(100vh-120px)] pr-1"
            ):
                ui.label("LOCAL PROVIDER").classes(
                    "text-[11px] font-bold text-zinc-400 tracking-wider"
                )
                with ui.row().classes(
                    "header-provider items-center gap-2 w-full bg-zinc-950/70 border border-zinc-800 px-2 py-1 rounded-lg"
                ):
                    self.status_spinner = ui.spinner("dots", size="xs", color="grey-5")
                    self.status_badge = ui.badge("Checking...", color="grey-8").classes(
                        "text-xs"
                    )
                    self.model_select = (
                        ui.select(
                            options=[config.default_model],
                            value=config.default_model,
                        )
                        .props("dense borderless dark options-dense")
                        .classes("header-model min-w-0 flex-grow text-xs")
                    )
                    self.refresh_btn = ui.button(
                        icon="refresh", on_click=self.refresh_models
                    ).props("flat round dense size=sm")

                self.input_base_url = (
                    ui.input("Base URL", value=config.base_url)
                    .props("outlined dense dark")
                    .classes("w-full text-xs")
                )
                self.input_api_key = (
                    ui.input("API Key", value=config.api_key, password=True)
                    .props("outlined dense dark")
                    .classes("w-full text-xs")
                )
                self.select_api_type = (
                    ui.select(
                        label="API Protocol",
                        options=["responses", "chat_completions"],
                        value=config.api_type,
                    )
                    .props("outlined dense dark")
                    .classes("w-full text-xs")
                )
                self.select_reasoning = (
                    ui.select(
                        label="Reasoning Effort (Thinking)",
                        options=[e.value for e in ReasoningEffortLMX],
                        value=config.reasoning_effort,
                    )
                    .props("outlined dense dark")
                    .classes("w-full text-xs")
                )
                self.save_button = (
                    ui.button("Save global settings", on_click=self._save_settings)
                    .props("no-caps icon=check")
                    .classes(
                        "w-full bg-zinc-800 text-zinc-100 border border-zinc-700 rounded-lg text-xs"
                    )
                )

                # --- TEMPORARY STORAGE SECTION ---
                ui.separator().classes("bg-zinc-800 my-2")
                with ui.row().classes("items-center justify-between w-full"):
                    ui.label("TEMPORARY FILES (TMP)").classes(
                        "text-[11px] font-bold text-zinc-400 tracking-wider"
                    )
                    self.tmp_badge = ui.badge(
                        "0 files (0 B)", color="zinc-800"
                    ).classes("text-[10px] text-zinc-300 font-mono")

                with ui.column().classes(
                    "w-full bg-zinc-950/70 border border-zinc-800 p-2.5 rounded-lg gap-2"
                ):
                    with ui.row().classes(
                        "items-center gap-1.5 w-full text-zinc-400 text-[11px]"
                    ):
                        ui.icon("folder", size="xs").classes("text-zinc-500")
                        self.tmp_path_label = ui.label("").classes(
                            "truncate font-mono text-[10px] text-zinc-400 flex-grow"
                        )

                    self.tmp_files_container = ui.column().classes(
                        "w-full max-h-40 overflow-y-auto gap-1 border-t border-zinc-800/80 pt-2"
                    )

                    with ui.row().classes("w-full gap-2 mt-1"):
                        self.refresh_tmp_btn = (
                            ui.button(
                                "Refresh",
                                icon="refresh",
                                on_click=self._handle_refresh_tmp,
                            )
                            .props("outline dense no-caps")
                            .classes(
                                "flex-1 text-xs text-zinc-300 border-zinc-700 hover:bg-zinc-800/60 rounded-lg"
                            )
                        )
                        self.clean_tmp_btn = (
                            ui.button(
                                "Clear",
                                icon="delete_sweep",
                                on_click=self._handle_clean_tmp,
                            )
                            .props("outline dense no-caps")
                            .classes(
                                "flex-1 text-xs text-rose-400 border-rose-900/60 hover:bg-rose-950/40 rounded-lg"
                            )
                        )

        ui.timer(0, self.refresh_tmp_stats, once=True)
        ui.timer(0.5, self.refresh_models, once=True)
        return drawer

    async def refresh_models(self) -> None:
        if self.refresh_btn:
            self.refresh_btn.props("loading")
        if self.status_spinner:
            self.status_spinner.set_visibility(True)
        is_alive, models = await check_provider_health(config.base_url, config.api_key)
        if self.refresh_btn:
            self.refresh_btn.props(remove="loading")
        if self.status_spinner:
            self.status_spinner.set_visibility(False)
        if self.status_badge:
            self.status_badge.text = (
                f"Online ({len(models)} models)" if is_alive else "Offline"
            )
            self.status_badge.props(f"color={'emerald-9' if is_alive else 'rose-9'}")
        if is_alive and models and self.model_select:
            self.model_select.options = list(
                dict.fromkeys([config.default_model, *models])
            )
        if self.on_models_refreshed:
            self.on_models_refreshed()

    async def refresh_tmp_stats(self) -> None:
        storage = (
            self.temp_storage_getter() if self.temp_storage_getter else TempStorage()
        )
        client = ui.context.client
        stats = await asyncio.get_running_loop().run_in_executor(
            None,
            storage.stats,
        )
        if client.is_deleted:
            return
        with client:
            self._render_tmp_stats(stats)

    def _render_tmp_stats(self, stats: TempStorageStats) -> None:
        if self.tmp_path_label:
            self.tmp_path_label.text = str(stats.folder_path)
            self.tmp_path_label.tooltip(str(stats.folder_path))
        if self.tmp_badge:
            size_str = format_file_size(stats.total_size)
            suffix = "" if stats.total_files == 1 else "s"
            self.tmp_badge.text = f"{stats.total_files} file{suffix} ({size_str})"
            if stats.is_over_limit:
                self.tmp_badge.props("color=amber-9")
            else:
                self.tmp_badge.props("color=zinc-800")
        if self.clean_tmp_btn:
            if stats.total_files == 0:
                self.clean_tmp_btn.props("disable")
            else:
                self.clean_tmp_btn.props(remove="disable")
        if self.tmp_files_container:
            self.tmp_files_container.clear()
            with self.tmp_files_container:
                if not stats.files:
                    ui.label("No temporary files").classes(
                        "text-[11px] text-zinc-400 italic py-1"
                    )
                else:
                    for file in stats.files:
                        icon_name, color_cls, _ = get_file_icon(file.name)
                        with ui.row().classes(
                            "items-center justify-between w-full py-0.5 px-1 rounded hover:bg-zinc-800/40 text-xs gap-2"
                        ):
                            with ui.row().classes(
                                "items-center gap-1.5 min-w-0 flex-grow"
                            ):
                                ui.icon(icon_name, size="16px").classes(color_cls)
                                ui.label(file.name).classes(
                                    "truncate text-[11px] text-zinc-300"
                                ).tooltip(file.name)
                            ui.label(format_file_size(file.size)).classes(
                                "text-[10px] text-zinc-400 font-mono shrink-0"
                            )

    async def _handle_refresh_tmp(self) -> None:
        client = ui.context.client
        await self.refresh_tmp_stats()
        if client.is_deleted:
            return
        with client:
            ui.notify("Temporary storage statistics refreshed.", type="info")

    async def _handle_clean_tmp(self) -> None:
        storage = (
            self.temp_storage_getter() if self.temp_storage_getter else TempStorage()
        )
        client = ui.context.client
        deleted_count, freed_bytes = await asyncio.get_running_loop().run_in_executor(
            None,
            storage.clean,
        )
        if client.is_deleted:
            return
        await self.refresh_tmp_stats()
        with client:
            self._notify_tmp_cleaned(deleted_count, freed_bytes)

    @staticmethod
    def _notify_tmp_cleaned(deleted_count: int, freed_bytes: int) -> None:
        if deleted_count > 0:
            freed_str = format_file_size(freed_bytes)
            suffix = "" if deleted_count == 1 else "s"
            ui.notify(
                f"Temporary storage cleared: {deleted_count} file{suffix} deleted ({freed_str} freed).",
                type="positive",
            )
        else:
            ui.notify("No temporary files to delete.", type="info")

    def toggle(self) -> None:
        if self.drawer:
            if not self.drawer.visible:
                self.drawer.set_visibility(True)
            self._sync_inputs_from_config()
            ui.timer(0, self.refresh_tmp_stats, once=True)
            self.drawer.toggle()

    def open(self) -> None:
        if self.drawer:
            if not self.drawer.visible:
                self.drawer.set_visibility(True)
            self._sync_inputs_from_config()
            ui.timer(0, self.refresh_tmp_stats, once=True)
            self.drawer.show()

    def close(self) -> None:
        if self.drawer:
            self.drawer.hide()

    def _sync_inputs_from_config(self) -> None:
        if self.input_base_url:
            self.input_base_url.value = config.base_url
        if self.input_api_key:
            self.input_api_key.value = config.api_key
        if self.select_api_type:
            self.select_api_type.value = config.api_type
        if self.select_reasoning:
            self.select_reasoning.value = config.reasoning_effort
        if self.model_select:
            self.model_select.value = config.default_model

    async def _save_settings(self) -> None:
        if self.save_button:
            self.save_button.props("loading")
        try:
            if (
                self.input_base_url
                and self.input_api_key
                and self.select_api_type
                and self.select_reasoning
            ):
                config.update(
                    base_url=self.input_base_url.value,
                    api_key=self.input_api_key.value,
                    api_type=self.select_api_type.value,
                    reasoning_effort=self.select_reasoning.value,
                    default_model=(self.model_select.value or config.default_model)
                    if self.model_select
                    else config.default_model,
                )
                result = self.on_settings_saved()
                if inspect.isawaitable(result):
                    await result
                ui.notify("Global settings saved", type="positive")
                self.close()
        finally:
            if self.save_button:
                self.save_button.props(remove="loading")
