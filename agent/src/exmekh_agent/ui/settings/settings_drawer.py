from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMX
from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.config.execution import DEFAULTS as EXECUTION_DEFAULTS
from exmekh_agent.config.provider import DEFAULTS as PROVIDER_DEFAULTS
from exmekh_agent.exmekh_bridge.providers.provider_health import check_provider_health
from exmekh_agent.state.temp_storage import TempStorage, TempStorageStats
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.shared.file_icons import get_file_icon
from exmekh_agent.utils.formatting import format_file_size

UI_PRESENTATION = presentation_config("controls/settings_drawer")


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
            UI_PRESENTATION[
                "build.classes.global_settings_drawer_bg_zinc_900_border_l_b"
            ]
        ) as drawer:
            self.drawer = drawer
            with ui.row().classes(
                UI_PRESENTATION[
                    "build.classes.items_center_justify_between_w_full_mb_4"
                ]
            ):
                ui.label(ui_text("settings.settings_drawer.global_settings")).classes(
                    UI_PRESENTATION["build.classes.font_bold_text_zinc_100_text_sm"]
                )
                ui.button(
                    icon=UI_PRESENTATION["build.icon.close"], on_click=self.close
                ).props(UI_PRESENTATION["build.props.flat_round_dense_size_sm"])

            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.w_full_gap_4_overflow_y_auto_max_h_calc_100vh"
                ]
            ):
                ui.label(ui_text("settings.settings_drawer.local_provider")).classes(
                    UI_PRESENTATION[
                        "build.classes.text_11px_font_bold_text_zinc_400_tracking_wi"
                    ]
                )
                with ui.row().classes(
                    UI_PRESENTATION[
                        "build.classes.header_provider_items_center_gap_2_w_full_bg_"
                    ]
                ):
                    self.status_spinner = ui.spinner(
                        UI_PRESENTATION["build.spinner.dots"],
                        size=UI_PRESENTATION["build.size.xs"],
                        color=UI_PRESENTATION["build.color.grey_5"],
                    )
                    self.status_badge = ui.badge(
                        ui_text("settings.settings_drawer.checking"),
                        color=UI_PRESENTATION["build.color.grey_8"],
                    ).classes(UI_PRESENTATION["build.classes.text_xs"])
                    self.model_select = (
                        ui.select(
                            options=[config.default_model],
                            value=config.default_model,
                        )
                        .props(
                            UI_PRESENTATION[
                                "build.props.dense_borderless_dark_options_dense"
                            ]
                        )
                        .classes(
                            UI_PRESENTATION[
                                "build.classes.header_model_min_w_0_flex_grow_text_xs"
                            ]
                        )
                    )
                    self.refresh_btn = ui.button(
                        icon=UI_PRESENTATION["build.icon.refresh"],
                        on_click=self.refresh_models,
                    ).props(UI_PRESENTATION["build.props.flat_round_dense_size_sm"])

                self.input_base_url = (
                    ui.input(
                        ui_text("settings.settings_drawer.base_url"),
                        value=config.base_url,
                    )
                    .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                    .classes(UI_PRESENTATION["build.classes.w_full_text_xs"])
                )
                self.input_api_key = (
                    ui.input(
                        ui_text("settings.settings_drawer.api_key"),
                        value=config.api_key,
                        password=True,
                    )
                    .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                    .classes(UI_PRESENTATION["build.classes.w_full_text_xs"])
                )
                self.select_api_type = (
                    ui.select(
                        label=ui_text("settings.settings_drawer.api_protocol"),
                        options=PROVIDER_DEFAULTS["api_types"],
                        value=config.api_type,
                    )
                    .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                    .classes(UI_PRESENTATION["build.classes.w_full_text_xs"])
                )
                self.select_reasoning = (
                    ui.select(
                        label=ui_text(
                            "settings.settings_drawer.reasoning_effort_thinking"
                        ),
                        options=[e.value for e in ReasoningEffortLMX],
                        value=config.reasoning_effort,
                    )
                    .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                    .classes(UI_PRESENTATION["build.classes.w_full_text_xs"])
                )
                self.save_button = (
                    ui.button(
                        ui_text("settings.settings_drawer.save_global_settings"),
                        on_click=self._save_settings,
                    )
                    .props(UI_PRESENTATION["build.props.no_caps_icon_check"])
                    .classes(
                        UI_PRESENTATION[
                            "build.classes.w_full_bg_zinc_800_text_zinc_100_border_borde"
                        ]
                    )
                )

                # --- TEMPORARY STORAGE SECTION ---
                ui.separator().classes(
                    UI_PRESENTATION["build.classes.bg_zinc_800_my_2"]
                )
                with ui.row().classes(
                    UI_PRESENTATION["build.classes.items_center_justify_between_w_full"]
                ):
                    ui.label(
                        ui_text("settings.settings_drawer.temporary_files_tmp")
                    ).classes(
                        UI_PRESENTATION[
                            "build.classes.text_11px_font_bold_text_zinc_400_tracking_wi"
                        ]
                    )
                    self.tmp_badge = ui.badge(
                        ui_text("settings.settings_drawer.0_files_0_b"),
                        color=UI_PRESENTATION["build.color.zinc_800"],
                    ).classes(
                        UI_PRESENTATION[
                            "build.classes.text_10px_text_zinc_300_font_mono"
                        ]
                    )

                with ui.column().classes(
                    UI_PRESENTATION[
                        "build.classes.w_full_bg_zinc_950_70_border_border_zinc_800_"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "build.classes.items_center_gap_1_5_w_full_text_zinc_400_tex"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION["build.icon.folder"],
                            size=UI_PRESENTATION["build.size.xs"],
                        ).classes(UI_PRESENTATION["build.classes.text_zinc_500"])
                        self.tmp_path_label = ui.label("").classes(
                            UI_PRESENTATION[
                                "build.classes.truncate_font_mono_text_10px_text_zinc_400_fl"
                            ]
                        )

                    self.tmp_files_container = ui.column().classes(
                        UI_PRESENTATION[
                            "build.classes.w_full_max_h_40_overflow_y_auto_gap_1_border_"
                        ]
                    )

                    with ui.row().classes(
                        UI_PRESENTATION["build.classes.w_full_gap_2_mt_1"]
                    ):
                        self.refresh_tmp_btn = (
                            ui.button(
                                ui_text("settings.settings_drawer.refresh"),
                                icon=UI_PRESENTATION["build.icon.refresh"],
                                on_click=self._handle_refresh_tmp,
                            )
                            .props(UI_PRESENTATION["build.props.outline_dense_no_caps"])
                            .classes(
                                UI_PRESENTATION[
                                    "build.classes.flex_1_text_xs_text_zinc_300_border_zinc_700_"
                                ]
                            )
                        )
                        self.clean_tmp_btn = (
                            ui.button(
                                ui_text("settings.settings_drawer.clear"),
                                icon=UI_PRESENTATION["build.icon.delete_sweep"],
                                on_click=self._handle_clean_tmp,
                            )
                            .props(UI_PRESENTATION["build.props.outline_dense_no_caps"])
                            .classes(
                                UI_PRESENTATION[
                                    "build.classes.flex_1_text_xs_text_rose_400_border_rose_900_"
                                ]
                            )
                        )

        ui.timer(0, self.refresh_tmp_stats, once=True)
        ui.timer(
            EXECUTION_DEFAULTS["provider_initial_delay_seconds"],
            self.refresh_models,
            once=True,
        )
        # A server restart can replace the loaded model or its allocated ctx
        # while this page remains open. Refresh metadata without running inference.
        ui.timer(
            EXECUTION_DEFAULTS["provider_refresh_interval_seconds"], self.refresh_models
        )
        return drawer

    async def refresh_models(self) -> None:
        client = ui.context.client
        if self.refresh_btn:
            self.refresh_btn.props(UI_PRESENTATION["refresh_models.props.loading"])
        if self.status_spinner:
            self.status_spinner.set_visibility(True)
        is_alive, models = await check_provider_health(config.base_url, config.api_key)
        if client.is_deleted:
            return
        with client:
            if self.refresh_btn:
                self.refresh_btn.props(remove="loading")
            if self.status_spinner:
                self.status_spinner.set_visibility(False)
            if self.status_badge:
                self.status_badge.text = (
                    ui_text("settings.settings_drawer.online", count=len(models))
                    if is_alive
                    else ui_text("settings.settings_drawer.offline")
                )
                self.status_badge.props(
                    f"color={UI_PRESENTATION['refresh_models.state_classes'] if is_alive else UI_PRESENTATION['refresh_models.state_classes_2']}"
                )
            if is_alive and models:
                config.select_available_model(models)
                if self.model_select:
                    selected = self.model_select.value
                    self.model_select.options = models
                    self.model_select.value = (
                        selected if selected in models else config.default_model
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
            self.tmp_badge.text = ui_text(
                "settings.settings_drawer.file",
                stats_total_files=stats.total_files,
                suffix=suffix,
                size_str=size_str,
            )
            if stats.is_over_limit:
                self.tmp_badge.props(
                    UI_PRESENTATION["render_tmp_stats.props.color_amber_9"]
                )
            else:
                self.tmp_badge.props(
                    UI_PRESENTATION["render_tmp_stats.props.color_zinc_800"]
                )
        if self.clean_tmp_btn:
            if stats.total_files == 0:
                self.clean_tmp_btn.props(
                    UI_PRESENTATION["render_tmp_stats.props.disable"]
                )
            else:
                self.clean_tmp_btn.props(remove="disable")
        if self.tmp_files_container:
            self.tmp_files_container.clear()
            with self.tmp_files_container:
                if not stats.files:
                    ui.label(
                        ui_text("settings.settings_drawer.no_temporary_files")
                    ).classes(
                        UI_PRESENTATION[
                            "render_tmp_stats.classes.text_11px_text_zinc_400_italic_py_1"
                        ]
                    )
                else:
                    for file in stats.files:
                        icon_name, color_cls, _ = get_file_icon(file.name)
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_tmp_stats.classes.items_center_justify_between_w_full_py_0_5_px"
                            ]
                        ):
                            with ui.row().classes(
                                UI_PRESENTATION[
                                    "render_tmp_stats.classes.items_center_gap_1_5_min_w_0_flex_grow"
                                ]
                            ):
                                ui.icon(
                                    icon_name,
                                    size=UI_PRESENTATION["render_tmp_stats.size.16px"],
                                ).classes(color_cls)
                                ui.label(file.name).classes(
                                    UI_PRESENTATION[
                                        "render_tmp_stats.classes.truncate_text_11px_text_zinc_300"
                                    ]
                                ).tooltip(file.name)
                            ui.label(format_file_size(file.size)).classes(
                                UI_PRESENTATION[
                                    "render_tmp_stats.classes.text_10px_text_zinc_400_font_mono_shrink_0"
                                ]
                            )

    async def _handle_refresh_tmp(self) -> None:
        client = ui.context.client
        await self.refresh_tmp_stats()
        if client.is_deleted:
            return
        with client:
            ui.notify(
                ui_text(
                    "settings.settings_drawer.temporary_storage_statistics_refreshed"
                ),
                type="info",
            )

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
                ui_text(
                    "settings.settings_drawer.temporary_storage_cleared_file_deleted_freed",
                    deleted_count=deleted_count,
                    suffix=suffix,
                    freed_str=freed_str,
                ),
                type="positive",
            )
        else:
            ui.notify(
                ui_text("settings.settings_drawer.no_temporary_files_to_delete"),
                type="info",
            )

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
            self.save_button.props(UI_PRESENTATION["save_settings.props.loading"])
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
                ui.notify(
                    ui_text("settings.settings_drawer.global_settings_saved"),
                    type="positive",
                )
                self.close()
        finally:
            if self.save_button:
                self.save_button.props(remove="loading")
