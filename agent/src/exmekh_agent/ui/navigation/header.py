from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/header")


class HeaderComponent:
    def __init__(
        self,
        on_toggle_drawer: Callable[[], None],
        on_toggle_sidebar: Callable[[], None] | None = None,
    ) -> None:
        self.on_toggle_drawer = on_toggle_drawer
        self.on_toggle_sidebar = on_toggle_sidebar or (lambda: None)

    def build(self) -> None:
        with ui.header().classes(
            UI_PRESENTATION[
                "build.classes.lm_header_items_center_justify_between_px_4_p"
            ]
        ):
            # Left: App branding
            with ui.row().classes(UI_PRESENTATION["build.classes.items_center_gap_3"]):
                ui.button(
                    icon=UI_PRESENTATION["build.icon.menu"],
                    on_click=self.on_toggle_sidebar,
                ).props(
                    UI_PRESENTATION[
                        "build.props.flat_round_dense_aria_label_open_projects_men"
                    ]
                ).classes(UI_PRESENTATION["build.classes.text_zinc_300"])
                ui.icon(
                    UI_PRESENTATION["build.icon.terminal"],
                    size=UI_PRESENTATION["build.size.sm"],
                ).classes(UI_PRESENTATION["build.classes.text_zinc_300"])
                with ui.column().classes(UI_PRESENTATION["build.classes.gap_0"]):
                    ui.label(ui_text("navigation.header.exmekh_agent")).classes(
                        UI_PRESENTATION[
                            "build.classes.font_bold_text_base_tracking_wider_text_zinc_"
                        ]
                    )
                    ui.label(
                        ui_text("navigation.header.hierarchical_agent_engine")
                    ).classes(
                        UI_PRESENTATION[
                            "build.classes.text_10px_text_zinc_400_font_mono"
                        ]
                    )

            # Global provider settings
            with ui.row().classes(
                UI_PRESENTATION["build.classes.header_actions_items_center_gap_3"]
            ):
                ui.button(
                    icon=UI_PRESENTATION["build.icon.tune"],
                    on_click=self.on_toggle_drawer,
                ).props(UI_PRESENTATION["build.props.flat_round_dense"]).classes(
                    UI_PRESENTATION["build.classes.text_zinc_400_hover_text_zinc_200"]
                )
