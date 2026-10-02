from __future__ import annotations

from collections.abc import Callable

from nicegui import ui


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
            "lm-header items-center justify-between px-4 py-2 z-10 min-h-16"
        ):
            # Left: App branding
            with ui.row().classes("items-center gap-3"):
                ui.button(icon="menu", on_click=self.on_toggle_sidebar).props(
                    "flat round dense aria-label='Open projects menu'"
                ).classes("text-zinc-300")
                ui.icon("terminal", size="sm").classes("text-zinc-300")
                with ui.column().classes("gap-0"):
                    ui.label("ExMekh Agent").classes(
                        "font-bold text-base tracking-wider text-zinc-100"
                    )
                    ui.label("Hierarchical Agent Engine").classes(
                        "text-[10px] text-zinc-400 font-mono"
                    )

            # Global provider settings
            with ui.row().classes("header-actions items-center gap-3"):
                ui.button(icon="tune", on_click=self.on_toggle_drawer).props(
                    "flat round dense"
                ).classes("text-zinc-400 hover:text-zinc-200")
