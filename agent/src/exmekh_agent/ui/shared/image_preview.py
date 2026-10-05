"""Clickable image thumbnails shared by the composer and saved messages."""

from nicegui import ui

from exmekh_agent.ui.resources import ui_text


def show_image_preview(url: str, label: str) -> None:
    with ui.dialog().classes("exmekh-image-dialog") as dialog:
        with ui.card().classes(
            "exmekh-image-preview bg-zinc-950 text-zinc-100 p-3 gap-2"
        ):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label(label).classes("text-sm truncate")
                ui.button(icon="close", on_click=dialog.close).props(
                    'flat round dense aria-label="Close image preview"'
                ).tooltip(ui_text("shared.image_preview.close_image_preview"))
            ui.image(url).props("fit=contain no-spinner").classes(
                "exmekh-preview-image"
            )
    dialog.on("hide", dialog.delete)
    dialog.open()


def render_image_thumbnail(url: str, label: str) -> None:
    ui.image(url).props("fit=contain no-spinner").classes(
        "exmekh-image-thumbnail rounded border border-zinc-700 cursor-pointer"
    ).on("click", lambda: show_image_preview(url, label)).tooltip(label)
