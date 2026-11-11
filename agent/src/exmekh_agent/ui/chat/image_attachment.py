"""Render the same image card in the composer, history and message editor."""

from collections.abc import Callable

from nicegui import ui

from exmekh_agent.exmekh_bridge.models.messages import ImageDestination
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.shared.file_icons import format_file_size, get_file_icon
from exmekh_agent.ui.shared.image_preview import render_image_thumbnail

UI_PRESENTATION = presentation_config("controls/input_bar")


def render_image_attachment(
    url: str,
    filename: str,
    file_size: int | None,
    destination: ImageDestination,
    *,
    card_class: str,
    on_change: Callable[[ImageDestination], None] | None = None,
    on_remove: Callable[[], None] | None = None,
) -> None:
    with (
        ui.row()
        .classes(UI_PRESENTATION["image_destination.card_classes"])
        .classes(card_class)
    ):
        render_image_thumbnail(url, filename)
        with ui.column().classes(
            UI_PRESENTATION["render_attachment_chip.classes.min_w_0_flex_1_gap_0"]
        ):
            ui.label(filename).classes(
                UI_PRESENTATION["image_destination.filename_classes"]
            ).tooltip(filename)
            if file_size is not None:
                _, _, extension = get_file_icon(filename)
                ui.label(
                    ui_text(
                        "chat.input_bar.attachment_size",
                        size=format_file_size(file_size),
                        extension=extension,
                    )
                ).classes(
                    UI_PRESENTATION[
                        "render_attachment_chip.classes.text_10px_text_zinc_400_leading_tight_font_mo"
                    ]
                )
            toggle = (
                ui.toggle(
                    {
                        route: ui_text(f"chat.input_bar.image_destination_{route}")
                        for route in ("llm", "image", "both")
                    },
                    value=destination,
                    on_change=(lambda event: on_change(event.value))
                    if on_change
                    else None,
                )
                .props(
                    UI_PRESENTATION["image_destination.props"].format(
                        label=ui_text("chat.input_bar.image_destination")
                    )
                )
                .classes(UI_PRESENTATION["image_destination.classes"])
            )
            if on_change is None:
                toggle.props(UI_PRESENTATION["image_destination.readonly_props"])
        if on_remove:
            label = ui_text("chat.message_editor.remove_attachment", name=filename)
            ui.button(
                icon=UI_PRESENTATION["render_attachment_chip.icon.close"],
                on_click=on_remove,
            ).props(
                UI_PRESENTATION["image_destination.remove_props"].format(label=label)
            ).tooltip(label)
