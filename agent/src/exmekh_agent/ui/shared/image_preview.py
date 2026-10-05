"""Clickable image thumbnails shared by the composer and saved messages."""

from nicegui import ui

from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/image_preview")


def show_image_preview(url: str, label: str) -> None:
    with ui.dialog().classes(
        UI_PRESENTATION["show_image_preview.classes.exmekh_image_dialog"]
    ) as dialog:
        with ui.card().classes(
            UI_PRESENTATION[
                "show_image_preview.classes.exmekh_image_preview_bg_zinc_950_text_zinc_10"
            ]
        ):
            with ui.row().classes(
                UI_PRESENTATION[
                    "show_image_preview.classes.w_full_items_center_justify_between_no_wrap"
                ]
            ):
                ui.label(label).classes(
                    UI_PRESENTATION["show_image_preview.classes.text_sm_truncate"]
                )
                ui.button(
                    icon=UI_PRESENTATION["show_image_preview.icon.close"],
                    on_click=dialog.close,
                ).props(
                    UI_PRESENTATION[
                        "show_image_preview.props.flat_round_dense_aria_label_close_image_previ"
                    ]
                ).tooltip(ui_text("shared.image_preview.close_image_preview"))
            ui.image(url).props(
                UI_PRESENTATION["show_image_preview.props.fit_contain_no_spinner"]
            ).classes(
                UI_PRESENTATION["show_image_preview.classes.exmekh_preview_image"]
            )
    dialog.on("hide", dialog.delete)
    dialog.open()


def render_image_thumbnail(url: str, label: str) -> None:
    ui.image(url).props(
        UI_PRESENTATION["render_image_thumbnail.props.fit_contain_no_spinner"]
    ).classes(
        UI_PRESENTATION[
            "render_image_thumbnail.classes.exmekh_image_thumbnail_rounded_border_border_"
        ]
    ).on("click", lambda: show_image_preview(url, label)).tooltip(label)
