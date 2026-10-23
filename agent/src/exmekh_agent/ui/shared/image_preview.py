"""Image previews with a shared gallery for generated batches."""

import base64
import json

from nicegui import ui

from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/image_preview")


def show_image_preview(
    url: str, label: str, *, gallery: list[str] | None = None
) -> None:
    urls = list(gallery or [url])
    if url not in urls:
        urls.append(url)
    index = urls.index(url)
    client = ui.context.client

    def move(delta: int) -> None:
        nonlocal index
        index = max(0, min(len(urls) - 1, index + delta))
        preview.set_source(urls[index])
        title.set_text(
            ui_text(
                "shared.image_preview.image_position", index=index + 1, count=len(urls)
            )
            if len(urls) > 1
            else label
        )
        previous.set_visibility(index > 0)
        following.set_visibility(index < len(urls) - 1)

    def download() -> None:
        current = urls[index]
        if current.startswith("data:image/png;base64,"):
            ui.download.content(
                base64.b64decode(current.split(",", 1)[1]),
                f"image-{index + 1}.png",
                "image/png",
            )
        else:
            ui.download(current)

    async def copy_image() -> None:
        with client:
            copied = await ui.run_javascript(
                "return (async () => { try { const response = await fetch("
                + json.dumps(urls[index])
                + "); const blob = await response.blob(); await navigator.clipboard.write([new ClipboardItem({[blob.type]: blob})]); return true; } catch { return false; } })()"
            )
            if not copied and not client.is_deleted:
                ui.notify(ui_text("shared.image_preview.copy_failed"), type="negative")

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
                title = ui.label(label).classes(
                    UI_PRESENTATION["show_image_preview.classes.text_sm_truncate"]
                )
                with ui.row().classes(UI_PRESENTATION["gallery.classes.actions"]):
                    for action, icon, callback in (
                        ("download", "download", download),
                        ("copy", "content_copy", copy_image),
                        ("close_image_preview", "close", dialog.close),
                    ):
                        text = ui_text(f"shared.image_preview.{action}")
                        ui.button(icon=icon, on_click=callback).props(
                            UI_PRESENTATION["gallery.props.button"]
                        ).props("aria-label=" + json.dumps(text)).tooltip(text)
            with ui.row().classes(UI_PRESENTATION["gallery.classes.body"]):
                previous = (
                    ui.button(icon="chevron_left", on_click=lambda: move(-1))
                    .props(UI_PRESENTATION["gallery.props.button"])
                    .props(
                        "aria-label="
                        + json.dumps(ui_text("shared.image_preview.previous"))
                    )
                )
                preview = (
                    ui.image(url)
                    .props(
                        UI_PRESENTATION[
                            "show_image_preview.props.fit_contain_no_spinner"
                        ]
                    )
                    .classes(
                        UI_PRESENTATION[
                            "show_image_preview.classes.exmekh_preview_image"
                        ]
                    )
                )
                following = (
                    ui.button(icon="chevron_right", on_click=lambda: move(1))
                    .props(UI_PRESENTATION["gallery.props.button"])
                    .props(
                        "aria-label=" + json.dumps(ui_text("shared.image_preview.next"))
                    )
                )
            previous.set_visibility(index > 0)
            following.set_visibility(index < len(urls) - 1)
            move(0)
    dialog.on("keydown.left", lambda: move(-1))
    dialog.on("keydown.right", lambda: move(1))
    dialog.on("hide", dialog.delete)
    dialog.open()


def render_image_thumbnail(
    url: str, label: str, *, gallery: list[str] | None = None
) -> None:
    ui.image(url).props(
        UI_PRESENTATION["render_image_thumbnail.props.fit_contain_no_spinner"]
    ).classes(
        UI_PRESENTATION[
            "render_image_thumbnail.classes.exmekh_image_thumbnail_rounded_border_border_"
        ]
    ).on("click", lambda: show_image_preview(url, label, gallery=gallery)).tooltip(
        label
    )
