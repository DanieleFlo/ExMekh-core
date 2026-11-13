"""Image previews with a shared gallery for generated batches."""

import base64
import json

from nicegui import ui

from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/image_preview")
FIT_ZOOM = 100
ZOOM_STEP = 25
MIN_ZOOM = 25
MAX_ZOOM = 400


def _enable_mouse_pan(viewport: ui.element) -> None:
    viewport.on(
        "pointerdown",
        js_handler="""event => {
            const viewport = event.currentTarget;
            if (event.pointerType !== 'mouse' || event.button !== 0
                || !event.target.closest('.exmekh-preview-image')
                || !viewport.classList.contains('image-preview-pannable')) return;
            event.preventDefault();
            viewport.imagePreviewDrag = {
                pointer: event.pointerId, x: event.clientX, y: event.clientY,
                left: viewport.scrollLeft, top: viewport.scrollTop,
            };
            viewport.setPointerCapture(event.pointerId);
            viewport.classList.add('image-preview-dragging');
        }""",
    )
    viewport.on(
        "pointermove",
        js_handler="""event => {
            const viewport = event.currentTarget;
            const drag = viewport.imagePreviewDrag;
            if (!drag || drag.pointer !== event.pointerId) return;
            viewport.scrollLeft = drag.left + drag.x - event.clientX;
            viewport.scrollTop = drag.top + drag.y - event.clientY;
        }""",
    )
    viewport.on(
        "lostpointercapture",
        js_handler="""event => {
            const viewport = event.currentTarget;
            delete viewport.imagePreviewDrag;
            viewport.classList.remove('image-preview-dragging');
        }""",
    )
    viewport.on("dragstart", js_handler="event => event.preventDefault()")


def show_image_preview(
    url: str, label: str, *, gallery: list[str] | None = None
) -> None:
    urls = list(gallery or [url])
    if url not in urls:
        urls.append(url)
    index = urls.index(url)
    client = ui.context.client
    zoom = FIT_ZOOM

    def set_zoom(value: int) -> None:
        nonlocal zoom
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, value))
        zoom_out.set_enabled(zoom > MIN_ZOOM)
        zoom_in.set_enabled(zoom < MAX_ZOOM)
        reset_zoom.set_text(ui_text("shared.image_preview.zoom_level", percent=zoom))
        reset_zoom.set_enabled(zoom != FIT_ZOOM)
        client.run_javascript(f"""
            (() => {{
                const viewport = getHtmlElement({viewport.id});
                if (!viewport) return;
                const x = (viewport.scrollLeft + viewport.clientWidth / 2) / viewport.scrollWidth;
                const y = (viewport.scrollTop + viewport.clientHeight / 2) / viewport.scrollHeight;
                viewport.style.setProperty('--image-preview-zoom', '{zoom}%');
                viewport.scrollLeft = x * viewport.scrollWidth - viewport.clientWidth / 2;
                viewport.scrollTop = y * viewport.scrollHeight - viewport.clientHeight / 2;
                viewport.classList.toggle(
                    'image-preview-pannable',
                    viewport.scrollWidth > viewport.clientWidth || viewport.scrollHeight > viewport.clientHeight,
                );
            }})()
        """)

    def move(delta: int) -> None:
        nonlocal index
        next_index = max(0, min(len(urls) - 1, index + delta))
        if next_index != index:
            index = next_index
            preview.set_source(urls[index])
            if zoom != FIT_ZOOM:
                set_zoom(FIT_ZOOM)
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
                    with ui.row().classes(UI_PRESENTATION["zoom.classes.controls"]):
                        out_label = ui_text("shared.image_preview.zoom_out")
                        reset_label = ui_text("shared.image_preview.reset_zoom")
                        in_label = ui_text("shared.image_preview.zoom_in")
                        zoom_out = (
                            ui.button(
                                icon=UI_PRESENTATION["zoom.icon.out"],
                                on_click=lambda: set_zoom(zoom - ZOOM_STEP),
                            )
                            .props(UI_PRESENTATION["gallery.props.button"])
                            .props("aria-label=" + json.dumps(out_label))
                            .tooltip(out_label)
                        )
                        reset_zoom = (
                            ui.button(
                                ui_text(
                                    "shared.image_preview.zoom_level", percent=FIT_ZOOM
                                ),
                                on_click=lambda: set_zoom(FIT_ZOOM),
                            )
                            .props(UI_PRESENTATION["zoom.props.reset"])
                            .props("aria-label=" + json.dumps(reset_label))
                            .classes(UI_PRESENTATION["zoom.classes.level"])
                            .tooltip(reset_label)
                            .disable()
                        )
                        zoom_in = (
                            ui.button(
                                icon=UI_PRESENTATION["zoom.icon.in"],
                                on_click=lambda: set_zoom(zoom + ZOOM_STEP),
                            )
                            .props(UI_PRESENTATION["gallery.props.button"])
                            .props("aria-label=" + json.dumps(in_label))
                            .tooltip(in_label)
                        )
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
                with ui.element().classes(
                    UI_PRESENTATION["zoom.classes.viewport"]
                ) as viewport:
                    _enable_mouse_pan(viewport)
                    with ui.element().classes(UI_PRESENTATION["zoom.classes.stage"]):
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
