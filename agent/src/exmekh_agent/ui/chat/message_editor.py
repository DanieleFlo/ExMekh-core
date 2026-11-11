"""Own the message editing dialog and the lifetime of its staged uploads."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import TypeAlias

from nicegui import ui
from nicegui.events import MultiUploadEventArguments

from exmekh_agent.exmekh_bridge.history.message_attachments import (
    retained_attachment_paths,
    saved_images,
)
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.exmekh_bridge.models.messages import ImageDestination
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.ui.chat.image_attachment import render_image_attachment
from exmekh_agent.ui.chat.message_text import message_display_text
from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/message_editor")

EditMessageCallback: TypeAlias = Callable[
    [str, str, str, list[str], list[StagedAttachment], dict[str, ImageDestination]],
    Awaitable[bool],
]


class MessageEditor:
    def __init__(self, on_save: EditMessageCallback) -> None:
        self._on_save = on_save
        self._added: list[StagedAttachment] = []
        self._saved = False
        self._uploading = False
        self._saving = False

    def open(self, session_id: str, message: ChatMessage, storage: TempStorage) -> None:
        self._session_id = session_id
        self._message_id = message.id
        self._paths = retained_attachment_paths(message)
        self._images = saved_images(message)
        self._image_destinations: dict[str, ImageDestination] = {}
        self._storage = storage
        self._client = ui.context.client
        with ui.dialog().classes(
            UI_PRESENTATION["open.classes.message_edit_dialog"]
        ) as self._dialog:
            with ui.card().classes(
                UI_PRESENTATION["open.classes.w_640px_max_w_94vw_bg_zinc_900_gap_4"]
            ):
                ui.label(ui_text("chat.message_editor.edit_message")).classes(
                    UI_PRESENTATION["open.classes.text_lg_text_zinc_100"]
                )
                self._editor = (
                    ui.textarea(
                        label=ui_text("chat.message_editor.message"),
                        value=message_display_text(message),
                    )
                    .props(UI_PRESENTATION["open.props.outlined_autogrow_autofocus"])
                    .classes(UI_PRESENTATION["open.classes.w_full"])
                )
                ui.label(
                    ui_text(
                        "chat.message_editor.saving_restarts_the_response_and_removes_the"
                    )
                ).classes(UI_PRESENTATION["open.classes.text_xs_text_zinc_400"])
                attachment_list = ui.column().classes(
                    UI_PRESENTATION["open.classes.w_full_gap_2"]
                )
                self._uploader = (
                    ui.upload(
                        label=ui_text("chat.message_editor.add_attachments"),
                        multiple=True,
                        auto_upload=True,
                        on_begin_upload=self._begin_upload,
                        on_multi_upload=self._upload,
                    )
                    .classes(UI_PRESENTATION["open.classes.message_edit_upload_w_full"])
                    .props(UI_PRESENTATION["open.props.flat_bordered_dark"])
                )
                with ui.row().classes(
                    UI_PRESENTATION["open.classes.w_full_justify_end_gap_2"]
                ):
                    self._cancel_button = ui.button(
                        ui_text("chat.message_editor.cancel"),
                        on_click=self._dialog.close,
                    ).props(UI_PRESENTATION["open.props.flat"])
                    self._save_button = ui.button(
                        ui_text("chat.message_editor.save_regenerate"),
                        on_click=self._save,
                    ).props(UI_PRESENTATION["open.props.unelevated"])
                self._editor.on_value_change(self._update_enabled)
                with attachment_list:
                    self._render_attachments()
        self._dialog.on("hide", self._discard_uploads)
        self._client.on_delete(self._discard_uploads)
        self._dialog.open()

    def _update_enabled(self) -> None:
        self._save_button.set_enabled(
            bool((self._editor.value or "").strip() or self._paths or self._added)
            and not self._uploading
            and not self._saving
        )

    def _remove_path(self, path: str) -> None:
        if self._saving:
            return
        self._paths.remove(path)
        self._image_destinations.pop(path, None)
        self._refresh_attachments()

    def _change_destination(self, path: str, destination: ImageDestination) -> None:
        if not self._saving:
            self._image_destinations[path] = destination

    def _remove_added(self, attachment: StagedAttachment) -> None:
        if self._saving:
            return
        self._added.remove(attachment)
        attachment.delete_file()
        self._refresh_attachments()

    @ui.refreshable
    def _render_attachments(self) -> None:
        with ui.row().classes(
            UI_PRESENTATION[
                "render_attachments.classes.w_full_flex_wrap_gap_2_message_edit_attachmen"
            ]
        ):
            images_by_path = {image.path: image for image in self._images if image.path}
            for path in self._paths:
                image = images_by_path.get(path)
                remove = partial(self._remove_path, path)
                if image is not None and image.destination is not None:
                    render_image_attachment(
                        image.url,
                        image.filename or Path(path).name,
                        image.file_size,
                        self._image_destinations.get(path, image.destination),
                        card_class=UI_PRESENTATION[
                            "image_destination.editor_card_classes"
                        ],
                        on_change=partial(self._change_destination, path),
                        on_remove=remove,
                    )
                else:
                    self._render_attachment(Path(path).name, remove)
            for index, image in enumerate(self._images):
                if image.path is None and image.destination is not None:
                    render_image_attachment(
                        image.url,
                        ui_text(
                            "chat.chat_view.attached_image_value_1", value_1=index + 1
                        ),
                        image.file_size,
                        image.destination,
                        card_class=UI_PRESENTATION[
                            "image_destination.editor_card_classes"
                        ],
                    )
            for attachment in self._added:
                remove = partial(self._remove_added, attachment)
                if attachment.is_raster_image:
                    render_image_attachment(
                        attachment.image_urls[0],
                        attachment.filename,
                        attachment.file_size,
                        attachment.image_destination,
                        card_class=UI_PRESENTATION[
                            "image_destination.editor_card_classes"
                        ],
                        on_change=partial(self._change_added_destination, attachment),
                        on_remove=remove,
                    )
                else:
                    self._render_attachment(attachment.filename, remove)
        self._update_enabled()

    def _change_added_destination(
        self, attachment: StagedAttachment, destination: ImageDestination
    ) -> None:
        if not self._saving:
            attachment.select_image_destination(destination)

    def _render_attachment(self, name: str, on_remove: Callable[[], None]) -> None:
        with ui.row().classes(
            UI_PRESENTATION[
                "render_attachment.classes.items_center_gap_1_rounded_bg_zinc_800_px_2_m"
            ]
        ):
            ui.label(name).classes(
                UI_PRESENTATION["render_attachment.classes.text_xs_truncate_max_w_60"]
            )
            ui.button(
                icon=UI_PRESENTATION["render_attachment.icon.close"], on_click=on_remove
            ).props(
                UI_PRESENTATION["render_attachment.props_template"].format(name=name)
            )

    def _refresh_attachments(self) -> None:
        if not self._client.is_deleted:
            with self._client:
                self._render_attachments.refresh()

    def _begin_upload(self) -> None:
        self._uploading = True
        self._update_enabled()

    async def _upload(self, event: MultiUploadEventArguments) -> None:
        try:
            for file in event.files:
                data = await file.read()
                attachment = await asyncio.get_running_loop().run_in_executor(
                    None,
                    partial(
                        create_staged_attachment_from_file,
                        file.name,
                        data,
                        storage=self._storage,
                    ),
                )
                if not self._dialog.value or self._client.is_deleted:
                    attachment.delete_file()
                else:
                    self._added.append(attachment)
        except (OSError, ValueError) as error:
            if not self._client.is_deleted:
                with self._client:
                    ui.notify(
                        ui_text(
                            "chat.message_editor.unable_to_add_attachment", error=error
                        ),
                        type="negative",
                    )
        finally:
            self._uploading = False
            if not self._client.is_deleted:
                with self._client:
                    self._uploader.reset()
                    self._refresh_attachments()

    def _discard_uploads(self) -> None:
        if self._saved or self._saving:
            return
        for attachment in self._added:
            attachment.delete_file()
        self._added.clear()

    def _set_saving(self, saving: bool) -> None:
        self._saving = saving
        self._editor.set_enabled(not saving)
        self._uploader.set_enabled(not saving)
        self._cancel_button.set_enabled(not saving)
        if saving:
            self._dialog.props(UI_PRESENTATION["set_saving.props.persistent"])
        else:
            self._dialog.props(remove="persistent")
        self._update_enabled()

    async def _save(self) -> None:
        if self._saving or self._uploading:
            return
        self._set_saving(True)
        try:
            self._saved = await self._on_save(
                self._session_id,
                self._message_id,
                self._editor.value or "",
                list(self._paths),
                list(self._added),
                dict(self._image_destinations),
            )
            if (
                self._saved
                and not self._client.is_deleted
                and not self._dialog.is_deleted
            ):
                with self._client:
                    self._dialog.close()
        finally:
            self._saving = False
            if (
                self._client.is_deleted
                or self._dialog.is_deleted
                or not self._dialog.value
            ):
                self._discard_uploads()
            if not self._client.is_deleted and not self._dialog.is_deleted:
                with self._client:
                    self._set_saving(False)
