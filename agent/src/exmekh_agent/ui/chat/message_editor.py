"""Own the message editing dialog and the lifetime of its staged uploads."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import TypeAlias

from nicegui import ui
from nicegui.events import MultiUploadEventArguments

from exmekh_agent.exmekh_bridge.history.message_edit import retained_attachment_paths
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.ui.chat.message_text import message_display_text
from exmekh_agent.ui.resources import ui_text

EditMessageCallback: TypeAlias = Callable[
    [str, str, str, list[str], list[StagedAttachment]], Awaitable[bool]
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
        self._storage = storage
        self._client = ui.context.client
        with ui.dialog().classes("message-edit-dialog") as self._dialog:
            with ui.card().classes("w-[640px] max-w-[94vw] bg-zinc-900 gap-4"):
                ui.label(ui_text("chat.message_editor.edit_message")).classes(
                    "text-lg text-zinc-100"
                )
                self._editor = (
                    ui.textarea(
                        label=ui_text("chat.message_editor.message"),
                        value=message_display_text(message),
                    )
                    .props("outlined autogrow autofocus")
                    .classes("w-full")
                )
                ui.label(
                    ui_text(
                        "chat.message_editor.saving_restarts_the_response_and_removes_the"
                    )
                ).classes("text-xs text-zinc-400")
                attachment_list = ui.column().classes("w-full gap-2")
                self._uploader = (
                    ui.upload(
                        label=ui_text("chat.message_editor.add_attachments"),
                        multiple=True,
                        auto_upload=True,
                        on_begin_upload=self._begin_upload,
                        on_multi_upload=self._upload,
                    )
                    .classes("message-edit-upload w-full")
                    .props("flat bordered dark")
                )
                with ui.row().classes("w-full justify-end gap-2"):
                    self._cancel_button = ui.button(
                        ui_text("chat.message_editor.cancel"),
                        on_click=self._dialog.close,
                    ).props("flat")
                    self._save_button = ui.button(
                        ui_text("chat.message_editor.save_regenerate"),
                        on_click=self._save,
                    ).props("unelevated")
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
        self._refresh_attachments()

    def _remove_added(self, attachment: StagedAttachment) -> None:
        if self._saving:
            return
        self._added.remove(attachment)
        attachment.delete_file()
        self._refresh_attachments()

    @ui.refreshable
    def _render_attachments(self) -> None:
        with ui.row().classes("w-full flex-wrap gap-2 message-edit-attachments"):
            for path in self._paths:
                self._render_attachment(
                    Path(path).name, partial(self._remove_path, path)
                )
            for attachment in self._added:
                self._render_attachment(
                    attachment.filename, partial(self._remove_added, attachment)
                )
        self._update_enabled()

    def _render_attachment(self, name: str, on_remove: Callable[[], None]) -> None:
        with ui.row().classes("items-center gap-1 rounded bg-zinc-800 px-2 min-w-0"):
            ui.label(name).classes("text-xs truncate max-w-60")
            ui.button(icon="close", on_click=on_remove).props(
                f'flat round dense aria-label="Remove {name}"'
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
            self._dialog.props("persistent")
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
            )
            if self._saved and not self._client.is_deleted:
                with self._client:
                    self._dialog.close()
        finally:
            self._saving = False
            if self._client.is_deleted or not self._dialog.value:
                self._discard_uploads()
            if not self._client.is_deleted:
                with self._client:
                    self._set_saving(False)
