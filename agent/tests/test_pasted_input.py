from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from nicegui.client import Client
from nicegui.page import page
from starlette.testclient import TestClient

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.exmekh_bridge.pasted_input import (
    StagedAttachment,
    create_input_markdown_file,
    create_staged_attachment,
    create_staged_attachment_from_file,
    create_staged_attachments_from_text,
    extract_attached_files_from_content,
    extract_content_preview,
    format_user_input_file_xml,
    is_large_input,
    prepare_user_input,
    process_user_input_text,
    resolve_attached_file_path,
    strip_user_input_file_xml,
)
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.state.temp_storage import (
    TEMP_STORAGE_ALERT_THRESHOLD_BYTES,
    TempFileInfo,
    TempStorage,
    TempStorageStats,
    sanitize_attachment_filename,
)
from exmekh_agent.ui.chat_view import ChatViewComponent
from exmekh_agent.ui.file_icons import format_file_size, get_file_icon
from exmekh_agent.ui.file_upload import upload_router
from exmekh_agent.ui.input_bar import InputBarComponent
from exmekh_agent.ui.page_controller import ChatPageController
from exmekh_agent.ui.settings_drawer import SettingsDrawerComponent
from tests.image_samples import png_bytes


@pytest.fixture
def ui_context(tmp_path: Path):
    session = session_manager.create_session("Attachment UI test")
    session.settings.work_root = str(tmp_path)
    c = Client(page("/test-pasted-input"))
    with c:
        yield c


def test_is_large_input() -> None:
    assert is_large_input("") is False
    assert is_large_input("Short text") is False
    long_text = "word " * 400
    assert is_large_input(long_text) is True


def test_create_input_markdown_file(tmp_path: Path) -> None:
    # Explicit work_root
    file_path, rel_path = create_input_markdown_file(
        "test content", work_root=tmp_path, temp_dirname="test_tmp"
    )
    assert file_path.exists()
    assert file_path.suffix == ".md"
    assert file_path.read_text(encoding="utf-8") == "test content"
    assert rel_path.startswith("test_tmp/")

    # Default work_root (uses config.work_root)
    old_root = config.work_root
    try:
        config.work_root = str(tmp_path)
        fp2, rel2 = create_input_markdown_file("default root content")
        assert fp2.exists()
        assert rel2.startswith("tmp/")
    finally:
        config.work_root = old_root


def test_format_user_input_file_xml() -> None:
    xml = format_user_input_file_xml("tmp/input_123.md")
    assert '<user_input_file path="tmp/input_123.md">' in xml
    assert "This file was attached from user input." in xml
    assert "Path: tmp/input_123.md" in xml
    assert "</user_input_file>" in xml


def test_process_user_input_text_empty_and_short(tmp_path: Path) -> None:
    # Empty
    prompt, display, fpaths = process_user_input_text("", work_root=tmp_path)
    assert prompt == ""
    assert display is None
    assert fpaths == []

    # Short
    prompt, display, fpaths = process_user_input_text("Hello there", work_root=tmp_path)
    assert prompt == "Hello there"
    assert display is None
    assert fpaths == []


def test_process_user_input_text_single_and_multiple_code_blocks(
    tmp_path: Path,
) -> None:
    # Single code block
    large_code_1 = "```python\n" + ("x = 1\n" * 350) + "```"
    text_1 = f"Analyze this:\n{large_code_1}\nThanks!"
    prompt_1, display_1, fpaths_1 = process_user_input_text(text_1, work_root=tmp_path)
    assert prompt_1.startswith("Analyze this:\n<user_input_file path=")
    assert prompt_1.endswith("</user_input_file>\nThanks!")
    assert display_1 == text_1
    assert len(fpaths_1) == 1
    assert fpaths_1[0].exists()
    assert fpaths_1[0].read_text(encoding="utf-8") == large_code_1

    # Multiple code blocks in the same message
    large_code_2 = "```javascript\n" + ("console.log('hello');\n" * 350) + "```"
    text_multi = f"First:\n{large_code_1}\nMiddle:\n{large_code_2}\nDone!"
    prompt_multi, display_multi, fpaths_multi = process_user_input_text(
        text_multi, work_root=tmp_path
    )
    assert len(fpaths_multi) == 2
    assert prompt_multi.count("<user_input_file path=") == 2
    assert prompt_multi.startswith("First:\n<user_input_file path=")
    assert "Middle:\n<user_input_file path=" in prompt_multi
    assert prompt_multi.endswith("</user_input_file>\nDone!")
    assert display_multi == text_multi
    assert fpaths_multi[0].exists()
    assert fpaths_multi[1].exists()
    assert fpaths_multi[0].read_text(encoding="utf-8") == large_code_1
    assert fpaths_multi[1].read_text(encoding="utf-8") == large_code_2


def test_process_user_input_text_multiple_paragraphs(tmp_path: Path) -> None:
    short_header = "Compare these two logs:"
    long_body_1 = "Error stack trace alpha " * 150
    middle = "Between logs"
    long_body_2 = "Error stack trace beta " * 150
    text = f"{short_header}\n\n{long_body_1}\n\n{middle}\n\n{long_body_2}"

    prompt, display, fpaths = process_user_input_text(text, work_root=tmp_path)
    assert len(fpaths) == 2
    assert prompt.count("<user_input_file path=") == 2
    assert f"{short_header}\n\n<user_input_file path=" in prompt
    assert f"\n\n{middle}\n\n<user_input_file path=" in prompt
    assert display == text
    assert fpaths[0].read_text(encoding="utf-8") == long_body_1
    assert fpaths[1].read_text(encoding="utf-8") == long_body_2


def test_process_user_input_text_full_raw(tmp_path: Path) -> None:
    long_raw = "Single giant paragraph of text with many tokens. " * 80
    prompt, display, fpaths = process_user_input_text(long_raw, work_root=tmp_path)
    assert prompt.startswith("<user_input_file path=")
    assert prompt.endswith("</user_input_file>")
    assert display == long_raw
    assert len(fpaths) == 1
    assert fpaths[0].exists()
    assert fpaths[0].read_text(encoding="utf-8") == long_raw


def test_process_user_input_text_mixed_code_blocks(tmp_path: Path) -> None:
    short_code = "```python\nx = 1\n```"
    large_code = "```python\n" + ("x = 1\n" * 350) + "```"
    text = f"Intro:\n{short_code}\nThen:\n{large_code}\nDone"
    prompt, display, fpaths = process_user_input_text(text, work_root=tmp_path)
    assert len(fpaths) == 1
    assert short_code in prompt
    assert "<user_input_file" in prompt
    assert display == text


def test_process_user_input_text_paragraphs_no_single_large(tmp_path: Path) -> None:
    p1 = "word " * 160
    p2 = "word " * 160
    text = f"{p1}\n\n{p2}"
    assert is_large_input(p1) is False
    assert is_large_input(p2) is False
    assert is_large_input(text) is True
    prompt, display, fpaths = process_user_input_text(text, work_root=tmp_path)
    assert len(fpaths) == 1
    assert prompt.startswith("<user_input_file path=")
    assert display == text


def test_chat_message_display_content() -> None:
    msg1 = ChatMessage(role="user", content="plain")
    assert msg1.display_content is None

    msg2 = ChatMessage(role="user", content="<xml>", display_content="original text")
    assert msg2.content == "<xml>"
    assert msg2.display_content == "original text"

    dump = msg2.model_dump()
    assert dump["display_content"] == "original text"
    restored = ChatMessage.model_validate(dump)
    assert restored.display_content == "original text"


def test_extract_content_preview() -> None:
    # 1. Code fence with inner code
    assert (
        extract_content_preview("```python\ndef compute_loss():\n    pass")
        == "def compute_loss():"
    )
    # 2. Only fence line
    assert extract_content_preview("```python\n```") == "```python"
    # 3. Blank / empty text
    assert extract_content_preview("   \n\n  ") == "pasted_text"
    # 4. Long line > max_chars
    long_line = "a" * 35
    assert extract_content_preview(long_line, max_chars=25) == ("a" * 25) + "..."
    # 5. Short line <= max_chars
    assert extract_content_preview("short line", max_chars=25) == "short line"


def test_create_staged_attachment(tmp_path: Path) -> None:
    content = "Hello staged world\n" + ("x = 1\n" * 200)
    att = create_staged_attachment(content, work_root=tmp_path)
    assert att.content == content
    assert att.preview == "Hello staged world"
    assert att.token_count > 0
    assert att.file_path is not None
    assert att.file_path.exists()
    assert att.relative_path is not None
    assert att.relative_path.startswith("tmp/user_input_")
    assert att.filename.endswith(".md")


def test_staged_attachment_reference_and_cleanup(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    attachment = create_staged_attachment("Source text", work_root=first_root)
    original_path = attachment.file_path
    assert original_path is not None

    first_reference = attachment.reference(first_root)
    assert attachment.file_path == original_path
    assert attachment.relative_path in first_reference

    second_reference = attachment.reference(second_root)
    assert attachment.file_path is not None
    assert attachment.file_path != original_path
    assert attachment.file_path.read_text(encoding="utf-8") == "Source text"
    assert attachment.relative_path in second_reference
    assert original_path.exists()

    second_path = attachment.file_path
    attachment.delete_file()
    assert attachment.file_path is None
    assert attachment.relative_path is None
    assert not second_path.exists()
    attachment.delete_file()
    attachment.reference(second_root)
    assert attachment.file_path is not None and attachment.file_path.exists()


def test_prepare_user_input_and_attachment_paths(tmp_path: Path) -> None:
    plain = prepare_user_input("Hello", work_root=tmp_path)
    assert plain.prompt == "Hello"
    assert plain.display_content is None
    assert plain.title_text == "Hello"

    long_text = "many words " * 350
    converted = prepare_user_input(long_text, work_root=tmp_path)
    assert "<user_input_file" in converted.prompt
    assert converted.display_content == long_text
    assert converted.title_text == long_text

    attachment = create_staged_attachment("Attached body", work_root=tmp_path)
    prepared = prepare_user_input(
        "Please inspect", work_root=tmp_path, attachments=[attachment]
    )
    assert prepared.prompt.startswith("Please inspect\n\n<user_input_file")
    assert prepared.display_content == "Please inspect"
    assert prepared.title_text == "Please inspect"
    attachment_only = prepare_user_input(
        "", work_root=tmp_path, attachments=[attachment]
    )
    assert attachment_only.title_text == attachment.preview
    assert attachment_only.display_content == ""
    empty = prepare_user_input("", work_root=tmp_path, attachments=[])
    assert empty.title_text == "Attached file"
    assert empty.display_content is None

    staged_path = resolve_attached_file_path(attachment.relative_path or "", tmp_path)
    assert staged_path == attachment.file_path
    with pytest.raises(ValueError):
        resolve_attached_file_path("../outside.md", tmp_path)


def test_create_staged_attachments_from_text(tmp_path: Path) -> None:
    # Empty
    assert create_staged_attachments_from_text("", work_root=tmp_path) == []
    # Short
    assert create_staged_attachments_from_text("Short text", work_root=tmp_path) == []

    # Single large text
    long_raw = "word " * 350
    single_list = create_staged_attachments_from_text(long_raw, work_root=tmp_path)
    assert len(single_list) == 1
    assert single_list[0].file_path is not None
    assert single_list[0].file_path.exists()

    # Multiple code blocks > 300 tokens
    b1 = "```python\n" + ("a = 1\n" * 350) + "```"
    b2 = "```javascript\n" + ("b = 2\n" * 350) + "```"
    multi_text = f"Intro\n{b1}\nMiddle\n{b2}\nEnd"
    multi_list = create_staged_attachments_from_text(multi_text, work_root=tmp_path)
    assert len(multi_list) == 2
    assert multi_list[0].content == b1
    assert multi_list[1].content == b2


def test_input_bar_get_temp_storage(tmp_path: Path) -> None:
    ib1 = InputBarComponent(
        on_send=lambda _: None,
        on_stop=lambda: None,
        temp_storage_getter=lambda _session_id: TempStorage(tmp_path),
    )
    assert ib1.get_temp_storage().work_root == tmp_path

    ib2 = InputBarComponent(
        on_send=lambda _: None,
        on_stop=lambda: None,
        temp_storage_getter=lambda _session_id: TempStorage(),
    )
    assert ib2.get_temp_storage().work_root == Path(config.work_root)

    ib3 = InputBarComponent(
        on_send=lambda _: None,
        on_stop=lambda: None,
    )
    assert ib3.get_temp_storage().work_root == Path(config.work_root)


def test_input_bar_attachments_management(ui_context: Client, tmp_path: Path) -> None:
    with ui_context:
        ib = InputBarComponent(
            on_send=lambda *args: None,
            on_stop=lambda: None,
            temp_storage_getter=lambda _session_id: TempStorage(tmp_path),
        )
        ib.build()
        assert ib.attachments_container is not None
        assert ib.attachments_container.visible is False

        # Render with empty attachments -> stays hidden
        ib.render_attachments()
        assert ib.attachments_container.visible is False

        # Add attachment
        att1 = create_staged_attachment("long content " * 100, work_root=tmp_path)
        ib.staged_attachments.append(att1)
        ib.render_attachments()
        assert ib.attachments_container.visible is True

        # Insert into message
        assert att1.file_path is not None
        att1_path = att1.file_path
        ib._insert_attachment_into_message(att1)
        assert len(ib.staged_attachments) == 0
        assert ib.textarea is not None
        assert ib.textarea.value == att1.content
        assert not att1_path.exists()
        assert ib.attachments_container.visible is False

        # Insert another when textarea already has content
        att2 = create_staged_attachment("second snippet " * 100, work_root=tmp_path)
        ib.staged_attachments.append(att2)
        ib.render_attachments()
        ib._insert_attachment_into_message(att2)
        assert ib.textarea.value == f"{att1.content.strip()}\n\n{att2.content}"

        # Delete attachment
        att3 = create_staged_attachment("third snippet " * 100, work_root=tmp_path)
        assert att3.file_path is not None
        att3_path = att3.file_path
        ib.staged_attachments.append(att3)
        ib.render_attachments()
        assert ib.attachments_container.visible is True
        ib._delete_attachment(att3)
        assert len(ib.staged_attachments) == 0
        assert not att3_path.exists()
        assert ib.attachments_container.visible is False

        # Delete attachment where file does not exist
        att_missing = StagedAttachment(
            id="missing",
            content="abc",
            preview="abc",
            token_count=10,
            file_path=tmp_path / "does_not_exist.md",
        )
        ib.staged_attachments.append(att_missing)
        ib._delete_attachment(att_missing)

        # Insert attachment where unlinking raises OSError
        att_err = StagedAttachment(
            id="err",
            content="abc",
            preview="abc",
            token_count=10,
            file_path=tmp_path / "err.md",
        )
        (tmp_path / "err.md").write_text("dummy", encoding="utf-8")
        ib.staged_attachments.append(att_err)
        with patch.object(Path, "unlink", side_effect=OSError("locked")):
            ib._insert_attachment_into_message(att_err)


async def test_input_bar_paste_handling(ui_context: Client, tmp_path: Path) -> None:
    with ui_context:
        ib = InputBarComponent(
            on_send=lambda *args: None,
            on_stop=lambda: None,
            temp_storage_getter=lambda _session_id: TempStorage(tmp_path),
        )
        ib.build()
        assert ib.textarea is not None

        # 1. Empty / non-text event
        mock_empty = MagicMock()
        mock_empty.args = {}
        await ib._handle_paste(mock_empty)
        assert len(ib.staged_attachments) == 0

        # 2. String arg (short text)
        mock_short_str = MagicMock()
        mock_short_str.args = "short text"
        await ib._handle_paste(mock_short_str)
        assert len(ib.staged_attachments) == 0

        # 3. Prevented short text -> sets into textarea
        ib.textarea.set_value("")
        mock_prev_short = MagicMock()
        mock_prev_short.args = {"text": "pasted short", "prevented": True}
        await ib._handle_paste(mock_prev_short)
        assert ib.textarea.value == "pasted short"

        # 4. Large text with prevented=False (already entered textarea)
        long_code = "def large_fn():\n" + ("    x = 1\n" * 350)
        ib.textarea.set_value(f"prefix {long_code}")
        mock_large_unprev = MagicMock()
        mock_large_unprev.args = {"text": long_code, "prevented": False}
        await ib._handle_paste(mock_large_unprev)
        assert ib.textarea.value == "prefix "
        assert len(ib.staged_attachments) == 1
        assert ib.staged_attachments[0].content == long_code

        # 5. Large text with prevented=False but not in textarea
        ib.textarea.set_value("unrelated")
        long_code2 = "def second_fn():\n" + ("    y = 2\n" * 350)
        mock_large_unprev2 = MagicMock()
        mock_large_unprev2.args = {"text": long_code2, "prevented": False}
        await ib._handle_paste(mock_large_unprev2)
        assert ib.textarea.value == "unrelated"
        assert len(ib.staged_attachments) == 2

        # 6. Large text with prevented=True (browser didn't insert)
        long_code3 = "def third_fn():\n" + ("    z = 3\n" * 350)
        mock_large_prev = MagicMock()
        mock_large_prev.args = {"text": long_code3, "prevented": True}
        await ib._handle_paste(mock_large_prev)
        assert len(ib.staged_attachments) == 3


def test_input_bar_trigger_send_with_attachments(
    ui_context: Client, tmp_path: Path
) -> None:
    with ui_context:
        sent_calls: list[tuple[str, list[StagedAttachment]]] = []

        def on_send(text: str, attachments: list[StagedAttachment]) -> None:
            sent_calls.append((text, attachments))

        ib = InputBarComponent(
            on_send=lambda _: None,
            on_send_with_attachments=on_send,
            on_stop=lambda: None,
            temp_storage_getter=lambda _session_id: TempStorage(tmp_path),
        )
        ib.build()
        assert ib.textarea is not None
        assert ib.attachments_container is not None

        # Empty text, no attachments -> returns early
        ib._trigger_send()
        assert len(sent_calls) == 0

        # Attachments only (no text in textarea)
        att = create_staged_attachment("content " * 100, work_root=tmp_path)
        ib.staged_attachments.append(att)
        ib.render_attachments()
        ib._trigger_send()
        assert len(sent_calls) == 1
        assert sent_calls[0] == ("", [att])
        assert len(ib.staged_attachments) == 0
        assert ib.attachments_container.visible is False

        # Text and attachments
        att2 = create_staged_attachment("content2 " * 100, work_root=tmp_path)
        ib.staged_attachments.append(att2)
        ib.textarea.set_value("user message")
        ib._trigger_send()
        assert len(sent_calls) == 2
        assert sent_calls[1] == ("user message", [att2])
        assert ib.textarea.value == ""

        # Legacy 1-argument on_send fallback
        sent_legacy: list[str] = []
        ib_legacy = InputBarComponent(
            on_send=lambda text: sent_legacy.append(text),
            on_stop=lambda: None,
        )
        ib_legacy.build()
        assert ib_legacy.textarea is not None
        ib_legacy.textarea.set_value("legacy message")
        ib_legacy._trigger_send()
        assert sent_legacy == ["legacy message"]


def test_extract_attached_files_from_content() -> None:
    assert extract_attached_files_from_content("") == []
    assert extract_attached_files_from_content("plain text without tags") == []
    xml1 = '<user_input_file path="tmp/f1.md">\ncontent\n</user_input_file>'
    assert extract_attached_files_from_content(xml1) == ["tmp/f1.md"]
    xml2 = (
        f"{xml1}\n\n"
        + '<user_input_file path="tmp/f2.md">\ncontent\n</user_input_file>'
    )
    assert extract_attached_files_from_content(xml2) == ["tmp/f1.md", "tmp/f2.md"]


def test_strip_user_input_file_xml() -> None:
    assert strip_user_input_file_xml("") == ""
    assert strip_user_input_file_xml("just text") == "just text"
    xml = (
        'Before text\n\n<user_input_file path="tmp/f1.md">\ncontent\n'
        "</user_input_file>\n\nAfter text"
    )
    assert strip_user_input_file_xml(xml) == "Before text\n\nAfter text"


@pytest.mark.asyncio
async def test_chat_view_render_display_content(
    ui_context: Client, tmp_path: Path
) -> None:
    with ui_context:
        session = session_manager.get_current_session()
        session.settings.work_root = str(tmp_path)
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)

        # 1. Message with display_content and attached file XML
        msg_with_display = ChatMessage(
            role="user",
            content='My prompt\n\n<user_input_file path="tmp/input.md">\nhidden\n</user_input_file>',
            display_content="Visible user pasted text",
        )
        cv._render_message(msg_with_display)

        # 2. Message without display_content (fallback to strip_user_input_file_xml)
        msg_no_display = ChatMessage(
            role="user",
            content='Only this text should show\n\n<user_input_file path="tmp/input2.md">\nxml\n</user_input_file>',
        )
        cv._render_message(msg_no_display)

        # 3. Message plain without attachments
        msg_plain = ChatMessage(role="user", content="Visible plain text")
        cv._render_message(msg_plain)

        # 4. Message completely empty
        msg_empty = ChatMessage(role="user", content="", display_content="")
        cv._render_message(msg_empty)

        # 5. Modal dialog: existing file
        f_dir = tmp_path / "tmp"
        f_dir.mkdir(parents=True, exist_ok=True)
        sample_file = f_dir / "sample.md"
        sample_file.write_text("Hello file content", encoding="utf-8")
        await cv._show_attached_file_modal("tmp/sample.md")

        # 6. Modal dialog: non-existent file
        await cv._show_attached_file_modal("tmp/non_existent.md")

        # 7. Modal dialog: unreadable file (OSError)
        with patch.object(Path, "read_bytes", side_effect=OSError("denied")):
            await cv._show_attached_file_modal("tmp/sample.md")

        await cv._show_attached_file_modal("../outside.md")


@pytest.mark.asyncio
async def test_page_controller_send_long_input(
    ui_context: Client, tmp_path: Path
) -> None:
    with ui_context:
        ctrl = ChatPageController(ui_context)
        project = session_manager.create_project(tmp_path)
        session = session_manager.create_session(project_id=project.id)

        long_text = "Important text with code and details. " * 80
        assert is_large_input(long_text) is True

        try:
            with patch.object(
                ctrl, "_run_turn", new_callable=AsyncMock
            ) as mock_run_turn:
                await ctrl.send(long_text)

                assert len(session.messages) >= 1
                last_msg = session.messages[-1]
                assert last_msg.role == "user"
                assert "<user_input_file" in last_msg.content
                assert last_msg.display_content == long_text

                mock_run_turn.assert_called_once()
                called_kwargs = mock_run_turn.call_args.kwargs
                assert "<user_input_file" in called_kwargs["user_text"]
                assert called_kwargs["title_text"] == long_text
        finally:
            active_turns.cancel_and_forget(session.id)


@pytest.mark.asyncio
async def test_page_controller_send_with_attachments(
    ui_context: Client, tmp_path: Path
) -> None:
    with ui_context:
        ctrl = ChatPageController(ui_context)
        project = session_manager.create_project(tmp_path)
        session = session_manager.create_session(project_id=project.id)

        # 1. Verify _get_current_work_root
        assert ctrl._get_temp_storage().work_root == tmp_path

        # 2. Send with existing attachment file
        att = create_staged_attachment("attachment content " * 100, work_root=tmp_path)
        try:
            with patch.object(ctrl, "_run_turn", new_callable=AsyncMock):
                await ctrl.send("Here is my code:", attachments=[att])
                assert len(session.messages) >= 1
                last_msg = session.messages[-1]
                assert "<user_input_file path=" in last_msg.content
                assert "Here is my code:" in last_msg.content
                assert last_msg.display_content == "Here is my code:"
        finally:
            active_turns.cancel_and_forget(session.id)
            ctrl._active_task = None

        # 3. Send with attachment where file does not exist on disk
        att_nofile = StagedAttachment(
            id="nofile",
            content="content to write " * 100,
            preview="content to write",
            token_count=350,
        )
        try:
            with patch.object(ctrl, "_run_turn", new_callable=AsyncMock):
                await ctrl.send("", attachments=[att_nofile])
                last_msg = session.messages[-1]
                assert "<user_input_file path=" in last_msg.content
                assert last_msg.display_content == ""
        finally:
            active_turns.cancel_and_forget(session.id)
            ctrl._active_task = None

        # 4. Send with empty attachments list (user inlined attachment into text)
        try:
            with patch.object(ctrl, "_run_turn", new_callable=AsyncMock):
                await ctrl.send("Inlined code snippet", attachments=[])
                last_msg = session.messages[-1]
                assert last_msg.content == "Inlined code snippet"
                assert last_msg.display_content is None
        finally:
            active_turns.cancel_and_forget(session.id)
            ctrl._active_task = None

        # 5. Send while turn is running -> queues processed prompt
        running_task = asyncio.create_task(asyncio.sleep(10))
        active_turns.start(
            session.id, running_task, TurnStreamCollector(), started_at=100.0
        )
        try:
            await ctrl.send("Queued prompt", attachments=[att])
            assert len(session.queued_messages) >= 1
            queued = session.queued_messages[-1]
            assert "<user_input_file" in queued.text
            assert queued.visible_text == "Queued prompt"
        finally:
            active_turns.cancel_and_forget(session.id)
            with pytest.raises(asyncio.CancelledError):
                await running_task

        with patch.object(ctrl, "_run_turn", new_callable=AsyncMock):
            ctrl._check_and_dispatch_next_queued(session)
            assert session.messages[-1].content == queued.text
            assert session.messages[-1].display_content == "Queued prompt"
            assert not session.queued_messages
        active_turns.cancel_and_forget(session.id)


def test_sanitize_attachment_filename() -> None:
    # 1. Normal filename
    stem, suffix = sanitize_attachment_filename("document.pdf")
    assert stem == "document"
    assert suffix == ".pdf"

    # 2. Insanely long filename (> 60 chars)
    long_name = ("a" * 80) + ".tar.gz"
    stem_long, suffix_long = sanitize_attachment_filename(long_name, max_stem_len=60)
    assert len(stem_long) <= 60
    assert stem_long == "a" * 60
    assert suffix_long == ".gz"

    # 3. Invalid characters
    stem_inv, suffix_inv = sanitize_attachment_filename("bad:file*name?.txt")
    assert ":" not in stem_inv
    assert "*" not in stem_inv
    assert "?" not in stem_inv
    assert suffix_inv == ".txt"

    # 4. Empty or dot-only filename
    stem_empty, suffix_empty = sanitize_attachment_filename("")
    assert stem_empty == "uploaded_file"
    assert suffix_empty == ""

    stem_dots, suffix_dots = sanitize_attachment_filename("....")
    assert stem_dots == "uploaded_file"

    # 5. Trailing dots and spaces stripping
    stem_trail, suffix_trail = sanitize_attachment_filename("document. ")
    assert stem_trail == "document"

    stem_hidden, suffix_hidden = sanitize_attachment_filename(".hidden")
    assert stem_hidden == ".hidden"
    assert suffix_hidden == ""

    # 6. max_stem_len=0 fallback
    stem_zero, suffix_zero = sanitize_attachment_filename(
        "document.pdf", max_stem_len=0
    )
    assert stem_zero == "file"
    assert suffix_zero == ".pdf"


def test_create_staged_attachment_from_file(tmp_path: Path) -> None:
    # 1. Text file with explicit work_root
    text_data = b"print('Hello NiceGUI')"
    att_text = create_staged_attachment_from_file(
        "script.py", text_data, work_root=tmp_path
    )
    assert att_text.is_binary is False
    assert att_text.filename == "script.py"
    assert att_text.content == "print('Hello NiceGUI')"
    assert att_text.token_count > 0
    assert att_text.file_size == len(text_data)
    assert att_text.file_path is not None and att_text.file_path.exists()
    assert att_text.relative_path is not None and att_text.relative_path.startswith(
        "tmp/"
    )

    # 2. Binary file with null bytes
    bin_data = b"\x00\x01\x02\x03\x04"
    att_bin = create_staged_attachment_from_file(
        "data.bin", bin_data, work_root=tmp_path
    )
    assert att_bin.is_binary is True
    assert att_bin.content == ""
    assert att_bin.token_count == 0
    assert att_bin.preview == "data.bin"

    # 3. File with non-UTF-8 bytes (UnicodeDecodeError fallback)
    corrupt_data = b"\xff\xfe\xfd\x80\x81"
    att_corrupt = create_staged_attachment_from_file(
        "corrupt.dat", corrupt_data, work_root=tmp_path
    )
    assert att_corrupt.is_binary is True
    assert att_corrupt.token_count == 0

    # 4. Empty filename fallback
    att_anon = create_staged_attachment_from_file(
        "", b"anonymous file content", work_root=tmp_path
    )
    assert att_anon.filename.startswith("uploaded_file")

    # 5. Default work_root fallback (using config.work_root)
    old_root = config.work_root
    try:
        config.work_root = str(tmp_path)
        att_default = create_staged_attachment_from_file(
            "default.txt", b"default content"
        )
        assert att_default.file_path is not None and att_default.file_path.exists()
    finally:
        config.work_root = old_root


def test_staged_attachment_binary_recreation(tmp_path: Path) -> None:
    # 1. Binary attachment recreated if file deleted
    att = StagedAttachment(
        id="bin1",
        content="",
        preview="data.bin",
        token_count=0,
        filename="data.bin",
        is_binary=True,
        data=b"\x00original binary contents",
    )
    ref = att.reference(tmp_path)
    assert '<user_input_file path="tmp/' in ref
    assert att.file_path is not None and att.file_path.exists()

    # 2. Binary attachment without filename
    att_noname = StagedAttachment(
        id="bin2",
        content="",
        preview="file",
        token_count=0,
        filename="",
        is_binary=True,
        data=b"\x00original binary contents",
    )
    ref2 = att_noname.reference(tmp_path)
    assert '<user_input_file path="tmp/' in ref2
    assert att_noname.file_path is not None and att_noname.file_path.exists()


def test_file_icons_and_size_formatting() -> None:
    # Format file size
    assert format_file_size(500) == "500 B"
    assert format_file_size(1500) == "1.5 KB"
    assert format_file_size(2 * 1024 * 1024) == "2.0 MB"
    assert format_file_size(3 * 1024 * 1024 * 1024) == "3.0 GB"

    # Extension icons
    icon, color, ext = get_file_icon("doc.pdf")
    assert icon == "picture_as_pdf"
    assert "rose" in color
    assert ext == ".pdf"

    icon_py, _, _ = get_file_icon("code.py")
    assert icon_py == "terminal"

    icon_js, _, _ = get_file_icon("index.js")
    assert icon_js == "code"

    icon_json, _, _ = get_file_icon("data.json")
    assert icon_json == "data_object"

    icon_csv, _, _ = get_file_icon("table.csv")
    assert icon_csv == "table_chart"

    icon_img, _, _ = get_file_icon("picture.png")
    assert icon_img == "image"

    icon_zip, _, _ = get_file_icon("archive.zip")
    assert icon_zip == "folder_zip"

    # MIME fallbacks
    icon_m_img, _, _ = get_file_icon("unknown", mime_type="image/jpeg")
    assert icon_m_img == "image"

    icon_m_aud, _, _ = get_file_icon("unknown", mime_type="audio/mp3")
    assert icon_m_aud == "audiotrack"

    icon_m_vid, _, _ = get_file_icon("unknown", mime_type="video/mp4")
    assert icon_m_vid == "videocam"

    icon_m_pdf, _, _ = get_file_icon("unknown", mime_type="application/pdf")
    assert icon_m_pdf == "picture_as_pdf"

    icon_m_txt, _, _ = get_file_icon("unknown", mime_type="text/plain")
    assert icon_m_txt == "article"

    icon_def, _, _ = get_file_icon("mystery.xyz")
    assert icon_def == "insert_drive_file"

    icon_empty, _, _ = get_file_icon("")
    assert icon_empty == "insert_drive_file"
    assert get_file_icon(".env")[0] == "security"


async def test_input_bar_file_drop_handling(ui_context: Client, tmp_path: Path) -> None:
    with ui_context:
        ib = InputBarComponent(
            on_send=lambda *args: None,
            on_stop=lambda: None,
            temp_storage_getter=lambda _session_id: TempStorage(tmp_path),
        )
        ib.build()
        assert ib.drop_overlay is not None

        # 1. stage_file directly
        att_staged = await ib.stage_file("test_direct.py", b"x = 42")
        assert att_staged in ib.staged_attachments
        assert (
            ib.attachments_container is not None
            and ib.attachments_container.visible is True
        )

        # 2. _handle_files_dropped with empty / non-list-dict args
        await ib._handle_files_dropped(MagicMock(args=None))
        await ib._handle_files_dropped(MagicMock(args="not a list"))

        # 3. _handle_files_dropped with pre-staged file info
        mock_event1 = MagicMock(
            args=[
                {
                    "id": "item1",
                    "filename": "report.pdf",
                    "preview": "report.pdf",
                    "token_count": 0,
                    "file_path": str(tmp_path / "tmp" / "report.pdf"),
                    "relative_path": "tmp/report.pdf",
                    "is_binary": True,
                    "file_size": 2048,
                }
            ]
        )
        await ib._handle_files_dropped(mock_event1)
        assert len(ib.staged_attachments) == 2
        assert ib.staged_attachments[-1].is_binary is True
        assert ib.staged_attachments[-1].filename == "report.pdf"

        # 4. _handle_files_dropped with data URL
        b64_str = base64.b64encode(b"import os").decode("utf-8")

        mock_event2 = MagicMock(
            args={
                "files": [
                    {
                        "filename": "script2.py",
                        "data": f"data:text/plain;base64,{b64_str}",
                    }
                ]
            }
        )
        await ib._handle_files_dropped(mock_event2)
        assert len(ib.staged_attachments) == 3
        assert ib.staged_attachments[-1].content == "import os"

        # 5. _handle_files_dropped with raw bytes
        mock_event3 = MagicMock(
            args=[
                {"name": "raw.txt", "data": b"raw bytes content"},
                "invalid_non_dict",
                {"name": "invalid_data_type", "data": 12345},
            ]
        )
        await ib._handle_files_dropped(mock_event3)
        assert len(ib.staged_attachments) == 4

        # 6. _handle_files_dropped_error
        ib._handle_files_dropped_error(MagicMock(args={"error": "Test upload error"}))
        ib._handle_files_dropped_error(MagicMock(args="String upload error"))

        # 7. Render attachment chips for binary and text
        ib.render_attachments()

        # 8. Test insert attachment into message for text file
        text_att = [a for a in ib.staged_attachments if not a.is_binary][0]
        ib.textarea.set_value("Existing")
        ib._insert_attachment_into_message(text_att)
        assert "Existing" in (ib.textarea.value or "")


@pytest.mark.asyncio
async def test_chat_view_binary_file_modal_and_rendering(
    ui_context: Client, tmp_path: Path
) -> None:
    with ui_context:
        session = session_manager.get_current_session()
        session.settings.work_root = str(tmp_path)
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)

        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Binary file with null bytes
        bin_file = tmp_dir / "sample.bin"
        bin_file.write_bytes(b"BIN\x00\x01\x02")
        await cv._show_attached_file_modal("tmp/sample.bin")

        # 2. File with UnicodeDecodeError
        bad_utf8 = tmp_dir / "bad_utf8.txt"
        bad_utf8.write_bytes(b"\xff\xfe\x80\x81")
        await cv._show_attached_file_modal("tmp/bad_utf8.txt")

        # 3. Message rendering with binary attachment
        msg = ChatMessage(
            role="user",
            content='Review this binary file:\n\n<user_input_file path="tmp/sample.bin">\nPath: tmp/sample.bin\n</user_input_file>',
        )
        cv._render_message(msg)


def test_upload_endpoint(tmp_path: Path) -> None:
    project = session_manager.create_project(tmp_path)
    session = session_manager.create_session(project_id=project.id)

    upload_app = FastAPI()
    upload_app.include_router(upload_router)
    client = TestClient(upload_app)
    # Upload multiple files
    files = [
        ("files", ("test1.py", b"print('one')", "text/x-python")),
        ("files", ("test2.png", png_bytes(), "image/png")),
    ]
    response = client.post(
        "/_exmekh/upload", data={"session_id": session.id}, files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["filename"] == "test1.py"
    assert data[0]["is_binary"] is False
    assert data[1]["filename"] == "test2.png"
    assert data[1]["is_binary"] is True

    # Upload single file under 'file' key
    single_file = [("file", ("single.txt", b"single file text", "text/plain"))]
    resp_single = client.post(
        "/_exmekh/upload", data={"session_id": session.id}, files=single_file
    )
    assert resp_single.status_code == 200
    data_single = resp_single.json()
    assert len(data_single) == 1
    assert data_single[0]["filename"] == "single.txt"


@pytest.mark.parametrize("delete_chat", [False, True])
def test_upload_endpoint_reports_staging_errors_and_deleted_chats(
    tmp_path: Path, delete_chat: bool
) -> None:
    project = session_manager.create_project(tmp_path)
    session = session_manager.create_session(project_id=project.id)
    upload_app = FastAPI()
    upload_app.include_router(upload_router)

    def stage(*args, **kwargs):
        if not delete_chat:
            raise PermissionError("read-only")
        attachment = create_staged_attachment_from_file(*args, **kwargs)
        session_manager.delete_session(session.id)
        return attachment

    with patch(
        "exmekh_agent.ui.file_upload.create_staged_attachment_from_file",
        side_effect=stage,
    ):
        response = TestClient(upload_app).post(
            "/_exmekh/upload",
            data={"session_id": session.id},
            files=[("file", ("code.py", b"print(42)", "text/plain"))],
        )
    assert response.status_code == (404 if delete_chat else 400)
    assert "error" in response.json()
    assert session_manager.temp_storage(session).stats().total_files == 0


async def test_chat_drafts_preserve_text_and_slow_uploads(
    ui_context: Client, tmp_path: Path
) -> None:
    scopes = {
        "first": TempStorage(tmp_path, "tmp/projects/first"),
        "second": TempStorage(tmp_path, "tmp/projects/second"),
    }
    started = threading.Event()
    release = threading.Event()

    def stage(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return create_staged_attachment_from_file(*args, **kwargs)

    async def upload(bar):
        with ui_context:
            return await bar.stage_file("code.py", b"print(42)")

    with ui_context:
        bar = InputBarComponent(
            on_send=lambda _: None,
            on_stop=lambda: None,
            temp_storage_getter=lambda session_id: scopes[session_id],
        )
        bar.build()
        bar.set_session_id("first")
        bar.set_text("First draft")
        with patch(
            "exmekh_agent.ui.input_bar.create_staged_attachment_from_file",
            side_effect=stage,
        ):
            task = asyncio.create_task(upload(bar))
            try:
                assert await asyncio.get_running_loop().run_in_executor(
                    None, started.wait, 3
                )
                bar.set_session_id("second")
                bar.set_text("Second draft")
            finally:
                release.set()
            attachment = await task
        assert bar.staged_attachments == []
        assert scopes["first"].contains(attachment.file_path)
        bar.set_session_id("first")
        assert bar.textarea.value == "First draft"
        assert bar.staged_attachments == [attachment]
        bar.set_session_id("second")
        assert bar.textarea.value == "Second draft"

        late = create_staged_attachment_from_file(
            "late.txt", b"late", storage=scopes["first"]
        )
        await bar._handle_files_dropped(
            MagicMock(args={"session_id": "first", "files": [late.upload_payload()]})
        )
        assert bar.staged_attachments == []
        bar.set_session_id("first")
        assert [item.filename for item in bar.staged_attachments] == [
            "code.py",
            "late.txt",
        ]


def test_get_temp_storage_stats(tmp_path: Path) -> None:
    # 0. TempFileInfo dataclass
    info = TempFileInfo(name="sample.txt", size=100, modified=123456.0)
    assert info.name == "sample.txt"
    assert info.size == 100
    assert info.modified == 123456.0

    # 1. Non-existent directory
    empty_root = tmp_path / "empty_dir"
    stats = TempStorage(empty_root).stats()
    assert stats.total_files == 0
    assert stats.total_size == 0
    assert stats.files == []
    assert stats.is_over_limit is False

    # 2. Path exists but is a file, not directory
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("hello", encoding="utf-8")
    stats_file = TempStorage(file_path).stats()
    assert stats_file.total_files == 0
    assert stats_file.total_size == 0
    assert stats_file.files == []

    # 3. Real tmp directory with files
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    f1 = tmp_dir / "file_a.txt"
    f1.write_bytes(b"hello world")
    f2 = tmp_dir / "file_b.bin"
    f2.write_bytes(b"binary data 12345")
    # Subdirectory should be ignored
    sub_dir = tmp_dir / "subfolder"
    sub_dir.mkdir()

    stats = TempStorage(tmp_path).stats()
    assert stats.total_files == 2
    assert stats.total_size == 11 + 17
    assert len(stats.files) == 2
    assert stats.files[0].name == "file_a.txt"
    assert stats.files[0].size == 11
    assert stats.files[1].name == "file_b.bin"
    assert stats.files[1].size == 17
    assert stats.is_over_limit is False

    # 4. Default work_root (work_root=None uses config.work_root)
    old_root = config.work_root
    try:
        config.work_root = str(tmp_path)
        stats_default = TempStorage().stats()
        assert stats_default.total_files == 2
    finally:
        config.work_root = old_root

    # 5. Over limit property
    big_stats = TempStorageStats(
        folder_path=tmp_dir,
        total_files=1,
        total_size=TEMP_STORAGE_ALERT_THRESHOLD_BYTES + 100,
        files=[],
    )
    assert big_stats.is_over_limit is True

    # 6. OSError on iterdir()
    with patch.object(Path, "iterdir", side_effect=OSError("Disk error")):
        err_stats = TempStorage(tmp_path).stats()
        assert err_stats.total_files == 0
        assert err_stats.files == []

    # 7. OSError on stat()
    real_stat = Path.stat

    def mock_stat(self, *args, **kwargs):
        if self.name == "file_a.txt":
            raise OSError("Stat error on file")
        return real_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", side_effect=mock_stat, autospec=True):
        err_stat_stats = TempStorage(tmp_path).stats()
        assert err_stat_stats.total_files == 1
        assert len(err_stat_stats.files) == 1
        assert err_stat_stats.files[0].name == "file_b.bin"


def test_clean_temp_storage(tmp_path: Path) -> None:
    # 1. Non-existent directory
    empty_root = tmp_path / "non_existent"
    deleted, freed = TempStorage(empty_root).clean()
    assert deleted == 0
    assert freed == 0

    # 2. Path is a file, not directory
    file_path = tmp_path / "file_tmp"
    file_path.write_text("test", encoding="utf-8")
    deleted, freed = TempStorage(file_path).clean()
    assert deleted == 0
    assert freed == 0

    # 3. Directory with files and a subfolder
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    f1 = tmp_dir / "f1.txt"
    f1.write_bytes(b"content 1")
    f2 = tmp_dir / "f2.txt"
    f2.write_bytes(b"longer content 2")
    sub_dir = tmp_dir / "sub"
    sub_dir.mkdir()

    deleted, freed = TempStorage(tmp_path).clean()
    assert deleted == 2
    assert freed == len(b"content 1") + len(b"longer content 2")
    assert not f1.exists()
    assert not f2.exists()
    assert sub_dir.exists()

    # 4. Clean already empty directory
    deleted2, freed2 = TempStorage(tmp_path).clean()
    assert deleted2 == 0
    assert freed2 == 0

    # 5. Default work_root (work_root=None)
    f3 = tmp_dir / "f3.txt"
    f3.write_bytes(b"default root")
    old_root = config.work_root
    try:
        config.work_root = str(tmp_path)
        del_def, freed_def = TempStorage().clean()
        assert del_def == 1
        assert freed_def == len(b"default root")
        assert not f3.exists()
    finally:
        config.work_root = old_root

    # 6. OSError on iterdir()
    with patch.object(Path, "iterdir", side_effect=OSError("Iter error")):
        del_err, freed_err = TempStorage(tmp_path).clean()
        assert del_err == 0
        assert freed_err == 0

    # 7. OSError on unlink()
    f4 = tmp_dir / "f4.txt"
    f4.write_bytes(b"locked file")
    real_unlink = Path.unlink

    def mock_unlink(self, *args, **kwargs):
        if self.name == "f4.txt":
            raise OSError("Locked")
        return real_unlink(self, *args, **kwargs)

    with patch.object(Path, "unlink", side_effect=mock_unlink, autospec=True):
        del_lock, freed_lock = TempStorage(tmp_path).clean()
        assert del_lock == 0
        assert freed_lock == 0
    f4.unlink()


async def test_settings_drawer_tmp_storage_ui(
    ui_context: Client, tmp_path: Path
) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    f1 = tmp_dir / "report.pdf"
    f1.write_bytes(b"%PDF-sample pdf content")

    drawer = SettingsDrawerComponent(
        on_settings_saved=lambda: None,
        temp_storage_getter=lambda: TempStorage(tmp_path),
    )
    with ui_context:
        drawer.build()
        await drawer.refresh_tmp_stats()
        assert drawer.tmp_path_label is not None
        assert str(tmp_path / "tmp") in drawer.tmp_path_label.text
        assert drawer.tmp_badge is not None
        assert "1 file" in drawer.tmp_badge.text
        assert drawer.clean_tmp_btn is not None
        assert drawer.refresh_tmp_btn is not None

        # Refresh stats via handler
        with patch("nicegui.ui.notify") as mock_notify:
            await drawer._handle_refresh_tmp()
            mock_notify.assert_called_once()
            assert (
                "Temporary storage statistics refreshed" in mock_notify.call_args[0][0]
            )

        # Clean files via handler
        with patch("nicegui.ui.notify") as mock_notify:
            await drawer._handle_clean_tmp()
            mock_notify.assert_called_once()
            assert "Temporary storage cleared" in mock_notify.call_args[0][0]
            assert "1 file deleted" in mock_notify.call_args[0][0]

        # Stats should be updated now (0 files)
        assert "0 files (0 B)" in drawer.tmp_badge.text
        assert not f1.exists()

        # Clean again when empty
        with patch("nicegui.ui.notify") as mock_notify:
            await drawer._handle_clean_tmp()
            mock_notify.assert_called_once()
            assert "No temporary files to delete" in mock_notify.call_args[0][0]

        # Test toggle and open refresh stats
        drawer.open()
        drawer.toggle()


async def test_page_controller_temp_storage_alert(
    ui_context: Client, tmp_path: Path
) -> None:
    controller = ChatPageController(ui_context)
    session = session_manager.get_current_session()
    session.settings.work_root = str(tmp_path)

    # 1. Under limit: no alert
    with patch("nicegui.ui.notify") as mock_notify:
        await controller._check_temp_storage_alert()
        mock_notify.assert_not_called()

    # 2. Over limit (mocking stats.is_over_limit)
    mock_stats = TempStorageStats(
        folder_path=tmp_path / "tmp",
        total_files=5,
        total_size=1024 * 1024 * 1024 + 500,
        files=[],
    )
    with (
        patch.object(TempStorage, "stats", return_value=mock_stats),
        patch("nicegui.ui.notify") as mock_notify,
    ):
        await controller._check_temp_storage_alert()
        mock_notify.assert_called_once()
        args, kwargs = mock_notify.call_args
        assert "High temporary storage usage" in args[0]
        assert "1.0 GB" in args[0]
        assert "5 file" in args[0]
        assert kwargs["type"] == "warning"
        assert kwargs["position"] == "bottom-right"
        assert kwargs["close_button"] is True
        assert kwargs["timeout"] == 8000

    # 3. Client is deleted: returns early
    with (
        patch.object(
            type(ui_context), "is_deleted", new_callable=PropertyMock, return_value=True
        ),
        patch.object(TempStorage, "stats", return_value=mock_stats),
        patch("nicegui.ui.notify") as mock_notify,
    ):
        await controller._check_temp_storage_alert()
        mock_notify.assert_not_called()
