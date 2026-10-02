from __future__ import annotations

import asyncio
import json
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from nicegui.client import Client as UIClient
from nicegui.page import page
from PIL import Image
from qs.core.agents.node import DefaultCapabilities
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from qs.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from qs.core.providers.to_openai_response_mapper import ToOpenAIResponseMapper
from starlette.testclient import TestClient

from fe.qs_bridge.agent_runner import run_agent_turn
from fe.qs_bridge.context_usage import count_message_dict_tokens
from fe.qs_bridge.converter import (
    chat_message_to_qs_messages,
    qs_message_to_raw_dict,
    qs_messages_to_chat_messages,
    raw_dict_to_qs_message,
)
from fe.qs_bridge.ds4_prompt import apply_ds4_scope, render_ds4_qwen_prompt
from fe.qs_bridge.image_input import (
    MAX_IMAGE_SIDE,
    MAX_PDF_PAGES,
    attachment_images,
    content_text,
    deserialize_user_content,
    image_data_url,
    image_from_url,
    restore_image_history,
    serialize_user_content,
    user_content,
)
from fe.qs_bridge.models import ChatMessage, ChatSession, ChatSettings, QueuedMessage
from fe.qs_bridge.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
    format_user_input_file_xml,
    prepare_user_input,
)
from fe.qs_bridge.steering import TurnSteering
from fe.state.session_manager import SessionManager, session_manager
from fe.ui.chat_export import export_chat
from fe.ui.file_upload import upload_router
from fe.ui.image_preview import render_image_thumbnail, show_image_preview
from tests.image_samples import pdf_bytes, png_bytes
from tests.mock_client import MockProvider
from tests.steering_probe import SteeringProbeProvider


@pytest.mark.parametrize("references", [True, False])
def test_multiple_images_and_pdf_are_one_ordered_user_message(tmp_path, references):
    attachments = [
        create_staged_attachment_from_file("red.png", png_bytes(), tmp_path),
        create_staged_attachment_from_file("blue.png", png_bytes("blue"), tmp_path),
        create_staged_attachment_from_file("pages.pdf", pdf_bytes(), tmp_path),
    ]
    prepared = prepare_user_input(
        "Compare these",
        attachments=attachments,
        work_root=tmp_path,
        use_file_references=references,
    )
    assert prepared.display_content == "Compare these"
    assert len(prepared.image_urls) == 4
    assert attachments[-1].pdf_pages == 2
    assert "pages 1–2 of 2" in prepared.prompt
    message = UserMessage(user_content(prepared.prompt, prepared.image_urls))
    parts = list(message.iter_content_parts())
    assert parts[0] == prepared.prompt
    assert len(parts) == 5
    assert parts[1].getpixel((0, 0)) == (255, 0, 0)
    assert parts[2].getpixel((0, 0)) == (0, 0, 255)
    assert parts[3].getpixel((0, 0))[0] > 240
    assert parts[4].getpixel((0, 0))[2] > 240
    descriptor = attachments[0].upload_payload()
    restored = StagedAttachment.from_upload(descriptor, tmp_path)
    assert restored.image_urls == attachments[0].image_urls
    assert restored.pdf_pages == 0
    assert attachments[0].file_path.read_bytes() == png_bytes()


def test_pdf_limit_is_visible_in_the_prompt_and_does_not_render_extra_pages(tmp_path):
    data = pdf_bytes(MAX_PDF_PAGES + 2)
    attachment = create_staged_attachment_from_file("long.pdf", data, tmp_path)
    assert attachment.pdf_pages == 12
    assert len(attachment.image_urls) == MAX_PDF_PAGES
    prepared = prepare_user_input("", attachments=[attachment], work_root=tmp_path)
    assert "pages 1–10 of 12" in prepared.prompt
    assert "Remaining pages were not sent" in prepared.prompt
    assert prepared.title_text == "long.pdf"
    assert attachment.file_path.read_bytes() == data


def test_image_decoding_uses_contents_and_bounds_resolution():
    urls, pages = attachment_images("no-extension", png_bytes(size=(3000, 1500)))
    assert pages == 0
    assert image_from_url(urls[0]).size == (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE // 2)
    assert attachment_images("notes.txt", b"Hello") == ([], 0)
    urls, pages = attachment_images("document", pdf_bytes(1))
    assert len(urls) == pages == 1


def test_image_exif_orientation_is_applied():
    source = Image.new("RGB", (20, 10), "red")
    exif = source.getexif()
    exif[274] = 6
    output = BytesIO()
    source.save(output, format="JPEG", exif=exif)
    urls, _ = attachment_images("rotated.jpg", output.getvalue())
    assert image_from_url(urls[0]).size == (10, 20)


@pytest.mark.parametrize(
    "filename,data",
    [
        ("bad.png", b"broken image"),
        ("bad.pdf", b"%PDF-broken"),
    ],
)
def test_bad_visual_attachments_fail_before_writing(tmp_path, filename, data):
    with pytest.raises(ValueError):
        create_staged_attachment_from_file(filename, data, tmp_path)
    assert not (tmp_path / "tmp").exists()


def test_empty_pdf_and_oversized_image_have_clear_errors():
    with patch("fe.qs_bridge.image_input.pdfium.PdfDocument") as cls:
        cls.return_value.__enter__.return_value.__len__.return_value = 0
        with pytest.raises(ValueError, match="no pages"):
            attachment_images("empty.pdf", b"%PDF-")
    with patch(
        "fe.qs_bridge.image_input.Image.open", side_effect=Image.DecompressionBombError
    ):
        with pytest.raises(ValueError, match="too large"):
            attachment_images("large.png", b"large")


def test_saved_content_round_trips_text_containers_images_and_empty_parts():
    message = UserMessage(
        [
            "Before",
            ImagesUserMessage([Image.new("RGB", (2, 3), "blue")]),
            TextsUserMessage(["After", ""]),
        ]
    )
    raw = qs_message_to_raw_dict(message)
    assert [part["type"] for part in raw["content"]] == [
        "text",
        "image_url",
        "text",
        "text",
    ]
    restored = raw_dict_to_qs_message(json.loads(json.dumps(raw)))
    assert serialize_user_content(restored) == raw["content"]
    chat = qs_messages_to_chat_messages([restored])[0]
    assert chat.content == "Before\nAfter\n"
    assert len(chat.image_urls) == 1
    assert serialize_user_content(UserMessage("Plain")) == "Plain"
    assert deserialize_user_content(None) == ""
    assert deserialize_user_content([]) == []
    assert content_text(raw["content"]) == "Before\nAfter\n"
    with pytest.raises(ValueError, match="Unsupported"):
        deserialize_user_content([{"type": "unknown"}])
    with pytest.raises(ValueError, match="PNG"):
        image_from_url("https://example.com/image.png")
    with pytest.raises(ValueError, match="converts documents"):
        serialize_user_content(
            UserMessage(
                FilesUserMessage(
                    [
                        DocumentUserMessage(b"txt", "notes", "txt", "text/plain"),
                    ]
                )
            )
        )


def test_both_qs_provider_mappers_send_png_images_in_one_user_message():
    url = image_data_url(Image.new("RGB", (2, 2), "blue"))
    message = UserMessage(user_content("Inspect", [url, url]))
    chat = ToOpenAIChatCompletionMapper.map_user_message(message)
    responses = ToOpenAIResponseMapper.map_user_message(message)
    assert chat["role"] == "user"
    assert len(responses) == 1
    assert [part["type"] for part in chat["content"]] == [
        "text",
        "image_url",
        "image_url",
    ]
    assert [part["type"] for part in responses[0]["content"]] == [
        "input_text",
        "input_image",
        "input_image",
    ]
    assert chat["content"][1]["image_url"]["url"] == url
    assert responses[0]["content"][1]["image_url"] == url


def test_images_survive_json_storage_history_replay_and_exports(tmp_path):
    url = image_data_url(Image.new("RGB", (2, 2), "red"))
    manager = SessionManager(storage_dir=tmp_path / "chats")
    session = manager.get_current_session()
    message = ChatMessage(role="user", content="Inspect", image_urls=[url])
    session.add_message(message)
    queued = session.add_queued_message("Next", image_urls=[url])
    assert queued.to_chat_message().image_urls == [url]
    manager.save_session(session)
    restored = SessionManager(storage_dir=tmp_path / "chats").get_session(session.id)
    assert restored.messages[0].image_urls == [url]
    assert restored.queued_messages[0].image_urls == [url]
    qs_message = chat_message_to_qs_messages(restored.messages[0])[0]
    assert qs_messages_to_chat_messages([qs_message]) == [message]
    assert url in export_chat(restored, "json", include_history=False)
    assert f"![Attached image 1]({url})" in export_chat(
        restored, "markdown", include_history=False
    )
    steering = ChatMessage(role="assistant", steering_messages=[queued])
    assert f"![Steering image]({url})" in export_chat(
        ChatSession(messages=[steering]),
        "markdown",
        include_history=True,
    )


def test_context_estimates_do_not_tokenize_binary_base64():
    url = image_data_url(Image.new("RGB", (32, 32), "blue"))
    raw = qs_message_to_raw_dict(UserMessage(user_content("Inspect", [url])))
    assert count_message_dict_tokens(raw) == count_message_dict_tokens(
        {"content": "Inspect"}
    )
    messages = [raw]
    apply_ds4_scope(messages, {"allowed_tools": [], "allowed_skills": []})
    prompt = render_ds4_qwen_prompt(messages, [])
    assert "Inspect" in prompt and "base64" not in prompt


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
async def test_real_qs_runner_receives_and_replays_images(tmp_path, mode):
    url = image_data_url(Image.new("RGB", (2, 2), "red"))
    provider = MockProvider()
    histories = []
    with patch("fe.qs_bridge.agent_runner.LocalProvider", return_value=provider):
        result = await run_agent_turn(
            user_prompt="Inspect",
            image_urls=[url, url],
            model="mock",
            agent_mode=mode,
            on_token=lambda _: None,
            on_reasoning=lambda *_: None,
            on_tool_event=lambda _: None,
            on_raw_history=lambda raw, _: histories.append(raw),
            settings=ChatSettings(
                work_root=str(tmp_path),
                capabilities_path=str(
                    DefaultCapabilities(mocks=True).capabilities_path
                ),
                tool_approval_mode="Approve All",
            ),
        )
        assert result
        user = next(
            message
            for message in reversed(provider.last_request.get_messages())
            if isinstance(message, UserMessage)
        )
        assert (
            len(
                [
                    part
                    for part in user.iter_content_parts()
                    if isinstance(part, Image.Image)
                ]
            )
            == 2
        )
        assert histories[-1][-2]["content"][1]["type"] == "image_url"
        await run_agent_turn(
            user_prompt="Continue",
            model="mock",
            agent_mode=mode,
            on_raw_history=lambda raw, _: histories.append(raw),
            on_token=lambda _: None,
            on_reasoning=lambda *_: None,
            on_tool_event=lambda _: None,
            history=[
                ChatMessage(role="user", content="Inspect", image_urls=[url]),
                ChatMessage(role="assistant", content="Done"),
            ],
            settings=ChatSettings(
                work_root=str(tmp_path),
                capabilities_path=str(
                    DefaultCapabilities(mocks=True).capabilities_path
                ),
            ),
        )
        previous = next(
            message
            for message in provider.last_request.get_messages()
            if isinstance(message, UserMessage)
        )
        assert isinstance(list(previous.iter_content_parts())[1], Image.Image)
        saved_users = [item for item in histories[-1] if item["role"] == "user"]
        assert isinstance(saved_users[0]["content"], list)
        assert saved_users[-1]["content"] == "Continue"
        # On the third turn the assistant raw history is authoritative for replay.
        replayed_history = [
            ChatMessage(role="user", content="Inspect", image_urls=[url]),
            ChatMessage(role="assistant", content="Done"),
            ChatMessage(role="user", content="Continue"),
            ChatMessage(role="assistant", content="Done", raw_history=histories[-1]),
        ]
        await run_agent_turn(
            user_prompt="Once more",
            model="mock",
            agent_mode=mode,
            history=replayed_history,
            on_token=lambda _: None,
            on_reasoning=lambda *_: None,
            on_tool_event=lambda _: None,
            settings=ChatSettings(
                work_root=str(tmp_path),
                capabilities_path=str(
                    DefaultCapabilities(mocks=True).capabilities_path
                ),
            ),
        )
        previous = next(
            message
            for message in provider.last_request.get_messages()
            if isinstance(message, UserMessage)
        )
        assert isinstance(list(previous.iter_content_parts())[1], Image.Image)


async def test_multimodal_steering_is_confirmed_and_restored_in_nested_history(
    tmp_path,
):
    url = image_data_url(Image.new("RGB", (2, 2), "blue"))
    provider = SteeringProbeProvider(first_response="answer")
    steering = TurnSteering()
    histories = []
    with patch("fe.qs_bridge.agent_runner.LocalProvider", return_value=provider):
        task = asyncio.create_task(
            run_agent_turn(
                user_prompt="Start",
                model="mock",
                agent_mode="deep_agent",
                on_token=lambda _: None,
                on_reasoning=lambda *_: None,
                on_tool_event=lambda _: None,
                steering=steering,
                on_raw_history=lambda raw, _: histories.append(raw),
                settings=ChatSettings(
                    work_root=str(tmp_path),
                    capabilities_path=str(
                        DefaultCapabilities(mocks=True).capabilities_path
                    ),
                ),
            )
        )
        try:
            assert await asyncio.to_thread(provider.started.wait, 5)
            assert steering.send(QueuedMessage(text="See image", image_urls=[url]))
            provider.release.set()
            await asyncio.wait_for(task, 5)
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    dumped = UserMessage(user_content("See image", [url])).message_dump()
    message = ChatMessage(
        role="assistant",
        raw_history=histories[-1],
        skill_history_tree=[{"history": [dumped], "children": []}],
    )
    steering.apply_to(message)
    assert len(message.steering_insertions) == 1
    assert message.steering_messages[0].image_urls == [url]
    assert isinstance(message.skill_history_tree[0]["history"][0]["content"], list)
    replay = chat_message_to_qs_messages(message)
    guidance = next(item for item in replay if isinstance(item, UserMessage))
    assert isinstance(list(guidance.iter_content_parts())[1], Image.Image)


def test_steering_compares_images_and_retains_unaccepted_guidance():
    url = image_data_url(Image.new("RGB", (2, 2), "red"))
    steering = TurnSteering()
    steering.bind(lambda _: None, history_start=1)
    assert steering.send(QueuedMessage(text="Same text", image_urls=[url]))
    assert not steering.record_added(
        UserMessage("Same text"),
        after_tool_call_id=None,
        tool_name=None,
        skill_path="/",
    )
    assert steering.record_added(
        UserMessage(user_content("Same text", [url])),
        after_tool_call_id=None,
        tool_name=None,
        skill_path="/",
    )


def test_thumbnails_and_modal_build_in_the_client_context():
    client = UIClient(page("/image-preview-test"))
    with client:
        url = image_data_url(Image.new("RGB", (2, 2), "red"))
        render_image_thumbnail(url, "Red")
        show_image_preview(url, "Red")
        assert any(
            "qs-image-thumbnail" in element.classes
            for element in client.elements.values()
        )
        assert any(
            "qs-image-dialog" in element.classes for element in client.elements.values()
        )


def test_partial_upload_failure_removes_previously_staged_files(tmp_path):
    project = session_manager.create_project(tmp_path)
    session = session_manager.create_session(project_id=project.id)
    app = FastAPI()
    app.include_router(upload_router)
    with TestClient(app) as client:
        result = client.post(
            "/_qs/upload",
            data={"session_id": session.id},
            files=[
                ("files", ("good.png", png_bytes(), "image/png")),
                ("files", ("bad.pdf", b"%PDF-broken", "application/pdf")),
            ],
        )
    assert result.status_code == 400
    assert session_manager.temp_storage(session).stats().total_files == 0


def test_editing_queued_references_restores_images_without_duplicates(tmp_path):
    image = create_staged_attachment_from_file("image.png", png_bytes(), tmp_path)
    prepared = prepare_user_input("Inspect", attachments=[image], work_root=tmp_path)
    edited = prepare_user_input(prepared.prompt, attachments=[], work_root=tmp_path)
    assert edited.image_urls == prepared.image_urls
    assert edited.image_paths == prepared.image_paths
    duplicate = prepare_user_input(
        prepared.prompt, attachments=[image], work_root=tmp_path
    )
    assert duplicate.image_urls == prepared.image_urls
    assert duplicate.image_paths == prepared.image_paths


def test_nonvisual_missing_and_outside_references_do_not_become_images(tmp_path):
    attachment = create_staged_attachment_from_file("notes.txt", b"notes", tmp_path)
    text = "\n".join(
        format_user_input_file_xml(path)
        for path in [attachment.relative_path, "tmp/missing.png", "../outside.png"]
    )
    prepared = prepare_user_input(text, attachments=[], work_root=tmp_path)
    assert prepared.image_urls == []
    assert prepared.image_paths == []


def test_transparent_pixels_get_a_white_background():
    output = BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(output, format="PNG")
    urls, _ = attachment_images("transparent.png", output.getvalue())
    assert image_from_url(urls[0]).getpixel((0, 0)) == (255, 255, 255)


def test_image_hashes_are_restored_in_nested_execution_snapshots():
    message = UserMessage(
        user_content("Inspect", [image_data_url(Image.new("RGB", (2, 2)))])
    )
    dumped = message.message_dump()
    parts = serialize_user_content(message)
    history = [dumped, {"role": "assistant", "content": dumped["content"]}]
    tree = [{"history": [dumped], "children": [{"history": [dumped], "children": []}]}]
    restored, restored_tree = restore_image_history(
        history, tree, {dumped["content"]: parts}
    )
    assert restored[0]["content"] == parts
    assert restored[1]["content"] == dumped["content"]
    assert restored_tree[0]["history"][0]["content"] == parts
    assert restored_tree[0]["children"][0]["history"][0]["content"] == parts
    assert isinstance(history[0]["content"], str)
    assert isinstance(tree[0]["history"][0]["content"], str)
    assert restore_image_history([{"role": "user", "content": parts}], [], {}) == (
        [{"role": "user", "content": parts}],
        [],
    )
