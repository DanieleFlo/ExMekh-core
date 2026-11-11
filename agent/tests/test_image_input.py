from __future__ import annotations

import asyncio
import json
from io import BytesIO
from unittest.mock import patch

import pytest
from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from exmekh_lib.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from exmekh_lib.core.providers.to_openai_response_mapper import ToOpenAIResponseMapper
from fastapi import FastAPI
from nicegui.client import Client as UIClient
from nicegui.page import page
from PIL import Image
from starlette.testclient import TestClient

from exmekh_agent.exmekh_bridge.agents.agent_runner import run_agent_turn
from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.history.converter import (
    chat_message_to_exmekh_messages,
    exmekh_message_to_raw_dict,
    exmekh_messages_to_chat_messages,
    raw_dict_to_exmekh_message,
)
from exmekh_agent.exmekh_bridge.history.image_input import (
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
from exmekh_agent.exmekh_bridge.history.message_attachments import saved_images
from exmekh_agent.exmekh_bridge.history.message_edit import prepare_message_edit
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
    format_user_input_file_xml,
    prepare_user_input,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    QueuedMessage,
)
from exmekh_agent.exmekh_bridge.models.messages import ImageReference
from exmekh_agent.exmekh_bridge.providers.context_usage import count_message_dict_tokens
from exmekh_agent.exmekh_bridge.providers.ds4_prompt import (
    render_ds4_qwen_prompt,
)
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.ui.chat.chat_export import export_chat
from exmekh_agent.ui.chat.image_attachment import render_image_attachment
from exmekh_agent.ui.shared.file_upload import upload_router
from exmekh_agent.ui.shared.image_preview import (
    render_image_thumbnail,
    show_image_preview,
)
from tests.capability_fixtures import TEST_CAPABILITIES
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


def test_image_edit_reference_survives_queue_and_message_edit(tmp_path):
    storage = TempStorage(tmp_path)
    reference = create_staged_attachment_from_file(
        "reference.png", png_bytes(), storage=storage
    )
    reference.image_destination = "image"
    prepared = prepare_user_input(
        "Make it blue", attachments=[reference], storage=storage
    )
    queued = ChatSession().add_queued_message(
        prepared.prompt,
        prepared.display_content,
        image_paths=prepared.image_paths,
        image_references=prepared.image_references,
        attachment_paths=prepared.attachment_paths,
    )
    queued = QueuedMessage.model_validate_json(queued.model_dump_json())
    saved = queued.to_chat_message()
    assert saved.visible_image_urls == reference.image_urls
    edited = prepare_message_edit(
        saved, "Make it green", storage=storage, use_file_references=True
    )
    assert not edited.image_urls
    assert edited.image_references == prepared.image_references
    assert reference.relative_path not in edited.prompt
    assert edited.display_content == "Make it green"


def test_legacy_history_keeps_saved_routes_next_to_pdf_previews(tmp_path):
    storage = TempStorage(tmp_path)
    document = create_staged_attachment_from_file(
        "report.pdf", pdf_bytes(), storage=storage
    )
    vision = create_staged_attachment_from_file(
        "vision.png", png_bytes(), storage=storage
    )
    edit = create_staged_attachment_from_file(
        "edit.png", png_bytes("blue"), storage=storage
    )
    both = create_staged_attachment_from_file(
        "both.png", png_bytes("green"), storage=storage
    )
    edit.image_destination = "image"
    both.image_destination = "both"
    prepared = prepare_user_input(
        "Edit these", attachments=[edit, document, vision, both], storage=storage
    )
    saved = prepared.to_chat_message()
    assert [card.filename for card in saved_images(saved) if card.path] == [
        "vision.png",
        "both.png",
        "edit.png",
    ]
    legacy = ChatMessage.model_validate(saved.model_dump(exclude={"image_attachments"}))
    cards = saved_images(legacy)
    assert [card.url for card in cards] == legacy.visible_image_urls
    assert [card.destination for card in cards] == [None, None, "llm", "both", "image"]
    assert [card.path for card in cards[2:]] == [
        vision.relative_path,
        both.relative_path,
        edit.relative_path,
    ]

    # Older records without a PDF page count keep their previews without
    # associating a PDF page with the image that follows it.
    without_notice = legacy.model_copy(update={"content": "Edit these"})
    assert [card.path for card in saved_images(without_notice)] == [
        None,
        None,
        None,
        None,
        edit.relative_path,
    ]


def test_compact_upload_and_legacy_pdf_use_scoped_storage(tmp_path):
    storage = TempStorage(tmp_path)
    pdf = create_staged_attachment_from_file("pages.pdf", pdf_bytes(), storage=storage)
    descriptor = pdf.upload_payload(compact=True)
    assert "image_urls" not in descriptor
    restored = StagedAttachment.from_upload(descriptor, storage=storage)
    assert restored.image_urls == pdf.image_urls
    prepared = prepare_user_input(restored.reference(storage=storage), storage=storage)
    assert prepared.image_urls == pdf.image_urls
    assert not prepared.image_attachments

    inline = StagedAttachment.from_file_data("notes.txt", b"Unstaged text")
    prepared = prepare_user_input(
        "Read", storage=storage, attachments=[inline], use_file_references=False
    )
    assert "Unstaged text" in prepared.prompt
    assert prepared.attachment_paths == []


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
    with patch(
        "exmekh_agent.exmekh_bridge.history.image_input.pdfium.PdfDocument"
    ) as cls:
        cls.return_value.__enter__.return_value.__len__.return_value = 0
        with pytest.raises(ValueError, match="no pages"):
            attachment_images("empty.pdf", b"%PDF-")
    with patch(
        "exmekh_agent.exmekh_bridge.history.image_input.Image.open",
        side_effect=Image.DecompressionBombError,
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
    raw = exmekh_message_to_raw_dict(message)
    assert [part["type"] for part in raw["content"]] == [
        "text",
        "image_url",
        "text",
        "text",
    ]
    restored = raw_dict_to_exmekh_message(json.loads(json.dumps(raw)))
    assert serialize_user_content(restored) == raw["content"]
    chat = exmekh_messages_to_chat_messages([restored])[0]
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


def test_both_exmekh_provider_mappers_send_png_images_in_one_user_message():
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
    exmekh_message = chat_message_to_exmekh_messages(restored.messages[0])[0]
    assert exmekh_messages_to_chat_messages([exmekh_message]) == [message]
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
    raw = exmekh_message_to_raw_dict(UserMessage(user_content("Inspect", [url])))
    assert count_message_dict_tokens(raw) == count_message_dict_tokens(
        {"content": "Inspect"}
    )
    messages = [raw]
    prompt = render_ds4_qwen_prompt(messages, [])
    assert "Inspect" in prompt and "base64" not in prompt


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
async def test_real_exmekh_runner_receives_and_replays_images(tmp_path, mode):
    url = image_data_url(Image.new("RGB", (2, 2), "red"))
    provider = MockProvider()
    histories = []
    with patch(
        "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
        return_value=provider,
    ):
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
                capabilities_path=str(TEST_CAPABILITIES),
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
                capabilities_path=str(TEST_CAPABILITIES),
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
                capabilities_path=str(TEST_CAPABILITIES),
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
    with patch(
        "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
        return_value=provider,
    ):
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
                    capabilities_path=str(TEST_CAPABILITIES),
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
    replay = chat_message_to_exmekh_messages(message)
    guidance = next(item for item in replay if isinstance(item, UserMessage))
    assert isinstance(list(guidance.iter_content_parts())[1], Image.Image)


def test_steering_compares_images_and_retains_unaccepted_guidance():
    url = image_data_url(Image.new("RGB", (2, 2), "red"))
    steering = TurnSteering()
    steering.bind(lambda _: None, history_start=1)
    assert steering.send(
        QueuedMessage(
            text="Same text",
            image_urls=[url],
            image_paths=["preview.png", "edit.png"],
            image_references=[ImageReference(path="edit.png", image_only_preview=url)],
        )
    )
    assert steering.image_routes() == {}
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
    assert steering.image_routes() == {"preview.png": False, "edit.png": True}


def test_thumbnails_and_modal_build_in_the_client_context():
    client = UIClient(page("/image-preview-test"))
    with client:
        url = image_data_url(Image.new("RGB", (2, 2), "red"))
        render_image_thumbnail(url, "Red")
        show_image_preview(url, "Red")
        render_image_attachment(url, "Legacy image", None, "llm", card_class="history")
        toggle = next(
            element
            for element in client.elements.values()
            if "image-attachment-destination" in element.classes
        )
        assert toggle.props["readonly"] is True
        assert any(
            "exmekh-image-thumbnail" in element.classes
            for element in client.elements.values()
        )
        assert any(
            "exmekh-image-dialog" in element.classes
            for element in client.elements.values()
        )


def test_partial_upload_failure_removes_previously_staged_files(tmp_path):
    project = session_manager.create_project(tmp_path)
    session = session_manager.create_session(project_id=project.id)
    app = FastAPI()
    app.include_router(upload_router)
    with TestClient(app) as client:
        result = client.post(
            "/_exmekh/upload",
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
