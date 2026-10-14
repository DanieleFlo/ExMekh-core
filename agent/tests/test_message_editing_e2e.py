from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx
import pytest
from playwright.async_api import Page, async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)
from tests.image_samples import pdf_bytes, png_bytes


@pytest.fixture(scope="module")
def edit_server():
    yield from start_mock_server(0, "tests.run_message_edit_agent")


async def state(url: str) -> dict:
    address = urlsplit(url)
    origin = f"{address.scheme}://{address.netloc}"
    async with httpx.AsyncClient(trust_env=False) as client:
        return (await client.get(origin + "/test/state")).json()


async def open_editor(page, index: int = 0):
    message = page.locator(".chat-message-user").nth(index)
    await message.hover()
    await message.get_by_role("button", name="Edit message", exact=True).click()
    editor = page.get_by_role("dialog")
    await expect(editor).to_be_visible()
    return editor


async def test_completed_message_edit_cancel_trim_regenerate_and_reload(edit_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            await page.goto(edit_server, wait_until="networkidle")
            await page.get_by_role("button", name="New Quick Chat", exact=True).click()
            await send_command(page, "First original")
            await wait_turn_completion(page)
            await send_command(page, "Second obsolete")
            await wait_turn_completion(page)
            before = await state(edit_server)
            dialog = await open_editor(page)
            await expect(dialog.get_by_label("Message", exact=True)).to_have_value(
                "First original"
            )
            await dialog.get_by_label("Message", exact=True).fill("Cancelled change")
            await dialog.get_by_role("button", name="Cancel", exact=True).click()
            assert (await state(edit_server))["session"]["messages"] == before[
                "session"
            ]["messages"]
            dialog = await open_editor(page)
            await dialog.get_by_label("Message", exact=True).fill(" ")
            await expect(
                dialog.get_by_role("button", name="Save & regenerate")
            ).to_be_disabled()
            await dialog.get_by_label("Message", exact=True).fill("First revised")
            await dialog.get_by_role("button", name="Save & regenerate").click()
            await expect(dialog).not_to_be_visible()
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Reply: First revised"
            )
            data = await state(edit_server)
            assert [message["content"] for message in data["session"]["messages"]] == [
                "First revised",
                "Reply: First revised",
            ]
            assert (
                data["session"]["messages"][0]["id"]
                == before["session"]["messages"][0]["id"]
            )
            assert data["calls"][-1]["history"] == []
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".chat-message-user")).to_have_count(1)
            await expect(page.locator(".chat-message-user")).to_contain_text(
                "First revised"
            )
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Reply: First revised"
            )
        finally:
            await browser.close()


async def test_edit_reconnected_stream_stops_old_turn_and_clears_queue(edit_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            await page.goto(edit_server, wait_until="networkidle")
            await page.get_by_role("button", name="New Quick Chat", exact=True).click()
            await expect(page.locator(".msg-user")).to_have_count(0)
            await send_command(page, "Slow original")
            await expect(page.locator(".chat-message-streaming")).to_contain_text(
                "Reply: Slow original"
            )
            await send_command(page, "Obsolete queued input")
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            dialog = await open_editor(page)
            await dialog.get_by_label("Message", exact=True).fill("Fresh replacement")
            await dialog.get_by_role("button", name="Save & regenerate").click()
            await expect(dialog).not_to_be_visible()
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Reply: Fresh replacement"
            )
            await expect(page.locator(".chat-message-streaming")).to_have_count(0)
            await expect(page.locator(".generation-stopped")).to_have_count(0)
            data = await state(edit_server)
            assert [message["content"] for message in data["session"]["messages"]] == [
                "Fresh replacement",
                "Reply: Fresh replacement",
            ]
            assert not data["session"]["queued_messages"]
            assert data["calls"][-2]["cancelled"]
            assert data["calls"][-1]["history"] == []
            assert not any(
                call["prompt"] == "Obsolete queued input" for call in data["calls"]
            )
            assert (
                "STALE TOKEN"
                not in await page.locator(".chat-scroll-container").first.inner_text()
            )
        finally:
            await browser.close()


@pytest.fixture
async def edit_page(edit_server: str) -> AsyncIterator[Page]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(edit_server, wait_until="networkidle")
            yield page
        finally:
            await browser.close()


async def new_conversation(page: Page, mode: str) -> None:
    button = "New Quick Chat" if mode == "quick_chat" else "New project chat"
    await page.get_by_role("button", name=button, exact=True).first.click()
    await expect(page.locator(".chat-message-user")).to_have_count(0)


async def wait_saved_response(page: Page, content: str) -> dict:
    async with asyncio.timeout(10):
        while True:
            data = await state(page.url.rstrip("/"))
            messages = data["session"]["messages"]
            if (
                messages
                and messages[-1]["role"] == "assistant"
                and messages[-1]["content"] == content
            ):
                await expect(page.locator(".chat-message-streaming")).to_have_count(0)
                return data
            await asyncio.sleep(0.05)


async def save_edit(page: Page, text: str, index: int = 0) -> None:
    dialog = await open_editor(page, index)
    await dialog.get_by_label("Message", exact=True).fill(text)
    await dialog.get_by_role("button", name="Save & regenerate", exact=True).click()
    await expect(dialog).not_to_be_visible()


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
@pytest.mark.parametrize("edited_index", [0, 1, 2], ids=["first", "middle", "last"])
async def test_edit_position_retains_only_prior_turns_with_mock_client(
    edit_page: Page,
    mode: str,
    edited_index: int,
):
    page = edit_page
    await new_conversation(page, mode)
    inputs = ["First original", "Middle original", "Last original"]
    for text in inputs:
        await send_command(page, text)
        await wait_saved_response(page, "Reply: " + text)
    before = await state(page.url.rstrip("/"))
    replacement = "Revised message"
    await save_edit(page, replacement, edited_index)
    data = await wait_saved_response(page, "Reply: " + replacement)
    prefix = [
        part for text in inputs[:edited_index] for part in [text, "Reply: " + text]
    ]
    expected = prefix + [replacement, "Reply: " + replacement]
    assert [item["content"] for item in data["session"]["messages"]] == expected
    assert data["calls"][-1]["history"] == prefix
    assert (
        data["session"]["messages"][2 * edited_index]["id"]
        == before["session"]["messages"][2 * edited_index]["id"]
    )
    assert data["session"]["agent_mode"] == mode
    # Verify the same history reaches the mock client on a subsequent turn.
    await page.reload(wait_until="networkidle")
    await expect(page.locator(".chat-message-user")).to_have_count(edited_index + 1)
    await send_command(page, "Follow up")
    data = await wait_saved_response(page, "Reply: Follow up")
    assert data["calls"][-1]["history"] == expected
    assert not any(item["error"] for item in data["session"]["messages"])


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
@pytest.mark.parametrize("reload", [False, True], ids=["same-page", "reconnected"])
async def test_edit_running_turn_with_mock_client_cancels_and_discards_queue(
    edit_page: Page,
    mode: str,
    reload: bool,
):
    page = edit_page
    await new_conversation(page, mode)
    await send_command(page, "Retained input")
    await wait_saved_response(page, "Reply: Retained input")
    await send_command(page, "Slow original")
    await expect(page.locator(".chat-message-streaming")).to_contain_text(
        "Reply: Slow original"
    )
    await send_command(page, "Discarded queue")
    await expect(page.locator(".exmekh-queue-item")).to_contain_text("Discarded queue")
    if reload:
        await page.reload(wait_until="networkidle")
        await expect(page.locator(".chat-message-streaming")).to_be_visible()
    await save_edit(page, "Replacement while streaming", 1)
    data = await wait_saved_response(page, "Reply: Replacement while streaming")
    assert data["calls"][-2]["cancelled"]
    assert data["calls"][-1]["history"] == ["Retained input", "Reply: Retained input"]
    assert not data["session"]["queued_messages"]
    assert len(data["session"]["messages"]) == 4
    assert not any(item["error"] for item in data["session"]["messages"])
    await expect(page.locator(".exmekh-queue-box")).not_to_be_visible()
    await expect(page.locator(".generation-stopped")).to_have_count(0)


async def test_cancel_editor_preserves_the_running_generation(edit_page: Page):
    page = edit_page
    await new_conversation(page, "quick_chat")
    await send_command(page, "Slow original")
    await expect(page.locator(".chat-message-streaming")).to_contain_text(
        "Reply: Slow original"
    )
    before = await state(page.url.rstrip("/"))
    dialog = await open_editor(page)
    await dialog.get_by_label("Message", exact=True).fill("Cancelled edit")
    await dialog.get_by_role("button", name="Cancel", exact=True).click()
    await expect(page.locator(".chat-message-streaming")).to_be_visible()
    address = urlsplit(page.url)
    await page.request.post(
        f"{address.scheme}://{address.netloc}/test/release-inference"
    )
    data = await wait_saved_response(page, "Reply: Slow original")
    assert len(data["calls"]) == len(before["calls"])
    assert not data["calls"][-1]["cancelled"]
    assert data["session"]["messages"][0]["content"] == "Slow original"


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
async def test_edit_caption_keeps_image_pdf_parts_in_mock_requests_after_reload(
    edit_page: Page,
    mode: str,
):
    page = edit_page
    await new_conversation(page, mode)
    await page.locator(".exmekh-file-picker").set_input_files(
        [
            {"name": "red.png", "mimeType": "image/png", "buffer": png_bytes()},
            {
                "name": "report.pdf",
                "mimeType": "application/pdf",
                "buffer": pdf_bytes(),
            },
        ]
    )
    await expect(page.locator(".exmekh-attachment-chip")).to_have_count(2)
    await send_command(page, "Original caption")
    await wait_turn_completion(page)
    before = await state(page.url.rstrip("/"))
    original = before["session"]["messages"][0]
    await save_edit(page, "Corrected caption")
    await wait_turn_completion(page)
    await expect(page.locator(".msg-assistant").last).to_contain_text(
        "Reply: Corrected caption"
    )
    data = await state(page.url.rstrip("/"))
    edited = data["session"]["messages"][0]
    assert edited["attachment_paths"] == original["attachment_paths"]
    assert edited["image_urls"] == original["image_urls"]
    assert len(edited["image_urls"]) == 3
    request = data["calls"][-1]
    assert request["image_count"] == 3
    assert "Original caption" not in request["prompt"]
    assert "PDF report.pdf: pages 1–2 of 2" in request["prompt"]
    await page.reload(wait_until="networkidle")
    await expect(
        page.locator(".exmekh-message-images .exmekh-image-thumbnail")
    ).to_have_count(3)
    await send_command(page, "Inspect previous attachments")
    await wait_saved_response(page, "Reply: Inspect previous attachments")
    data = await state(page.url.rstrip("/"))
    first_user = next(
        item for item in data["calls"][-1]["messages"] if item["role"] == "user"
    )
    assert [
        part["image_url"]["url"]
        for part in first_user["content"]
        if part["type"] == "image_url"
    ] == original["image_urls"]


async def test_edit_waiting_command_approval_restarts_without_executing_the_command(
    edit_page: Page,
):
    page = edit_page
    await new_conversation(page, "deep_agent")
    await page.locator(".input-bar-approval-select").first.click()
    await page.get_by_role("option", name="Manual", exact=True).click()
    await send_command(page, "Await approval")
    await expect(page.locator(".exmekh-approval-command")).to_have_text(
        "echo edit_approval"
    )
    before = await state(page.url.rstrip("/"))
    await save_edit(page, "Answer without a command")
    data = await wait_saved_response(page, "Reply: Answer without a command")
    assert len(data["calls"]) == len(before["calls"]) + 1
    assert data["calls"][-1]["history"] == []
    assert len(data["session"]["messages"]) == 2
    assert not data["session"]["settings"]["approved_commands_history"]
    await expect(page.locator(".generation-stopped")).to_have_count(0)
    await expect(page.locator(".exmekh-approval-command")).not_to_be_visible()
    await expect(page.locator(".tool-argument-value")).to_have_count(0)


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
async def test_text_upload_remains_available_after_edit_and_reload(
    edit_page: Page, mode: str
):
    page = edit_page
    await new_conversation(page, mode)
    await page.locator(".exmekh-file-picker").set_input_files(
        {
            "name": "notes.txt",
            "mimeType": "text/plain",
            "buffer": b"Content from the retained attachment",
        }
    )
    await expect(page.locator(".exmekh-attachment-chip")).to_have_count(1)
    await send_command(page, "Old text instruction")
    await wait_turn_completion(page)
    original = (await state(page.url.rstrip("/")))["session"]["messages"][0]
    dialog = await open_editor(page)
    await expect(dialog.get_by_label("Message", exact=True)).to_have_value(
        "Old text instruction"
    )
    await dialog.get_by_label("Message", exact=True).fill("New text instruction")
    await dialog.get_by_role("button", name="Save & regenerate", exact=True).click()
    await expect(dialog).not_to_be_visible()
    await wait_turn_completion(page)
    data = await state(page.url.rstrip("/"))
    edited = data["session"]["messages"][0]
    assert edited["attachment_paths"] == original["attachment_paths"]
    assert "Old text instruction" not in data["calls"][-1]["prompt"]
    if mode == "quick_chat":
        assert "Content from the retained attachment" in data["calls"][-1]["prompt"]
        assert "<user_input_file" not in data["calls"][-1]["prompt"]
    else:
        assert original["attachment_paths"][0] in data["calls"][-1]["prompt"]
    await page.reload(wait_until="networkidle")
    await expect(page.locator(".chat-message-user")).to_contain_text(
        "New text instruction"
    )
    await send_command(page, "After attachment edit")
    await wait_saved_response(page, "Reply: After attachment edit")
    data = await state(page.url.rstrip("/"))
    assert data["calls"][-1]["history"][0] == edited["content"]


@pytest.mark.parametrize("mode", ["quick_chat", "deep_agent"])
async def test_edit_replaces_attachments_and_cancel_keeps_original(
    edit_page: Page, mode: str
):
    page = edit_page
    await new_conversation(page, mode)
    await page.locator(".exmekh-file-picker").set_input_files(
        {"name": "old.png", "mimeType": "image/png", "buffer": png_bytes()}
    )
    await expect(page.locator(".exmekh-attachment-chip")).to_have_count(1)
    await send_command(page, "Original attachment")
    await wait_turn_completion(page)
    original = (await state(page.url.rstrip("/")))["session"]["messages"][0]
    dialog = await open_editor(page)
    await dialog.get_by_role("button", name="Remove", exact=False).first.click()
    await dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert (await state(page.url.rstrip("/")))["session"]["messages"][0] == original
    dialog = await open_editor(page)
    await dialog.get_by_role("button", name="Remove", exact=False).first.click()
    await dialog.locator(".message-edit-upload input[type=file]").set_input_files(
        [
            {"name": "replacement.png", "mimeType": "image/png", "buffer": png_bytes()},
            {
                "name": "replacement.txt",
                "mimeType": "text/plain",
                "buffer": b"New attachment content",
            },
        ]
    )
    await expect(dialog.locator(".message-edit-attachments")).to_contain_text(
        "replacement.txt"
    )
    await dialog.get_by_label("Message", exact=True).fill("Updated attachments")
    await page.screenshot(
        path=f"screenshots/attachment-edit-{mode}.png", full_page=True
    )
    await dialog.get_by_role("button", name="Save & regenerate").click()
    await expect(dialog).not_to_be_visible()
    await wait_turn_completion(page)
    data = await state(page.url.rstrip("/"))
    edited = data["session"]["messages"][0]
    assert len(edited["attachment_paths"]) == 2
    assert original["attachment_paths"][0] not in edited["attachment_paths"]
    assert data["calls"][-1]["image_count"] == 1
    assert "old.png" not in data["calls"][-1]["prompt"]
    if mode == "quick_chat":
        assert "New attachment content" in data["calls"][-1]["prompt"]
    await page.reload(wait_until="networkidle")
    await expect(page.locator(".chat-message-user")).to_contain_text(
        "Updated attachments"
    )
    await expect(
        page.locator(".exmekh-message-images .exmekh-image-thumbnail")
    ).to_have_count(1)
    dialog = await open_editor(page)
    for _ in range(2):
        await dialog.get_by_role("button", name="Remove", exact=False).first.click()
    await dialog.get_by_label("Message", exact=True).fill("")
    await expect(
        dialog.get_by_role("button", name="Save & regenerate")
    ).to_be_disabled()
    await dialog.get_by_label("Message", exact=True).fill("Text only")
    await dialog.get_by_role("button", name="Save & regenerate").click()
    await wait_saved_response(page, "Reply: Text only")
    data = await state(page.url.rstrip("/"))
    assert data["session"]["messages"][0]["attachment_paths"] == []
    assert data["calls"][-1]["image_count"] == 0
