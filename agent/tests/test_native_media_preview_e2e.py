from base64 import b64encode
from io import BytesIO
from pathlib import Path
from random import Random

from PIL import Image
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    open_editor,
    send_command,
    start_mock_server,
    wait_turn_completion,
)
from tests.image_samples import png_bytes


async def test_assistant_tool_image_preview_is_visible_and_persistent():
    server = start_mock_server(0, "tests.run_native_media_preview_agent")
    url = next(server)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page(reduced_motion="reduce")
                await page.goto(url, wait_until="networkidle")
                await send_command(page, "Generate a red image")
                await expect(page.locator(".generated-image-loading")).to_have_count(1)
                await wait_turn_completion(page)
                preview = page.locator(".msg-assistant .exmekh-image-thumbnail")
                await expect(preview).to_have_count(1)
                await preview.click()
                await expect(page.locator(".exmekh-image-preview")).to_be_visible()
                await page.get_by_role("button", name="Close image preview").click()
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(1)

                await page.get_by_role(
                    "button", name="Attach image to next message"
                ).click()
                chips = page.locator(".exmekh-attachment-chip")
                await expect(chips).to_have_count(1)
                await expect(
                    chips.nth(0).get_by_role("button", name="Image model")
                ).to_have_attribute("aria-pressed", "true")
                route_group = chips.nth(0).get_by_role(
                    "group", name="Image attachment destination"
                )
                assert (
                    await route_group.evaluate("node => getComputedStyle(node).display")
                    == "grid"
                )
                switch_box = await route_group.bounding_box()
                thumbnail_box = (
                    await chips.nth(0).locator(".exmekh-image-thumbnail").bounding_box()
                )
                filename_box = (
                    await chips.nth(0)
                    .get_by_text("generated-image.png", exact=True)
                    .bounding_box()
                )
                assert switch_box and thumbnail_box and filename_box
                assert switch_box["width"] <= 176 and switch_box["height"] <= 29
                assert switch_box["x"] >= thumbnail_box["x"] + thumbnail_box["width"]
                assert switch_box["y"] >= filename_box["y"] + filename_box["height"]

                # Exercise a preview larger than the Socket.IO event limit by
                # dropping a file on the message area, outside the input box.
                output = BytesIO()
                Image.frombytes(
                    "RGB", (700, 700), Random(23).randbytes(700 * 700 * 3)
                ).save(output, "PNG")
                data_transfer = await page.evaluate_handle(
                    """(encoded) => {
                    const transfer = new DataTransfer();
                    const bytes = Uint8Array.from(atob(encoded), byte => byte.charCodeAt(0));
                    transfer.items.add(new File([bytes], 'large.png', {type: 'image/png'}));
                    return transfer;
                }""",
                    b64encode(output.getvalue()).decode(),
                )
                await page.locator(
                    ".chat-scroll-container:visible"
                ).first.dispatch_event("drop", {"dataTransfer": data_transfer})
                await expect(chips).to_have_count(2)
                await expect(
                    chips.nth(1).get_by_role("button", name="LLM", exact=True)
                ).to_have_attribute("aria-pressed", "true")

                await page.locator(".exmekh-file-picker").set_input_files(
                    {
                        "name": "both.png",
                        "mimeType": "image/png",
                        "buffer": png_bytes("green"),
                    }
                )
                await expect(chips).to_have_count(3)
                attachment = chips.nth(2)
                border_colors = [
                    await attachment.evaluate(
                        "node => getComputedStyle(node).borderColor"
                    )
                ]
                image_route = attachment.get_by_role("button", name="Image model")
                await image_route.click()
                await expect(image_route).to_have_attribute("aria-pressed", "true")
                border_colors.append(
                    await attachment.evaluate(
                        "node => getComputedStyle(node).borderColor"
                    )
                )
                both_route = attachment.get_by_role("button", name="Both", exact=True)
                await both_route.click()
                await expect(both_route).to_have_attribute("aria-pressed", "true")
                border_colors.append(
                    await attachment.evaluate(
                        "node => getComputedStyle(node).borderColor"
                    )
                )
                assert len(set(border_colors)) == 3
                await send_command(
                    page,
                    "Edit the image references. Use the first image for the composition, "
                    "the second image for the subject, and the third image for the colors. "
                    "Keep the original background and make the lighting feel natural.",
                )
                await wait_turn_completion(page)
                state = await (await page.request.get(url + "/test/media-state")).json()
                message = state["session"]["messages"][-2]
                references = message["image_references"]
                assert references[0]["image_only_preview"]
                assert references[1]["image_only_preview"] is None
                assert state["edit_requests"] == [
                    [Path(ref["path"]).name for ref in references]
                ]
                assert len(message["image_urls"]) == 2
                assert state["user_image_counts"][-1] == 2
                assert references[0]["path"] not in message["content"]
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-user").last.locator(".exmekh-image-thumbnail")
                ).to_have_count(3)
                bubble = page.locator(".msg-user").last
                history = bubble.locator(".message-image-attachment")
                await expect(history).to_have_count(3)
                phone = await browser.new_page(
                    viewport={"width": 390, "height": 844}, has_touch=True
                )
                await phone.goto(page.url, wait_until="networkidle")
                for display in (page, phone):
                    message_bubble = display.locator(".msg-user").last
                    image_cards = message_bubble.locator(".message-image-attachment")
                    await expect(image_cards).to_have_count(3)
                    bubble_box = await message_bubble.bounding_box()
                    row_box = await message_bubble.locator("..").bounding_box()
                    assert bubble_box is not None and row_box is not None
                    right_edge = bubble_box["x"] + bubble_box["width"]
                    assert bubble_box["width"] <= 0.8 * row_box["width"] + 1
                    assert abs(right_edge - (row_box["x"] + row_box["width"])) <= 1
                    for card in await image_cards.all():
                        card_box = await card.bounding_box()
                        assert card_box is not None
                        assert card_box["x"] >= bubble_box["x"] - 1
                        assert card_box["x"] + card_box["width"] <= right_edge + 1, (
                            display.viewport_size,
                            bubble_box,
                            card_box,
                        )
                await phone.close()
                for filename, destination, color in zip(
                    ("large.png", "generated-image.png", "both.png"),
                    ("LLM", "Image model", "Both"),
                    border_colors,
                    strict=True,
                ):
                    card = history.filter(has_text=filename)
                    await expect(
                        card.get_by_role("button", name=destination, exact=True)
                    ).to_have_attribute("aria-pressed", "true")
                    assert (
                        await card.evaluate(
                            "node => getComputedStyle(node).borderColor"
                        )
                        == color
                    )
                generated = history.filter(has_text="generated-image.png")
                await generated.get_by_role("button", name="LLM", exact=True).click()
                await expect(
                    generated.get_by_role("button", name="Image model")
                ).to_have_attribute("aria-pressed", "true")

                dialog = await open_editor(page, 1)
                editable = dialog.locator(".message-edit-image-attachment")
                generated_edit = editable.filter(has_text="generated-image.png")
                await expect(
                    generated_edit.get_by_role("button", name="Image model")
                ).to_have_attribute("aria-pressed", "true")
                assert (
                    await generated_edit.evaluate(
                        "node => getComputedStyle(node).borderColor"
                    )
                    == border_colors[1]
                )
                llm_route = generated_edit.get_by_role("button", name="LLM", exact=True)
                await llm_route.click()
                await expect(llm_route).to_have_attribute("aria-pressed", "true")
                assert (
                    await generated_edit.evaluate(
                        "node => getComputedStyle(node).borderColor"
                    )
                    == border_colors[0]
                )
                await dialog.get_by_role("button", name="Cancel", exact=True).click()
                await expect(
                    generated.get_by_role("button", name="Image model")
                ).to_have_attribute("aria-pressed", "true")

                dialog = await open_editor(page, 1)
                await generated_edit.get_by_role(
                    "button", name="LLM", exact=True
                ).click()
                await (
                    editable.filter(has_text="both.png")
                    .get_by_role("button", name="Image model")
                    .click()
                )
                await dialog.get_by_role("button", name="Save & regenerate").click()
                await expect(dialog).not_to_be_visible()
                await wait_turn_completion(page)
                state = await (await page.request.get(url + "/test/media-state")).json()
                edited = state["session"]["messages"][-2]
                assert edited["id"] == message["id"]
                assert edited["attachment_paths"] == message["attachment_paths"]
                assert len(edited["image_references"]) == 1
                assert edited["image_references"][0]["image_only_preview"]
                assert state["edit_requests"][-1] == [Path(references[1]["path"]).name]
                assert len(edited["image_urls"]) == 2
                await page.reload(wait_until="networkidle")
                await expect(
                    generated.get_by_role("button", name="LLM", exact=True)
                ).to_have_attribute("aria-pressed", "true")
                assert (
                    await generated.evaluate(
                        "node => getComputedStyle(node).borderColor"
                    )
                    == border_colors[0]
                )
            finally:
                await browser.close()
    finally:
        server.close()
