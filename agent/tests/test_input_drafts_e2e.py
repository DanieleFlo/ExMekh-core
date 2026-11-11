from __future__ import annotations

import asyncio
import json

from playwright.async_api import async_playwright, expect

from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    wait_turn_completion,
)


async def test_f4_inserts_basito_into_the_chat_draft(tmp_path):
    storage = tmp_path / "state"
    sessions = SessionManager(storage / "chats")
    session = sessions.get_current_session()
    path = ChatRoutes(sessions).path_for(session)
    draft_path = storage / "drafts" / f"{session.id}.json"

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            with BrowserTestServer(storage, module="tests.run_projects_agent") as url:
                page = await browser.new_page()
                await page.goto(url + path, wait_until="networkidle")
                textarea = page.locator(".chat-input-textarea textarea")
                await textarea.fill("Sono sorpreso oggi")
                await textarea.evaluate("element => element.setSelectionRange(5, 13)")
                await textarea.press("F4")
                await expect(textarea).to_have_value("Sono basito oggi")
                assert await textarea.evaluate(
                    "element => [element.selectionStart, element.selectionEnd]"
                ) == [11, 11]
                await textarea.press("Shift+Enter")
                await expect(textarea).to_have_value("Sono basito\n oggi")
                await textarea.press("Backspace")

                await textarea.press("Shift+F4")
                await textarea.blur()
                await page.keyboard.press("F4")
                await expect(textarea).to_have_value("Sono basito oggi")
                await expect(page.locator(".msg-user")).to_have_count(0)

                async with asyncio.timeout(5):
                    while not draft_path.exists() or (
                        json.loads(draft_path.read_text())["text"] != "Sono basito oggi"
                    ):
                        await asyncio.sleep(0.05)

                await page.reload(wait_until="networkidle")
                await expect(textarea).to_be_visible()
                await expect(textarea).to_have_value("Sono basito oggi")
                await textarea.press("Enter")
                await wait_turn_completion(page)
                await expect(page.locator(".msg-user")).to_contain_text(
                    "Sono basito oggi"
                )
        finally:
            await browser.close()


async def test_unsent_text_survives_closed_windows_and_application_restart(tmp_path):
    storage = tmp_path / "state"
    sessions = SessionManager(storage / "chats")
    quick = sessions.get_current_session()
    quick.title = "Quick chat draft"
    sessions.save_session(quick)
    folder = tmp_path / "Draft project"
    folder.mkdir()
    project = sessions.create_project(folder)
    project_chat = sessions.create_session("Project chat draft", project.id)
    quick_path = ChatRoutes(sessions).path_for(quick)
    project_path = ChatRoutes(sessions).path_for(project_chat)
    quick_text = "  Testo da ritrovare ☕\nSeconda riga\n  "
    project_text = "Bozza separata per il progetto"

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            with BrowserTestServer(storage, module="tests.run_projects_agent") as url:
                page = await browser.new_page()
                await page.goto(url + quick_path, wait_until="networkidle")
                textarea = page.locator(".chat-input-textarea textarea")
                await textarea.fill(quick_text)
                await page.close()

                page = await browser.new_page()
                await page.goto(url + quick_path, wait_until="networkidle")
                textarea = page.locator(".chat-input-textarea textarea")
                await expect(textarea).to_have_value(quick_text)
                await expect(page.locator(".msg-user")).to_have_count(0)
                await page.goto(url + project_path, wait_until="networkidle")
                await expect(textarea).to_have_value("")
                await textarea.fill(project_text)
                await page.locator(".sidebar-item").filter(has_text=quick.title).click()
                await expect(textarea).to_have_value(quick_text)
                await page.close()

            # A new process and browser context can restore both conversations.
            with BrowserTestServer(storage, module="tests.run_projects_agent") as url:
                page = await browser.new_page()
                await page.goto(url + project_path, wait_until="networkidle")
                textarea = page.locator(".chat-input-textarea textarea")
                await expect(textarea).to_have_value(project_text)
                await textarea.fill("")
                await page.reload(wait_until="networkidle")
                await expect(textarea).to_have_value("")
                await page.locator(".sidebar-item").filter(has_text=quick.title).click()
                await expect(textarea).to_have_value(quick_text)
                await page.locator(".chat-navigation-overlay").wait_for(state="hidden")
                await textarea.press("Enter")
                await wait_turn_completion(page)
                await expect(textarea).to_have_value("")
                await expect(page.locator(".msg-user")).to_contain_text(
                    quick_text.strip()
                )
                await page.reload(wait_until="networkidle")
                await expect(textarea).to_have_value("")
                saved = json.loads((storage / "chats" / f"{quick.id}.json").read_text())
                assert saved["messages"][0]["content"] == quick_text.strip()
                assert not (storage / "drafts" / f"{quick.id}.json").exists()
                assert not (storage / "drafts" / f"{project_chat.id}.json").exists()
                await page.close()
        finally:
            await browser.close()


async def test_switching_back_to_a_chat_loads_another_windows_newer_draft(tmp_path):
    storage = tmp_path / "state"
    sessions = SessionManager(storage / "chats")
    first = sessions.get_current_session()
    first.title = "Shared draft chat"
    sessions.save_session(first)
    other = sessions.create_session("Other draft chat")
    path = ChatRoutes(sessions).path_for(first)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            with BrowserTestServer(storage, module="tests.run_projects_agent") as url:
                left = await browser.new_page()
                right = await browser.new_page()
                await left.goto(url + path, wait_until="networkidle")
                await right.goto(url + path, wait_until="networkidle")
                left_text = left.locator(".chat-input-textarea textarea")
                right_text = right.locator(".chat-input-textarea textarea")
                await left_text.fill("An older draft in this window")
                await left.locator(".sidebar-item").filter(has_text=other.title).click()
                await expect(left_text).to_have_value("")
                await right_text.fill("The newer draft from another window")
                await (
                    right.locator(".sidebar-item").filter(has_text=other.title).click()
                )
                await expect(right_text).to_have_value("")
                await left.locator(".sidebar-item").filter(has_text=first.title).click()
                await expect(left_text).to_have_value(
                    "The newer draft from another window"
                )
                await expect(left).to_have_url(url + path)
                await left.reload(wait_until="networkidle")
                await expect(left_text).to_have_value(
                    "The newer draft from another window"
                )
        finally:
            await browser.close()
