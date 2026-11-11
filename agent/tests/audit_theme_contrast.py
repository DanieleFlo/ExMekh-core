"""Audit the light theme in Chromium and save per-view evidence and WCAG results."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from urllib.parse import urljoin

from playwright.async_api import Error, Locator, Page, async_playwright, expect

from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.state.session_manager import SessionManager
from tests.browser_support import BrowserTestServer, get_edge_executable

PROBE = Path(__file__).with_name("contrast_probe.js").read_text(encoding="utf-8")
STABLE_STYLES = "*, *::before, *::after { transition-duration: 0s !important; animation-duration: 0s !important; }"


async def capture(
    page: Page, output: Path, name: str, scope: Locator | None = None
) -> dict:
    await page.mouse.move(0, 0)
    await page.evaluate("document.fonts.ready.then(() => true)")
    screenshot = f"{name}.png"
    await page.screenshot(path=output / screenshot, animations="disabled")
    result = await (scope if scope is not None else page.locator("body")).evaluate(
        PROBE
    )
    statuses = {item["status"] for item in result["findings"]}
    if not result["checked"] and not statuses:
        result["findings"].append(
            {"status": "WARNING", "reason": "No contrast pairs measured"}
        )
        statuses.add("WARNING")
    status = next(
        (item for item in ("NOT PASS", "WARNING") if item in statuses), "PASS"
    )
    print(f"{status:8} {name} ({result['checked']} measured)", flush=True)
    return {
        "view": name,
        "status": status,
        "url": page.url,
        "screenshot": screenshot,
        **result,
    }


async def capture_chat(page: Page, output: Path, prefix: str) -> AsyncIterator[dict]:
    await page.add_style_tag(content=STABLE_STYLES)
    for tab, label in (
        ("Chat", "chat"),
        ("Raw History", "history"),
        ("Graph", "graph"),
    ):
        control = page.get_by_role("tab", name=tab, exact=True)
        await control.click()
        await expect(control).to_have_attribute("aria-selected", "true")
        yield await capture(page, output, f"{prefix}-{label}")

    node = page.locator(".graph-node:visible").first
    if await node.is_visible():
        await node.click()
        panel = page.locator(".graph-bottom-menu-open")
        await expect(panel).to_be_visible()
        yield await capture(page, output, f"{prefix}-graph-detail", panel)

    await page.get_by_role("tab", name="Chat", exact=True).click()
    approval = page.locator(".input-bar-approval-select")
    if await approval.is_visible():
        await approval.click()
        menu = page.locator(".q-menu:visible").last
        await expect(menu).to_be_visible()
        yield await capture(page, output, f"{prefix}-approval", menu)
        await page.keyboard.press("Escape")


async def capture_warning(
    page: Page, output: Path, name: str, url: str, error: Exception
) -> dict:
    screenshot = f"{name}.png"
    try:
        await page.screenshot(path=output / screenshot, timeout=5000)
    except Error:
        screenshot = None
    print(f"WARNING  {name} (view could not be fully audited)", flush=True)
    return {
        "view": name,
        "status": "WARNING",
        "url": url,
        "screenshot": screenshot,
        "checked": 0,
        "findings": [{"status": "WARNING", "reason": str(error).splitlines()[0]}],
    }


async def audit(url: str, output: Path, viewport: dict[str, int]) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable()
        )
        try:
            page = await browser.new_page(viewport=viewport, reduced_motion="reduce")
            await page.goto(url, wait_until="networkidle")
            await page.add_style_tag(content=STABLE_STYLES)
            switch = page.get_by_role("switch", name="Light mode")
            if await switch.get_attribute("aria-checked") == "false":
                await switch.click()
            await expect(switch).to_have_attribute("aria-checked", "true")
            await expect(page.locator("[data-chat-path]").first).to_be_attached()
            paths = await page.locator("[data-chat-path]").evaluate_all(
                "rows => [...new Set(rows.map(row => row.dataset.chatPath))]"
            )
            reports = []
            await page.locator('.lm-header button:has(i:text-is("tune"))').click()
            settings = page.locator(".system-settings")
            await expect(settings).to_be_visible()
            reports.append(await capture(page, output, "settings", settings))
            for index, path in enumerate(paths, start=1):
                destination = urljoin(url, path)
                try:
                    await page.goto(destination, wait_until="networkidle")
                    await expect(switch).to_have_attribute("aria-checked", "true")
                    async for report in capture_chat(page, output, f"{index:03d}"):
                        reports.append(report)
                except (Error, AssertionError) as error:
                    reports.append(
                        await capture_warning(
                            page, output, f"{index:03d}-incomplete", destination, error
                        )
                    )

            await page.goto(url, wait_until="networkidle")
            await page.add_style_tag(content=STABLE_STYLES)

            sidebar = page.locator(".lm-sidebar")
            if await sidebar.evaluate(
                "element => element.getBoundingClientRect().right <= 0"
            ):
                await page.get_by_role("button", name="Open projects menu").click()
                await page.wait_for_function(
                    "document.querySelector('.lm-sidebar').getBoundingClientRect().right > 0"
                )
                reports.append(await capture(page, output, "sidebar", sidebar))
            await page.get_by_role("button", name="New project", exact=True).click()
            dialog = page.get_by_role("dialog").last
            await expect(dialog).to_be_visible()
            reports.append(await capture(page, output, "new-project", dialog))
            await dialog.get_by_role(
                "button", name="Project settings", exact=True
            ).click()
            dialog = page.get_by_role("dialog").last
            await expect(
                dialog.get_by_role("button", name="Save project settings")
            ).to_be_visible()
            reports.append(await capture(page, output, "project-settings", dialog))
            return reports
        finally:
            await browser.close()


def seed_demo(storage: Path) -> None:
    sessions = SessionManager(storage / "chats")
    sessions.update_title(sessions.get_current_session(), "Empty Quick Chat")
    other_workspace = storage / "Second project"
    other_workspace.mkdir()
    other_project = sessions.create_project(other_workspace)
    sessions.create_session("Empty project chat", project_id=other_project.id)
    workspace = storage / "Contrast project"
    workspace.mkdir()
    project = sessions.create_project(workspace)
    chat = sessions.create_session("Example conversation", project_id=project.id)
    sessions.add_message(chat, ChatMessage(role="user", content="Show the example."))
    sessions.add_message(
        chat,
        ChatMessage(
            role="assistant",
            content="## Example output\n\n[Documentation](https://example.com)\n\n"
            "```python\n# Example\nfor index in range(3):\n    print('hello', index)\n```\n\n"
            "| Name | Status |\n| --- | --- |\n| Example | Ready |",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url", help="Running agent frontend. Omit to use isolated demo chats."
    )
    source.add_argument(
        "--data-dir",
        type=Path,
        help="Audit an isolated copy of this agent data directory.",
    )
    parser.add_argument("--output", type=Path, default=Path(".test-tmp/contrast-audit"))
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()
    viewport = {"width": args.width, "height": args.height}
    if args.url:
        reports = asyncio.run(audit(args.url, args.output, viewport))
    else:
        with TemporaryDirectory(prefix="exmekh-contrast-") as directory:
            storage = Path(directory)
            if args.data_dir is None:
                seed_demo(storage)
            else:
                if not (args.data_dir / "chats").is_dir():
                    parser.error("--data-dir must contain a chats directory")
                for name in ("chats", "projects"):
                    if (args.data_dir / name).is_dir():
                        copytree(args.data_dir / name, storage / name)
            with BrowserTestServer(
                storage, module="tests.run_projects_agent", readiness_path="/test/ready"
            ) as url:
                entry_url = url
                if args.data_dir is not None:
                    records = list((storage / "chats").glob("*.json"))
                    if records:
                        smallest = min(records, key=lambda path: path.stat().st_size)
                        entry_url += f"/chat/contrast/start--{smallest.stem}"
                reports = asyncio.run(audit(entry_url, args.output, viewport))

    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_path.resolve()}")
    statuses = {report["status"] for report in reports}
    return 1 if "NOT PASS" in statuses else 2 if "WARNING" in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
