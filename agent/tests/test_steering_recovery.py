"""Exercise saved steering through real ExMekh Agent turns and the installed ExMekh-Lib wheel."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pytest
from exmekh_lib.core.agents.node import DefaultCapabilities
from PIL import Image

from exmekh_agent.exmekh_bridge.agent_runner import run_agent_turn
from exmekh_agent.exmekh_bridge.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.image_input import image_data_url
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    QueuedMessage,
)
from tests.steering_probe import SteeringProbeProvider


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("image_count", [0, 2])
async def test_saved_steering_is_discovered_and_read_in_later_fe_turns(
    tmp_path: Path,
    nested: bool,
    image_count: int,
) -> None:
    provider = SteeringProbeProvider(
        first_response="nested_skill" if nested else "skill",
        block_reply=True,
        recover_steering=True,
    )
    collector = TurnStreamCollector()
    settings = ChatSettings(
        work_root=str(tmp_path),
        capabilities_path=str(DefaultCapabilities(mocks=True).capabilities_path),
        tool_approval_mode="Approve All",
    )
    guidance = QueuedMessage(
        text="Read the attached correction",
        image_urls=[
            image_data_url(Image.new("RGB", (2, 2), color))
            for color in ("red", "blue")[:image_count]
        ],
    )

    async def run(prompt: str, current: TurnStreamCollector, history=None) -> str:
        return await run_agent_turn(
            user_prompt=prompt,
            model="steering-probe",
            agent_mode="deep_agent",
            settings=settings,
            history=history,
            steering=current.steering,
            on_token=current.append_token,
            on_reasoning=lambda *_: None,
            on_tool_event=current.append_tool,
            on_skill_event=current.append_skill,
            on_raw_history=current.set_raw_history,
        )

    with patch(
        "exmekh_agent.exmekh_bridge.agent_runner.LocalProvider", return_value=provider
    ):
        task = asyncio.create_task(run("Start", collector))
        try:
            assert await asyncio.to_thread(provider.started.wait, 5)
            if nested:
                provider.release.set()
                assert await asyncio.to_thread(provider.second_started.wait, 5)
            assert collector.steering.send(guidance)
            provider.release.set()
            provider.release_second.set()
            result = await asyncio.wait_for(task, 10)
        finally:
            provider.release.set()
            provider.release_second.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        message = collector.build_message(fallback_content=result)
        header_text = message.skill_history_tree[0]["output"].split("</header>")[0]
        references = ET.fromstring(header_text + "</header>").findall("used-tools/item")
        output_id = next(
            item.findtext("id")
            for item in references
            if item.findtext("name") == "steering_message"
        )
        session = ChatSession(
            messages=[ChatMessage(role="user", content="Start"), message]
        )
        for _ in range(2):
            session = ChatSession.model_validate_json(session.model_dump_json())
            current = TurnStreamCollector()
            result = await run("Inspect saved guidance", current, session.messages)
            assert f"Recovered steering with {image_count} images" in result
            assert guidance.text in result
            request = provider.requests[-1]
            reply = next(item for item in reversed(request) if item["role"] == "tool")
            assert reply["role"] == "tool"
            assert f"<id>{output_id}</id>" in reply["content"]
            assert "<type>UserMessage</type>" in reply["content"]
            if image_count:
                companion = request[-1]
                assert companion["role"] == "user"
                urls = [
                    part["image_url"]["url"]
                    for part in companion["content"]
                    if part["type"] == "image_url"
                ]
                assert urls == guidance.image_urls
            # Neither reopening ExMekh Agent nor rereading the output changes its ID.
            session.messages.extend(
                [
                    ChatMessage(role="user", content="Inspect saved guidance"),
                    current.build_message(fallback_content=result),
                ]
            )
