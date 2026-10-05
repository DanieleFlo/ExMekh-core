from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.models import ChatMessage, SkillNode, ToolEvent
from exmekh_agent.exmekh_bridge.providers.context_usage import ContextUsageSummary
from exmekh_agent.main import index_page, main, run
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.chat.page_controller import ChatPageController


@pytest.fixture
def app_client():
    c = Client(page("/test-main"))
    with c:
        yield c


@pytest.mark.asyncio
async def test_index_page_lifecycle_and_callbacks(app_client: Client) -> None:
    with app_client:
        # Patch run_agent_turn to simulate streaming callbacks and return
        async def fake_run_agent(
            user_prompt,
            model,
            on_token,
            on_reasoning,
            on_tool_event,
            agent_mode,
            on_skill_event,
            on_raw_history=None,
            history=None,
            agent_context_id=None,
            settings=None,
            on_context=None,
            **kwargs,
        ):
            on_token("Hello ")
            on_token("from mock agent!")
            on_reasoning("Reasoning 1 ", skill_path="/", depth=0)
            on_reasoning("Reasoning 2", skill_path="/skill", depth=1)
            on_tool_event(ToolEvent(tool_name="mock_tool", skill_path="/", depth=0))
            node = SkillNode(name="mock-skill", path="/mock-skill", depth=1)
            on_skill_event(node, "opened")
            return "Final output text"

        with patch(
            "exmekh_agent.ui.chat.page_controller.run_agent_turn",
            side_effect=fake_run_agent,
        ):
            # 1. Build index page
            index_page()

            # Retrieve created components from client descendants or context
            curr = session_manager.get_current_session()
            curr.messages.clear()

            # Find the input bar on_send callback or trigger handle_send directly
            # We can capture closures by patching HeaderComponent / InputBarComponent
            # Or by running index_page with captured handlers


@pytest.mark.asyncio
async def test_index_page_full_flow(app_client: Client) -> None:
    handlers = {}

    with (
        patch("exmekh_agent.ui.chat.page_controller.HeaderComponent") as mock_hdr_cls,
        patch(
            "exmekh_agent.ui.chat.page_controller.SidebarComponent"
        ) as mock_sidebar_cls,
        patch(
            "exmekh_agent.ui.chat.page_controller.InputBarComponent"
        ) as mock_input_cls,
        patch("exmekh_agent.ui.chat.page_controller.ChatViewComponent") as mock_cv_cls,
        patch(
            "exmekh_agent.ui.chat.page_controller.SettingsDrawerComponent"
        ) as mock_sd_cls,
    ):

        def capture_hdr(**kwargs):
            handlers["toggle_settings"] = kwargs["on_toggle_drawer"]
            m = MagicMock()
            m.model_select = MagicMock()
            m.mode_toggle = MagicMock()
            return m

        def capture_sidebar(**kwargs):
            handlers["session_switched"] = kwargs["on_session_switched"]
            handlers["new_chat"] = kwargs["on_new_chat"]
            return MagicMock()

        def capture_input(**kwargs):
            handlers["send"] = kwargs["on_send"]
            handlers["stop"] = kwargs["on_stop"]
            return MagicMock()

        def capture_chat(**kwargs):
            handlers["suggested_prompt"] = kwargs["on_suggested_prompt"]
            return MagicMock()

        mock_hdr_cls.side_effect = capture_hdr
        mock_sidebar_cls.side_effect = capture_sidebar
        mock_input_cls.side_effect = capture_input
        mock_cv_cls.side_effect = capture_chat
        mock_sd_cls.return_value = MagicMock()

        with app_client:
            index_page()

            # 1. Toggle settings
            handlers["toggle_settings"]()

            # 4. Suggested prompt
            handlers["suggested_prompt"]("Prompt suggestion")

            # 5. Session switched
            curr_session = session_manager.get_current_session()
            handlers["session_switched"](curr_session.id)

            # 6. New chat
            handlers["new_chat"]()

            # 7. Stop with no active task (graceful)
            handlers["stop"]()

            # 8. Send message - Success path
            async def fake_run_turn_success(
                user_prompt,
                model,
                on_token,
                on_reasoning,
                on_tool_event,
                agent_mode,
                on_skill_event,
                on_raw_history=None,
                history=None,
                agent_context_id=None,
                on_sub_agent_event=None,
                sub_agent_history=None,
                settings=None,
                on_context=None,
                **kwargs,
            ):
                on_token("Token chunk")
                on_reasoning("Reasoning step", skill_path="/", depth=0)
                on_tool_event(ToolEvent(tool_name="cmd", skill_path="/", depth=0))
                node = SkillNode(name="s1", path="/s1", depth=1)
                on_skill_event(node, "opened")
                return "Turn completed"

            with (
                patch(
                    "exmekh_agent.ui.chat.page_controller.run_agent_turn",
                    side_effect=fake_run_turn_success,
                ),
                patch(
                    "exmekh_agent.ui.chat.page_controller.generate_chat_title",
                    new_callable=AsyncMock,
                    return_value="",
                ),
            ):
                # Clear messages in session to test auto-title generation (<30 chars and >30 chars)
                s = session_manager.get_current_session()
                s.messages.clear()

                # Send prompt > 30 chars
                long_prompt = "A" * 35
                handlers["send"](long_prompt)
                for _ in range(50):
                    if len(s.messages) == 2:
                        break
                    await asyncio.sleep(0.02)
                assert len(s.title) <= 34
                assert "..." in s.title
                assert len(s.messages) == 2
                assert s.messages[-1].content == "Token chunk"
                assert s.messages[-1].error is None

            # 9. Send message - Exception error handling path
            async def fake_run_turn_error(*args, **kwargs):
                raise ConnectionError("Provider server offline")

            with patch(
                "exmekh_agent.ui.chat.page_controller.run_agent_turn",
                side_effect=fake_run_turn_error,
            ):
                handlers["send"]("Error prompt")
                for _ in range(50):
                    if len(s.messages) == 4:
                        break
                    await asyncio.sleep(0.02)
                last_msg = s.messages[-1]
                assert "⚠️" in last_msg.content
                assert last_msg.error == "Provider server offline"

            # 10. Stop generation with active task
            async def slow_turn(*args, **kwargs):
                await asyncio.sleep(10)
                return "slow"

            with patch(
                "exmekh_agent.ui.chat.page_controller.run_agent_turn",
                side_effect=slow_turn,
            ) as slow_mock:
                handlers["send"]("Slow prompt")
                await asyncio.sleep(0.05)
                handlers["send"]("Must be rejected while busy")
                await asyncio.sleep(0.01)
                assert slow_mock.call_count == 1
                # Call stop while active_task is running
                handlers["stop"]()
                await asyncio.sleep(0.05)


def test_main_and_run_entrypoints() -> None:
    with patch("exmekh_agent.main.ui.run") as mock_ui_run:
        run()
        mock_ui_run.assert_called_once()

    with patch("exmekh_agent.main.run") as mock_run:
        main()
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_page_controller_awaits_model_refresh(app_client: Client) -> None:
    controller = ChatPageController(app_client)
    controller.settings_drawer = MagicMock()
    controller.settings_drawer.refresh_models = AsyncMock()

    await controller.refresh_models()

    controller.settings_drawer.refresh_models.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_context_usage_is_counted_off_the_ui_thread(app_client: Client) -> None:
    controller = ChatPageController(app_client)
    controller.input_bar = MagicMock()
    session_manager.create_session()
    ui_thread = threading.get_ident()
    calculation_threads: list[int] = []

    def count_context(**kwargs):
        calculation_threads.append(threading.get_ident())
        return ContextUsageSummary("test-model", 12, 100, estimated=True)

    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage",
        side_effect=count_context,
    ):
        controller.update_context_usage()
        await controller._context_updates.wait_idle()

    assert calculation_threads and calculation_threads[0] != ui_thread
    controller.input_bar.update_context_info.assert_called_once_with(
        model_name="test-model", used_tokens=12, context_length=100, estimated=True
    )


@pytest.mark.asyncio
async def test_stream_view_updates_run_on_client_event_loop(app_client: Client) -> None:
    controller = ChatPageController(app_client)
    controller.chat_view = MagicMock()
    controller.chat_view.is_streaming_session.return_value = True
    controller.sidebar = MagicMock()
    controller.input_bar = MagicMock()
    session = session_manager.create_session()
    session.messages.append(ChatMessage(role="user", content="previous turn"))
    main_thread = threading.get_ident()
    update_threads: list[int] = []
    update_received = asyncio.Event()

    def record_stream_token(_: str) -> None:
        update_threads.append(threading.get_ident())
        update_received.set()

    controller.chat_view.stream_token.side_effect = record_stream_token

    async def run_in_worker(**kwargs):
        def produce_token() -> None:
            kwargs["on_token"]("worker token")

        await asyncio.to_thread(produce_token)
        return "worker token"

    with patch(
        "exmekh_agent.ui.chat.page_controller.run_agent_turn", side_effect=run_in_worker
    ):
        await controller.send("test")
        await asyncio.wait_for(update_received.wait(), timeout=3)
        async with asyncio.timeout(3):
            while len(session.messages) < 3:
                await asyncio.sleep(0.01)

    assert update_threads == [main_thread]
    assert session.messages[-1].content == "worker token"
