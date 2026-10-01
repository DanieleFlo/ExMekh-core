from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nicegui.client import Client as NiceGUIClient
from nicegui.page import page

import fe.qs_bridge.agent_runner as ar
from fe.qs_bridge.agent_runner import generate_chat_title, run_agent_turn
from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.models import ChatMessage, SkillNode, ToolEvent
from fe.state.session_manager import SessionManager, session_manager
from fe.ui.chat_view import ChatViewComponent
from fe.ui.header import HeaderComponent
from fe.ui.input_bar import InputBarComponent
from fe.ui.raw_history_view import build_raw_history_from_session
from fe.ui.sidebar import SidebarComponent
from tests.mock_client import (
    MockProvider,
    PipelineStep,
    ScenarioPipeline,
    create_default_hierarchical_pipeline,
    create_quick_chat_pipeline,
    install_mock_provider,
    uninstall_mock_provider,
)

pytestmark = [pytest.mark.live, pytest.mark.mock, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def live_mock_environment():
    """Installs MockProvider into agent_runner for all live mock tests and cleans up afterwards."""
    provider = MockProvider(pipeline=create_default_hierarchical_pipeline())
    install_mock_provider(provider)
    yield provider
    uninstall_mock_provider()


@pytest.fixture
def page_context():
    """Fixture providing a NiceGUI Client context for UI component tests."""
    c = NiceGUIClient(page("/test-live-mock"))
    with c:
        yield c


def safe_print(msg: str) -> None:
    """Safe print helper preventing Windows UnicodeEncodeError."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# =============================================================================
# 1. LIVE MOCK HEALTH CHECK & DISCOVERY
# =============================================================================


async def test_live_mock_health_check() -> None:
    """Verifies provider health check and model discovery using the MockProvider."""
    is_alive, models = await ar.check_provider_health()
    assert is_alive is True, "Expected MockProvider health check to return True"
    assert len(models) >= 3, f"Expected at least 3 mock models, got: {models}"
    assert "mock-pipeline-model" in models
    assert "mock-deep-agent" in models
    assert "mock-quick-chat" in models
    safe_print(f"\n[LIVE MOCK] Health check OK. Models available: {models}")


# =============================================================================
# 2. LIVE MOCK QUICK CHAT STREAMING
# =============================================================================


async def test_live_mock_quick_chat_streaming(
    live_mock_environment: MockProvider,
) -> None:
    """Tests live streaming of tokens and raw history in Quick Chat mode with MockProvider."""
    live_mock_environment.pipeline = create_quick_chat_pipeline()

    streamed_tokens: list[str] = []
    reasoning_tokens: list[str] = []
    tool_events: list[ToolEvent] = []
    raw_history_received: list[dict[str, Any]] = []

    def on_raw(hist: list[dict[str, Any]], tree: list[dict[str, Any]]) -> None:
        raw_history_received.extend(hist)

    response_text = await run_agent_turn(
        user_prompt="Ciao! Come funzioni?",
        model="mock-quick-chat",
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda r, p="/", d=0: reasoning_tokens.append(r),
        on_tool_event=lambda ev: tool_events.append(ev),
        agent_mode="quick_chat",
        on_raw_history=on_raw,
    )

    full_response = "".join(streamed_tokens)
    assert response_text.strip(), "Expected non-empty response from mock provider"
    assert "Quick Chat" in full_response or "Quick Chat" in response_text
    assert len(streamed_tokens) > 0, "Expected token-by-token streaming"

    # Verify raw history captured from mock turn
    assert len(raw_history_received) >= 2
    assert raw_history_received[0]["role"] == "system"
    assert raw_history_received[1]["role"] == "user"
    assert raw_history_received[1]["content"] == "Ciao! Come funzioni?"
    safe_print(
        f"\n[LIVE MOCK Quick Chat]: Streamed {len(streamed_tokens)} tokens: {response_text}"
    )


# =============================================================================
# 3. LIVE MOCK DEEP AGENT: REASONING & TOOL EXECUTION
# =============================================================================


async def test_live_mock_deep_agent_reasoning_and_tools() -> None:
    """Tests live DeepAgent execution with reasoning chunks, tool calling, and token generation."""
    streamed_tokens: list[str] = []
    reasoning_chunks: list[tuple[str, str, int]] = []
    tool_events: list[ToolEvent] = []
    skill_events: list[tuple[str, str]] = []

    # Prompt matches Step 1 in create_default_hierarchical_pipeline
    prompt = "Inizializza pipeline ed esegui diagnostica preliminare"

    response_text = await run_agent_turn(
        user_prompt=prompt,
        model="mock-deep-agent",
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda c, p="/", d=0: reasoning_chunks.append((c, p, d)),
        on_tool_event=lambda ev: tool_events.append(ev),
        agent_mode="deep_agent",
        on_skill_event=lambda node, act: skill_events.append((node.path, act)),
    )

    # 1. Reasoning chunks streamed at root level
    assert len(reasoning_chunks) >= 2, (
        f"Expected at least 2 reasoning chunks, got: {len(reasoning_chunks)}"
    )
    assert all(path == "/" and depth == 0 for _, path, depth in reasoning_chunks)
    assert any("SYS_DIAG_OK" in chunk for chunk, _, _ in reasoning_chunks)

    # 2. Tool executed
    assert len(tool_events) == 1
    assert tool_events[0].tool_name == "mock-tool"
    assert tool_events[0].skill_path == "/"

    # 3. Final response token streaming
    full_response = "".join(streamed_tokens) or response_text
    assert "Diagnostica di sistema completata" in full_response
    assert "SYS_DIAG_OK" in full_response
    safe_print(
        f"\n[LIVE MOCK DeepAgent Step 1]: Reasoning: {len(reasoning_chunks)} blocks, Tools: {len(tool_events)}, Response: {full_response[:80]}..."
    )


# =============================================================================
# 4. LIVE MOCK HIERARCHICAL SUB-SKILLS (MATRYOSHKA)
# =============================================================================


async def test_live_mock_hierarchical_sub_skills_matryoshka() -> None:
    """Tests live multi-depth nested skills execution: mock-skill (depth 1) -> nested-sub-skill (depth 2)."""
    streamed_tokens: list[str] = []
    reasoning_chunks: list[tuple[str, str, int]] = []
    tool_events: list[ToolEvent] = []
    skill_events: list[tuple[str, str]] = []
    captured_raw: list[dict[str, Any]] = []
    captured_tree: list[dict[str, Any]] = []

    def on_raw(hist: list[dict[str, Any]], tree: list[dict[str, Any]]) -> None:
        captured_raw.extend(hist)
        captured_tree.extend(tree)

    # Advance pipeline to step 2 by calling prompt that matches step 2
    prompt = "Approfondisci con la sotto-skill i dati diagnostici"

    await run_agent_turn(
        user_prompt=prompt,
        model="mock-deep-agent",
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda c, p="/", d=0: reasoning_chunks.append((c, p, d)),
        on_tool_event=lambda ev: tool_events.append(ev),
        on_skill_event=lambda node, act: skill_events.append((node.path, act)),
        agent_mode="deep_agent",
        on_raw_history=on_raw,
    )

    # 1. Multi-depth reasoning verified
    depths = [d for _, _, d in reasoning_chunks]
    assert 0 in depths, "Expected root depth 0 reasoning"
    assert 1 in depths, "Expected depth 1 reasoning inside mock-skill"
    assert 2 in depths, "Expected depth 2 reasoning inside nested-sub-skill"

    # 2. Skill lifecycle events verified
    skill_paths = [path for path, _ in skill_events]
    assert "/mock-skill" in skill_paths
    assert "/mock-skill/nested-sub-skill" in skill_paths

    # 3. Tool events at multiple depths verified
    tool_depths = [ev.depth for ev in tool_events]
    assert 1 in tool_depths
    assert 2 in tool_depths

    # 4. Hierarchical Matryoshka tree verified
    assert len(captured_tree) >= 1
    root_sk = captured_tree[0]
    assert root_sk["skill"] == "mock-skill"
    assert len(root_sk["children"]) >= 1
    assert root_sk["children"][0]["skill"] == "nested-sub-skill"

    safe_print(
        f"\n[LIVE MOCK Matrioska Step 2]: Multi-depth skills verified (depths: {depths})"
    )


# =============================================================================
# 5. LIVE MOCK RAW HISTORY PROGRESSIVE STREAMING ("MAN MANO")
# =============================================================================


async def test_live_mock_raw_history_live_streaming_updates(
    page_context: NiceGUIClient,
) -> None:
    """Verifies that ChatViewComponent and RawHistoryComponent update progressively during mock live streaming."""
    with page_context:
        session = session_manager.get_current_session()
        session.messages.clear()
        session.messages.append(ChatMessage(role="user", content="Calcola la derivata"))

        chat_view = ChatViewComponent(on_suggested_prompt=lambda _: None)
        chat_view.build()

        # 1. Start live streaming turn
        chat_view.start_assistant_stream()
        assert chat_view.raw_history_view.is_streaming is True
        assert chat_view.raw_history_view.live_container is not None
        assert chat_view.raw_history_view.live_assistant_card is not None

        # 2. Stream reasoning chunk
        chat_view.stream_reasoning(
            "Analisi del gradiente in corso...", skill_path="/", depth=0
        )
        assert (
            chat_view.raw_history_view.live_reasoning_text
            == "Analisi del gradiente in corso..."
        )

        # 3. Stream tool event live
        tool_ev = ToolEvent(tool_name="mock-tool", depth=0, skill_path="/")
        chat_view.add_tool_event(tool_ev)
        assert len(chat_view.raw_history_view.live_tool_events) == 1

        # 4. Stream sub-skill live
        skill_node = SkillNode(
            name="calc-skill", path="/calc-skill", depth=1, status="active"
        )
        chat_view.handle_live_skill_event(skill_node, "opened")
        assert "/calc-skill" in chat_view.raw_history_view.live_skill_nodes

        # 5. Stream content tokens live
        chat_view.stream_token("Risultato: ")
        chat_view.stream_token("f'(x) = 2x")
        assert chat_view.raw_history_view.live_response_text == "Risultato: f'(x) = 2x"

        # 6. Finalize stream
        chat_view.finalize_stream()
        assert chat_view.raw_history_view.is_streaming is False
        assert chat_view.raw_history_view.live_status_badge.visible is False

        safe_print(
            "\n[LIVE MOCK Raw History Streaming]: Progressive streaming updates verified cleanly."
        )


# =============================================================================
# 6. LIVE MOCK STRUCTURED TITLE GENERATION
# =============================================================================


async def test_live_mock_structured_title_generation() -> None:
    """Verifies that generate_chat_title works asynchronously with MockProvider."""
    prompt = "Spiegami il funzionamento dei tensori in PyTorch"
    title = await generate_chat_title(user_prompt=prompt, model="mock-pipeline-model")
    assert title.strip(), "Expected non-empty title"
    assert title != "New Conversation"
    assert "tensori" in title.lower() or "pytorch" in title.lower() or len(title) <= 35
    safe_print(
        f"\n[LIVE MOCK Title Generation]: Generated title '{title}' for prompt '{prompt}'"
    )


# =============================================================================
# 7. LIVE MOCK MULTI-TURN SESSION PERSISTENCE
# =============================================================================


async def test_live_mock_multi_turn_session_persistence(tmp_path: Path) -> None:
    """Verifies multi-turn conversation with MockProvider saved to disk and reloaded without duplicates."""
    sm = SessionManager()
    sm.storage_dir = tmp_path
    sm.sessions.clear()

    session = sm.create_session(title="Multi-Turn Live Mock Test")

    # Turn 1: User asks initialization
    prompt_1 = "Inizializza pipeline di diagnostica"
    session.messages.append(ChatMessage(role="user", content=prompt_1))
    sm.save_session(session)

    coll_1 = TurnStreamCollector()
    res_1 = await run_agent_turn(
        user_prompt=prompt_1,
        model="mock-deep-agent",
        on_token=coll_1.append_token,
        on_reasoning=lambda c, p="/", d=0: coll_1.append_reasoning(
            c, skill_path=p, depth=d
        ),
        on_tool_event=coll_1.append_tool,
        on_raw_history=coll_1.set_raw_history,
        agent_mode="deep_agent",
    )
    msg_1 = coll_1.build_message(role="assistant", fallback_content=res_1)
    session.messages.append(msg_1)
    sm.save_session(session)

    # Turn 2: User asks for sub-skill drill-down
    prompt_2 = "Approfondisci con la sotto-skill i dati"
    session.messages.append(ChatMessage(role="user", content=prompt_2))
    sm.save_session(session)

    coll_2 = TurnStreamCollector()
    res_2 = await run_agent_turn(
        user_prompt=prompt_2,
        model="mock-deep-agent",
        on_token=coll_2.append_token,
        on_reasoning=lambda c, p="/", d=0: coll_2.append_reasoning(
            c, skill_path=p, depth=d
        ),
        on_tool_event=coll_2.append_tool,
        on_raw_history=coll_2.set_raw_history,
        agent_mode="deep_agent",
    )
    msg_2 = coll_2.build_message(role="assistant", fallback_content=res_2)
    session.messages.append(msg_2)
    sm.save_session(session)

    # Reload session from disk
    new_sm = SessionManager()
    new_sm.storage_dir = tmp_path
    new_sm.load_all()

    loaded = new_sm.sessions.get(session.id)
    assert loaded is not None, "Failed to reload session"
    assert len(loaded.messages) == 4, (
        f"Expected 4 messages (2 turns), got {len(loaded.messages)}"
    )
    assert loaded.messages[0].role == "user"
    assert loaded.messages[1].role == "assistant"
    assert loaded.messages[2].role == "user"
    assert loaded.messages[3].role == "assistant"

    # Verify build_raw_history_from_session has system prompt first and no duplicate user messages
    raw_msgs, skill_tree = build_raw_history_from_session(loaded)
    assert raw_msgs[0]["role"] == "system"
    user_msgs = [m for m in raw_msgs if m.get("role") == "user"]
    assert len(user_msgs) == 2, (
        f"Expected exactly 2 user messages in raw history, got {len(user_msgs)}"
    )
    assert user_msgs[0]["content"] == prompt_1
    assert user_msgs[1]["content"] == prompt_2

    safe_print(
        "\n[LIVE MOCK Persistence]: Successfully verified 2-turn session persistence and raw history ordering."
    )


# =============================================================================
# 8. LIVE MOCK PROVIDER ERROR & RECOVERY SIMULATION
# =============================================================================


async def test_live_mock_error_and_recovery(
    live_mock_environment: MockProvider,
) -> None:
    """Verifies that simulated network errors are caught cleanly and subsequent turns recover."""
    error_pipeline = ScenarioPipeline(name="error_recovery_pipeline")
    error_pipeline.add_step(
        PipelineStep(
            name="err_step",
            prompt_trigger="genera errore di rete",
            exception=ConnectionResetError(
                "Simulazione: socket chiuso dal provider mock"
            ),
        )
    ).add_step(
        PipelineStep(
            name="recovery_step",
            prompt_trigger="riprova connessione",
            response_tokens=["Connessione ripristinata con successo."],
            final_response="Connessione ripristinata con successo.",
        )
    )
    live_mock_environment.pipeline = error_pipeline

    # 1. Turn 1 fails with ConnectionResetError
    with pytest.raises(ConnectionResetError, match="Simulazione: socket chiuso"):
        await run_agent_turn(
            user_prompt="genera errore di rete",
            model="mock-pipeline-model",
            on_token=lambda _: None,
            on_reasoning=lambda *args: None,
            on_tool_event=lambda _: None,
        )

    # 2. Turn 2 recovers smoothly
    recovered_tokens: list[str] = []
    res = await run_agent_turn(
        user_prompt="riprova connessione",
        model="mock-pipeline-model",
        on_token=lambda t: recovered_tokens.append(t),
        on_reasoning=lambda *args: None,
        on_tool_event=lambda _: None,
    )
    assert "ripristinata con successo" in res
    assert "".join(recovered_tokens) == "Connessione ripristinata con successo."
    safe_print(
        "\n[LIVE MOCK Error & Recovery]: Error simulation and subsequent recovery verified."
    )


# =============================================================================
# 9. LIVE MOCK UI COMPONENTS WIREUP
# =============================================================================


async def test_live_mock_ui_components_wireup() -> None:
    """Verifies all UI components wire callbacks cleanly and interact with mock provider state."""

    header = HeaderComponent(
        on_toggle_drawer=lambda: None,
    )
    sidebar = SidebarComponent(
        on_session_switched=lambda _: None,
        on_new_chat=lambda: None,
    )
    input_bar = InputBarComponent(
        on_send=lambda _: None,
        on_stop=lambda: None,
    )
    chat_view = ChatViewComponent(on_suggested_prompt=lambda _: None)

    assert header is not None
    assert sidebar is not None
    assert input_bar is not None
    assert chat_view is not None

    assert not hasattr(header, "mode_toggle")
    safe_print(
        "\n[LIVE MOCK UI Wireup]: UI components tree and mock callbacks verified."
    )
