from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from fe.config import config
from fe.exmekh_bridge.agent_runner import check_provider_health, run_agent_turn
from fe.exmekh_bridge.models import ChatMessage, SkillNode, ToolEvent
from fe.state.session_manager import SessionManager
from fe.ui.chat_view import ChatViewComponent
from fe.ui.header import HeaderComponent
from fe.ui.input_bar import InputBarComponent
from fe.ui.settings_drawer import SettingsDrawerComponent
from fe.ui.sidebar import SidebarComponent

load_dotenv()
pytestmark = pytest.mark.live


def safe_print(msg: str) -> None:
    """Safe print helper preventing Windows cp1252 UnicodeEncodeError with emojis."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


@pytest.fixture(autouse=True)
async def check_real_provider_available(request: pytest.FixtureRequest) -> None:
    if request.node.name == "test_live_web_ui_components_init":
        return
    is_alive, models = await check_provider_health()
    if not is_alive or not models:
        pytest.skip(
            f"Provider locale non pronto per model '{config.default_model}' (modelli disponibili: {models}). "
            "I test live con provider mock sono disponibili in tests/test_live_mock_provider.py."
        )
    if config.default_model not in models:
        config.default_model = models[0]


@pytest.mark.asyncio
async def test_live_provider_health_check() -> None:
    """Verifies that the provider health check contacts the real local provider."""
    is_alive, models = await check_provider_health()
    assert is_alive is True, (
        f"Expected local provider to be online at {config.base_url}"
    )
    assert len(models) > 0, (
        "Expected at least one model to be returned by local provider"
    )
    safe_print(f"\n[LIVE] Detected models: {models}")


def test_live_web_ui_components_init() -> None:
    """Verifies that all FE UI components instantiate and wire callbacks cleanly."""
    header = HeaderComponent(
        on_toggle_drawer=lambda: None,
    )
    sidebar = SidebarComponent(
        on_session_switched=lambda _: None,
        on_new_chat=lambda: None,
    )
    settings_drawer = SettingsDrawerComponent(on_settings_saved=lambda: None)
    chat_view = ChatViewComponent(on_suggested_prompt=lambda _: None)
    input_bar = InputBarComponent(on_send=lambda _: None, on_stop=lambda: None)

    assert header is not None
    assert sidebar is not None
    assert settings_drawer is not None
    assert chat_view is not None
    assert input_bar is not None
    safe_print(
        "\n[LIVE Web UI]: Successfully verified UI component tree initialization."
    )


@pytest.mark.asyncio
async def test_live_e2e_quick_chat_waiting_for_response() -> None:
    """End-to-end test of Quick Chat mode: streams tokens and waits for final answer."""
    streamed_tokens: list[str] = []
    reasoning_tokens: list[str] = []
    tool_events: list[ToolEvent] = []

    prompt = "Rispondi esclusivamente con la parola 'CONFERMATO'."

    response_text = await run_agent_turn(
        user_prompt=prompt,
        model=config.default_model,
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda r: reasoning_tokens.append(r),
        on_tool_event=lambda ev: tool_events.append(ev),
        agent_mode="quick_chat",
    )

    full_streamed = "".join(streamed_tokens)
    assert response_text.strip(), "Expected non-empty response from model"
    assert "CONFERMATO" in response_text or "CONFERMATO" in full_streamed, (
        f"Expected 'CONFERMATO' in response, got: {response_text!r}"
    )
    safe_print(f"\n[LIVE Quick Chat Response]: {response_text.strip()}")


@pytest.mark.asyncio
async def test_live_e2e_deep_agent_reasoning_and_response() -> None:
    """End-to-end test of DeepAgent: verifies reasoning streaming and final calculation."""
    streamed_tokens: list[str] = []
    reasoning_tokens: list[str] = []
    tool_events: list[ToolEvent] = []

    prompt = "Risolvi passo a passo: quanto fa 25 * 4? Fornisci il risultato finale."

    response_text = await run_agent_turn(
        user_prompt=prompt,
        model=config.default_model,
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda r: reasoning_tokens.append(r),
        on_tool_event=lambda ev: tool_events.append(ev),
        agent_mode="deep_agent",
    )

    full_response = "".join(streamed_tokens) or response_text
    full_reasoning = "".join(reasoning_tokens)

    assert full_response.strip(), "Expected non-empty response from DeepAgent"
    assert "100" in full_response or "100" in full_reasoning, (
        f"Expected calculation result '100' in response or reasoning, got response: {full_response}"
    )

    safe_print(f"\n[LIVE DeepAgent Response]: {full_response[:100]}...")
    if full_reasoning:
        safe_print(f"[LIVE DeepAgent Reasoning]: {full_reasoning[:80]}...")


@pytest.mark.asyncio
async def test_live_e2e_deep_agent_tool_execution() -> None:
    """End-to-end test of DeepAgent tool calling: verifies mock-tool is executed."""
    streamed_tokens: list[str] = []
    reasoning_tokens: list[str] = []
    tool_events: list[ToolEvent] = []

    token_value = "FE-LIVE-E2E-OK"
    prompt = (
        f"Usa il tool mock-tool una sola volta con il parametro text='{token_value}'. "
        f"Poi restituisci una breve conferma che includa la parola {token_value}."
    )

    response_text = await run_agent_turn(
        user_prompt=prompt,
        model=config.default_model,
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda r: reasoning_tokens.append(r),
        on_tool_event=lambda ev: tool_events.append(ev),
        agent_mode="deep_agent",
    )

    # Verify that mock-tool was invoked and registered by the FE runner
    tool_names = [ev.tool_name for ev in tool_events]
    assert "mock-tool" in tool_names, (
        f"Expected 'mock-tool' to be executed during DeepAgent run. Observed tools: {tool_names}"
    )

    # Verify tool event structure
    mock_ev = next(ev for ev in tool_events if ev.tool_name == "mock-tool")
    assert mock_ev.depth >= 0
    assert mock_ev.skill_path != ""

    full_response = "".join(streamed_tokens) or response_text
    safe_print(f"\n[LIVE Tool Execution Event]: {mock_ev}")
    safe_print(f"[LIVE Final Response]: {full_response}")


@pytest.mark.asyncio
async def test_live_e2e_full_session_persistence(tmp_path: Path) -> None:
    """End-to-end test of full user interaction saved and reloaded with SessionManager."""
    sm = SessionManager(tmp_path)

    session = sm.get_or_create_empty_session()
    assert sm.update_title(session, "Live E2E Persistence Test")
    assert sm.get_or_create_empty_session() is session
    prompt = "Scrivi semplicemente 'TEST_PERSISTENZA_OK'."

    user_msg = ChatMessage(role="user", content=prompt)
    assert sm.add_message(session, user_msg)

    streamed_tokens: list[str] = []
    reasoning_tokens: list[str] = []
    tool_events: list[ToolEvent] = []

    response_text = await run_agent_turn(
        user_prompt=prompt,
        model=config.default_model,
        on_token=lambda t: streamed_tokens.append(t),
        on_reasoning=lambda r: reasoning_tokens.append(r),
        on_tool_event=lambda ev: tool_events.append(ev),
        agent_mode="quick_chat",
    )

    assistant_msg = ChatMessage(
        role="assistant",
        content="".join(streamed_tokens) or response_text,
        reasoning_content="".join(reasoning_tokens),
        tool_events=tool_events,
    )
    assert sm.add_message(session, assistant_msg)

    # Reload from disk with fresh SessionManager
    new_sm = SessionManager(tmp_path)

    loaded = new_sm.get_session(session.id)
    assert loaded is not None, "Failed to reload session from storage"
    assert len(loaded.messages) == 2, f"Expected 2 messages, got {len(loaded.messages)}"
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == prompt
    assert loaded.messages[1].role == "assistant"
    assert loaded.messages[1].content != ""

    empty = new_sm.get_or_create_empty_session()
    assert empty.id != loaded.id
    assert empty.is_empty
    new_sm.switch_session(loaded.id)
    assert new_sm.get_or_create_empty_session() is empty
    assert len(new_sm.list_sessions_sorted()) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2

    safe_print(
        f"\n[LIVE Session Persistence]: Successfully persisted {len(loaded.messages)} messages to disk."
    )


@pytest.mark.asyncio
async def test_live_e2e_multi_turn_execution_history_preserves_prior_tools() -> None:
    """Live E2E test verifying that after Turn 1 completes (executing skills/tools),

    Turn 2 starts strictly after Turn 1 finishes and calling get_execution_history
    shows all tools/skills used in Turn 1 (not <total>1</total>).
    """
    session_messages: list[ChatMessage] = []

    # =========================================================================
    # TURN 1: Execute mock-skill / tools and wait for full completion
    # =========================================================================
    prompt_1 = "Usa la skill mock-skill e invoca al suo interno i sotto-tool per dimostrare l'annidamento."
    user_msg_1 = ChatMessage(role="user", content=prompt_1)
    session_messages.append(user_msg_1)

    t1_tokens: list[str] = []
    t1_reasoning: list[str] = []
    t1_tools: list[ToolEvent] = []
    t1_skills: list[SkillNode] = []
    t1_raw_history: list[dict] = []
    t1_skill_tree: list[dict] = []

    def on_skill_t1(node: SkillNode, action: str) -> None:
        if (
            action == "opened"
            and node.depth == 1
            and all(s.id != node.id for s in t1_skills)
        ):
            t1_skills.append(node)

    def on_raw_t1(raw_hist: list[dict], skill_tree: list[dict]) -> None:
        t1_raw_history.clear()
        t1_raw_history.extend(raw_hist)
        t1_skill_tree.clear()
        t1_skill_tree.extend(skill_tree)

    response_1 = await run_agent_turn(
        user_prompt=prompt_1,
        model=config.default_model,
        on_token=lambda t: t1_tokens.append(t),
        on_reasoning=lambda r, *_: t1_reasoning.append(r),
        on_tool_event=lambda ev: t1_tools.append(ev),
        on_skill_event=on_skill_t1,
        on_raw_history=on_raw_t1,
        agent_mode="deep_agent",
        history=[],
    )

    assistant_msg_1 = ChatMessage(
        role="assistant",
        content="".join(t1_tokens) or response_1,
        reasoning_content="".join(t1_reasoning),
        tool_events=t1_tools,
        skills=t1_skills,
        raw_history=t1_raw_history,
        skill_history_tree=t1_skill_tree,
    )
    session_messages.append(assistant_msg_1)

    t1_tool_names = [ev.tool_name for ev in t1_tools]
    assert len(t1_tool_names) > 0, (
        f"Expected Turn 1 to execute tools/skills, got: {t1_tool_names}"
    )
    assert assistant_msg_1.content.strip() != "", (
        "Turn 1 must finish with non-empty assistant response"
    )
    safe_print(f"\n[LIVE Turn 1 Completed] Executed tools: {t1_tool_names}")

    # =========================================================================
    # TURN 2: Starts strictly AFTER Turn 1 has completed
    # =========================================================================
    prior_history = list(session_messages)
    prompt_2 = (
        "Utilizza ora il tool get_execution_history per verificare la history di esecuzione "
        "e fai un confronto con i tool eseguiti nel turno precedente."
    )
    user_msg_2 = ChatMessage(role="user", content=prompt_2)
    session_messages.append(user_msg_2)

    t2_tokens: list[str] = []
    t2_reasoning: list[str] = []
    t2_tools: list[ToolEvent] = []
    t2_raw_history: list[dict] = []
    t2_skill_tree: list[dict] = []

    def on_raw_t2(raw_hist: list[dict], skill_tree: list[dict]) -> None:
        t2_raw_history.clear()
        t2_raw_history.extend(raw_hist)
        t2_skill_tree.clear()
        t2_skill_tree.extend(skill_tree)

    response_2 = await run_agent_turn(
        user_prompt=prompt_2,
        model=config.default_model,
        on_token=lambda t: t2_tokens.append(t),
        on_reasoning=lambda r, *_: t2_reasoning.append(r),
        on_tool_event=lambda ev: t2_tools.append(ev),
        on_raw_history=on_raw_t2,
        agent_mode="deep_agent",
        history=prior_history,
    )

    t2_tool_names = [ev.tool_name for ev in t2_tools]
    assert "get_execution_history" in t2_tool_names, (
        f"Expected 'get_execution_history' to be called in Turn 2, observed: {t2_tool_names}"
    )

    # Extract the <execution-history> tool output from Turn 2's raw_history
    exec_history_outputs = [
        str(m.get("content", ""))
        for m in t2_raw_history
        if m.get("role") == "tool"
        and "<execution-history>" in str(m.get("content", ""))
    ]
    assert len(exec_history_outputs) > 0, (
        "Expected <execution-history> XML output in Turn 2 raw_history"
    )
    latest_history_xml = exec_history_outputs[-1]
    safe_print(f"\n[LIVE Turn 2 <execution-history> XML]:\n{latest_history_xml}")

    # Crucial bug check: must NOT be <total>1</total>, and must contain Turn 1's tools!
    assert "<total>1</total>" not in latest_history_xml, (
        f"Bug detected! get_execution_history returned <total>1</total> instead of prior tools:\n{latest_history_xml}"
    )
    assert (
        "<name>mock-skill</name>" in latest_history_xml
        or "<name>mock-tool</name>" in latest_history_xml
    ), (
        f"Expected Turn 1 capabilities in <execution-history>, got:\n{latest_history_xml}"
    )
    safe_print(f"\n[LIVE Turn 2 Final Response]: {response_2[:300]}...")
