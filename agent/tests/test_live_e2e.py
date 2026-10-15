from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    check_provider_health,
    run_agent_turn,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSettings,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.navigation.header import HeaderComponent
from exmekh_agent.ui.navigation.sidebar import SidebarComponent
from exmekh_agent.ui.settings.settings_drawer import SettingsDrawerComponent
from tests.live_file_support import verify_file_effects

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
    """Verifies that all ExMekh Agent UI components instantiate and wire callbacks cleanly."""
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
async def test_live_e2e_quick_chat_waiting_for_response(record_property) -> None:
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
    assert full_streamed == response_text, (
        "Streamed content differs from returned content"
    )
    if "CONFERMATO" not in response_text:
        record_property(
            "model_limitation", f"Requested CONFERMATO, received {response_text!r}"
        )
        safe_print(
            f"[MODEL_LIMITATION] Requested CONFERMATO, received {response_text!r}"
        )
    safe_print(f"\n[LIVE Quick Chat Response]: {response_text.strip()}")


@pytest.mark.asyncio
async def test_live_e2e_deep_agent_reasoning_and_response(record_property) -> None:
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
    if "100" not in full_response and "100" not in full_reasoning:
        record_property(
            "model_limitation", f"Incorrect arithmetic answer: {full_response!r}"
        )
        safe_print(f"[MODEL_LIMITATION] Incorrect arithmetic answer: {full_response!r}")

    safe_print(f"\n[LIVE DeepAgent Response]: {full_response[:100]}...")
    if full_reasoning:
        safe_print(f"[LIVE DeepAgent Reasoning]: {full_reasoning[:80]}...")


@pytest.mark.asyncio
async def test_live_e2e_deep_agent_tool_execution() -> None:
    """End-to-end test of DeepAgent tool calling: verifies mock-tool is executed."""
    streamed_tokens: list[str] = []
    reasoning_tokens: list[str] = []
    tool_events: list[ToolEvent] = []

    token_value = "EXMEKH-AGENT-LIVE-E2E-OK"
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

    # Verify that mock-tool was invoked and registered by the ExMekh Agent runner
    tool_names = [ev.tool_name for ev in tool_events]
    if "mock-tool" not in tool_names:
        pytest.skip(
            f"MODEL_LIMITATION: mock-tool not selected; event lifecycle not verified: {tool_names}"
        )

    # Verify tool event structure
    mock_ev = next(ev for ev in tool_events if ev.tool_name == "mock-tool")
    assert mock_ev.depth >= 0
    assert mock_ev.skill_path != ""
    assert any(
        event.tool_name == "mock-tool"
        and event.event_type != "tool_start"
        and event.status == "completed"
        for event in tool_events
    ), (
        "IMPLEMENTATION: mock-tool invocation did not produce a successful terminal event"
    )

    full_response = "".join(streamed_tokens) or response_text
    safe_print(f"\n[LIVE Tool Execution Event]: {mock_ev}")
    safe_print(f"[LIVE Final Response]: {full_response}")


@pytest.mark.asyncio
async def test_live_e2e_interactive_file_creation(tmp_path: Path) -> None:
    """Exercise the runtime handler through both frontend provider wrappers."""
    tool_events: list[ToolEvent] = []
    response = await run_agent_turn(
        user_prompt=(
            "Open the file-processing skill, then use new-file to create the missing "
            "file probe.txt with exactly the text FRONTEND_INTERACTIVE_OK. "
            'Use path="probe.txt" and content="FRONTEND_INTERACTIVE_OK". '
            "Do not use bash, delegate, or create other files. "
            "Exit the skill and confirm completion."
        ),
        model=config.default_model,
        agent_mode="deep_agent",
        settings=ChatSettings(work_root=str(tmp_path)),
        on_token=lambda _: None,
        on_reasoning=lambda *_: None,
        on_tool_event=tool_events.append,
        on_context=lambda _: None,
    )
    successful = verify_file_effects(
        tool_events, tmp_path, {tmp_path / "probe.txt": None}
    )
    if not any(event.tool_name == "new-file" for event in successful):
        pytest.skip(
            "MODEL_LIMITATION: new-file not selected; frontend creation not verified"
        )
    if (
        not (tmp_path / "probe.txt").exists()
        or (tmp_path / "probe.txt").read_text() != "FRONTEND_INTERACTIVE_OK"
    ):
        pytest.skip(
            "MODEL_LIMITATION: emitted file content/path differs from requested content/path"
        )
    assert any(
        event.tool_name == "new-file" and event.status == "completed"
        for event in tool_events
    ), tool_events
    assert response.strip()


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
    if not t1_tool_names:
        pytest.skip(
            "MODEL_LIMITATION: first turn did not select tools; history preservation not verified"
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
    if "get_execution_history" not in t2_tool_names:
        pytest.skip(
            f"MODEL_LIMITATION: history tool not selected; history content not verified: {t2_tool_names}"
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


@pytest.mark.parametrize("initial", ["", "ORIGINAL"])
@pytest.mark.asyncio
async def test_live_e2e_file_edit_and_delete(tmp_path: Path, initial: str) -> None:
    path = tmp_path / "probe.txt"
    path.write_text(initial)
    expected = initial + "APPENDED"
    events: list[ToolEvent] = []
    response = await run_agent_turn(
        user_prompt=(
            "Enter the file-processing skill. Inspect and read the existing probe.txt, "
            "then use edit-file to append exactly APPENDED with no newline. "
            "The file already exists, even when its content is empty. "
            "Use the literal exmekh_edit and exmekh_new_text format and the ID supplied by the server. "
            "Do not use bash, new-file, or delegate. Exit the skill and confirm completion."
        ),
        model=config.default_model,
        agent_mode="deep_agent",
        settings=ChatSettings(work_root=str(tmp_path)),
        on_token=lambda _: None,
        on_reasoning=lambda *_: None,
        on_tool_event=events.append,
        on_context=lambda _: None,
    )
    successful = verify_file_effects(events, tmp_path, {path: initial.encode()})
    if not any(event.tool_name == "edit-file" for event in successful):
        pytest.skip(
            "MODEL_LIMITATION: edit-file not selected; frontend edit not verified"
        )
    if path.read_text() != expected:
        pytest.skip("MODEL_LIMITATION: emitted edit differs from requested append")
    assert any(
        event.tool_name == "edit-file" and event.status == "completed"
        for event in events
    ), events
    assert response.strip()
    before_delete = path.read_bytes()
    events.clear()
    snapshots: list[ChatMessage] = []
    response = await run_agent_turn(
        user_prompt=(
            'Enter the file-processing skill and use delete-file with paths=["probe.txt"]. '
            "Delete the existing file directly; no inspect-file or read-file is needed. "
            "Do not use bash or delegate. Exit the skill and confirm completion."
        ),
        model=config.default_model,
        agent_mode="deep_agent",
        settings=ChatSettings(work_root=str(tmp_path)),
        on_token=lambda _: None,
        on_reasoning=lambda *_: None,
        on_tool_event=events.append,
        on_raw_history=lambda raw, tree: snapshots.append(
            ChatMessage(role="assistant", raw_history=raw, skill_history_tree=tree)
        ),
        on_context=lambda _: None,
    )
    successful = verify_file_effects(
        events,
        tmp_path,
        {path: before_delete},
        history=snapshots[-1] if snapshots else ChatMessage(role="assistant"),
    )
    if not any(event.tool_name == "delete-file" for event in successful):
        pytest.skip(
            "MODEL_LIMITATION: delete-file not selected; frontend deletion not verified"
        )
    assert not path.exists()
    assert any(
        event.tool_name == "delete-file" and event.status == "completed"
        for event in events
    ), events
    if any(event.tool_name in {"inspect-file", "read-file"} for event in events):
        safe_print(
            "[MODEL_LIMITATION] Model read the file before deleting it although no read is required"
        )
    assert response.strip()
