from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ReasoningBlock,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.history.raw_history_view import (
    RawHistoryComponent,
    build_raw_history_from_session,
    format_json_to_highlighted_html,
)


@pytest.fixture
def ui_context():
    c = Client(page("/test-raw-history"))
    with c:
        yield c


# =============================================================================
# 1. JSON SYNTAX HIGHLIGHTER TESTS
# =============================================================================


def test_format_json_primitives() -> None:
    # Null
    res_null = format_json_to_highlighted_html(None)
    assert 'class="json-val-null">null<' in res_null

    # Booleans
    res_true = format_json_to_highlighted_html(True)
    assert 'class="json-val-bool">true<' in res_true
    res_false = format_json_to_highlighted_html(False)
    assert 'class="json-val-bool">false<' in res_false

    # Numbers
    res_int = format_json_to_highlighted_html(42)
    assert 'class="json-val-num">42<' in res_int
    res_float = format_json_to_highlighted_html(3.14)
    assert 'class="json-val-num">3.14<' in res_float

    # String with HTML escaping
    res_str = format_json_to_highlighted_html("<script>alert('xss')</script>")
    assert "&lt;script&gt;" in res_str
    assert 'class="json-readable-string">' in res_str
    assert "<script>" not in res_str


def test_format_json_tool_keys_and_roles() -> None:
    data = {
        "role": "assistant",
        "content": "Running tool now",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "mock-skill",
                    "arguments": {"depth": 1},
                },
            }
        ],
    }

    html_out = format_json_to_highlighted_html(data)

    # Tool keys have distinct amber styling
    assert 'class="json-key json-key-tool"' in html_out
    assert "tool_calls" in html_out
    assert "json-tool-badge" in html_out

    # Role badge
    assert 'class="role-badge role-badge-assistant"' in html_out
    assert "ASSISTANT" in html_out

    # Punctuation
    assert 'class="json-punct">' in html_out


def test_format_json_embedded_json_string() -> None:
    # Embedded JSON string in arguments should be parsed and highlighted rather than escaped
    raw_str = json.dumps({"param_a": 100, "param_b": "value"})
    data = {"arguments": raw_str}

    html_out = format_json_to_highlighted_html(data)
    assert "param_a" in html_out
    assert 'class="json-val-num">100<' in html_out


# =============================================================================
# 2. RAW HISTORY EXTRACTION & SYNTHESIS TESTS
# =============================================================================


def test_build_raw_history_from_recorded_session() -> None:
    session = ChatSession(title="Recorded Session")
    recorded_raw = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "mock-skill"}}],
        },
        {
            "role": "tool",
            "name": "mock-skill",
            "tool_call_id": "call_1",
            "content": "done",
        },
        {"role": "assistant", "content": "Completed skill"},
    ]
    recorded_tree = [
        {
            "skill": "mock-skill",
            "opening_tool_call_id": "call_1",
            "history": [{"role": "system", "content": "Inside skill"}],
            "children": [],
        }
    ]

    session.messages = [
        ChatMessage(
            role="user",
            content="hello",
            raw_history=[{"role": "user", "content": "hello"}],
        ),
        ChatMessage(
            role="assistant",
            content="Completed skill",
            raw_history=recorded_raw[1:],
            skill_history_tree=recorded_tree,
        ),
    ]

    messages, tree = build_raw_history_from_session(session)
    assert len(messages) == 5
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "hello"
    assert len(tree) == 1
    assert tree[0]["skill"] == "mock-skill"
    assert tree[0]["opening_tool_call_id"] == "call_1"


def test_no_duplicate_user_message_and_system_first() -> None:
    """Bug fix verification: initial user message is never duplicated before or after system prompt."""
    session = ChatSession(title="Duplicate Bug Check")
    session.messages = [
        ChatMessage(role="user", content="Prima domanda"),
        ChatMessage(
            role="assistant",
            content="Prima risposta",
            raw_history=[
                {"role": "system", "content": "Custom system prompt"},
                {"role": "user", "content": "Prima domanda"},
                {"role": "assistant", "content": "Prima risposta"},
            ],
        ),
    ]
    messages, _ = build_raw_history_from_session(session)
    assert len(messages) == 3
    # 1. System prompt is placed first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Custom system prompt"
    # 2. User message appears exactly once and after system prompt
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Prima domanda"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Prima risposta"

    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert len(user_msgs) == 1


def test_build_raw_history_synthesis_with_nested_skills() -> None:
    # Test fallback synthesis for sessions containing hierarchical SkillNode objects
    session = ChatSession(title="Synthesized Hierarchical Session")

    sub_sub = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="leaf-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
            )
        ],
    )

    root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/mock-skill",
                depth=1,
                generation_index=1,
                content="Thinking in skill",
            )
        ],
        tool_events=[
            ToolEvent(tool_name="sub-tool", depth=1, skill_path="/mock-skill")
        ],
        sub_skills=[sub_sub],
    )

    session.messages = [
        ChatMessage(role="user", content="Test prompt"),
        ChatMessage(
            role="assistant",
            content="Final answer",
            skills=[root_skill],
        ),
    ]

    messages, tree = build_raw_history_from_session(session)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["source"] == "session_summary"
    assert messages[2]["skills"] == ["mock-skill"]
    assert not any("tool_calls" in message for message in messages)

    # Verify hierarchical matryoshka tree
    assert len(tree) == 1
    assert tree[0]["skill"] == "mock-skill"
    assert len(tree[0]["children"]) == 1
    assert tree[0]["children"][0]["skill"] == "nested-sub-skill"
    assert tree[0]["source"] == "session_summary"
    assert all("tool_calls" not in item for item in tree[0]["history"])


def test_legacy_skill_summary_does_not_fabricate_exit_agent() -> None:
    """The reported success in a chat is not evidence of an exit_agent call."""
    session = ChatSession(title="Unverified exit-agent demonstration")
    session.messages = [
        ChatMessage(role="user", content="Prova un exit_agent invalido"),
        ChatMessage(
            role="assistant",
            content="La skill è tornata al caller.",
            skills=[
                SkillNode(
                    name="mock-skill",
                    path="/mock-skill",
                    depth=1,
                    status="completed",
                )
            ],
        ),
    ]

    messages, tree = build_raw_history_from_session(session)

    assert messages[-1]["source"] == "session_summary"
    assert tree[0]["source"] == "session_summary"
    assert tree[0]["history"] == []
    assert not any(message.get("tool_calls") for message in messages)
    assert not any(message.get("role") == "tool" for message in messages)


# =============================================================================
# 3. RAW HISTORY COMPONENT UI & COMPACTION TESTS
# =============================================================================


def test_raw_history_component_render(ui_context: Client) -> None:
    with ui_context:
        rh = RawHistoryComponent()
        rh.build()

        session = ChatSession(title="UI Test Session")
        session.messages = [
            ChatMessage(role="user", content="Che tempo fa?"),
            ChatMessage(role="assistant", content="C'è il sole."),
        ]

        rh.render_session(session)
        assert rh.status_label is not None
        assert "3 messages" in rh.status_label.text

        # Test empty session
        session.messages.clear()
        rh.render_session(session)
        assert "0 messages" in rh.status_label.text


def test_raw_history_vertical_compaction_and_matryoshka(ui_context: Client) -> None:
    with ui_context:
        rh = RawHistoryComponent()
        rh.build()

        # Create session with sub-skill and sub-sub-skill
        sub_sub = SkillNode(
            name="sub-sub-skill", path="/skill/sub-sub", depth=2, status="completed"
        )
        sub_skill = SkillNode(
            name="root-skill",
            path="/skill",
            depth=1,
            status="completed",
            sub_skills=[sub_sub],
        )

        session = ChatSession(title="Matryoshka Test")
        session.messages = [
            ChatMessage(role="user", content="Esegui"),
            ChatMessage(role="assistant", content="Fatto", skills=[sub_skill]),
        ]

        rh.render_session(session)

        # There should be expansions registered for the nested sub-skills
        assert len(rh.expansions) >= 2

        # Initially all expansions are collapsed
        for exp in rh.expansions:
            assert exp.value is False

        # Test toggle_all expand
        rh.toggle_all(True)
        for exp in rh.expansions:
            assert exp.value is True

        # Test toggle_all collapse
        rh.toggle_all(False)
        for exp in rh.expansions:
            assert exp.value is False


def test_raw_history_copy_full_json(ui_context: Client) -> None:
    with ui_context:
        rh = RawHistoryComponent()
        rh.build()

        # Copy when empty
        with patch.object(ui, "notify") as mock_notify:
            rh.raw_json_data = []
            rh._handle_copy_full_json()
            mock_notify.assert_called_with("No JSON data to copy", type="info")

        # Copy when populated
        with (
            patch.object(ui, "notify") as mock_notify,
            patch.object(ui.clipboard, "write") as mock_clip,
        ):
            rh.raw_json_data = [{"role": "user", "content": "test"}]
            rh._handle_copy_full_json()
            mock_clip.assert_called_once()
            mock_notify.assert_called_with(
                "Full Raw History copied to clipboard", type="positive"
            )


# =============================================================================
# 4. CHAT VIEW SUB-TABS SWITCHING TESTS
# =============================================================================


def test_chat_view_sub_tabs_switching(ui_context: Client) -> None:
    with ui_context:
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
        cv.build()

        # 1. Verify all 3 containers and tab references are initialized
        assert cv.tabs_ref is not None
        assert cv.tab_chat is not None
        assert cv.tab_raw is not None
        assert cv.tab_graph is not None
        assert cv.chat_container is not None
        assert cv.raw_history_container is not None
        assert cv.graph_container is not None

        # 2. Initial state is "chat"
        assert cv.active_tab == "chat"
        assert cv.chat_container.visible is True
        assert cv.raw_history_container.visible is False
        assert cv.graph_container.visible is False

        # 3. Switch to Raw History tab
        cv.switch_tab("raw_history")
        assert cv.active_tab == "raw_history"
        assert cv.chat_container.visible is False
        assert cv.raw_history_container.visible is True
        assert cv.graph_container.visible is False

        # 4. Switch to Graph tab
        cv.switch_tab("graph")
        assert cv.active_tab == "graph"
        assert cv.chat_container.visible is False
        assert cv.raw_history_container.visible is False
        assert cv.graph_container.visible is True

        # 5. Switch back to Chat tab
        cv.switch_tab("chat")
        assert cv.active_tab == "chat"
        assert cv.chat_container.visible is True
        assert cv.raw_history_container.visible is False
        assert cv.graph_container.visible is False

        # 6. Tab change event handler
        event_mock = MagicMock()
        event_mock.value = "raw_history"
        cv._handle_tab_changed(event_mock)
        assert cv.active_tab == "raw_history"
        assert cv.raw_history_container.visible is True


# =============================================================================
# 5. PROGRESSIVE LIVE STREAMING TESTS ("man mano")
# =============================================================================


def test_raw_history_live_streaming(ui_context: Client) -> None:
    with ui_context:
        rh = RawHistoryComponent()
        rh.build()

        session = ChatSession(title="Streaming Session")
        session.messages = [
            ChatMessage(role="user", content="Calcola la radice quadrata")
        ]

        # 1. Start live turn
        rh.start_live_turn(session)
        assert rh.is_streaming is True
        assert rh.live_container is not None
        assert rh.live_assistant_card is not None
        assert rh.live_json_html is not None
        assert rh.live_status_badge is not None
        assert "streaming in progress" in rh.status_label.text

        # 2. Progressive reasoning
        rh.stream_reasoning("Sto analizzando la formula...")
        assert rh.live_reasoning_text == "Sto analizzando la formula..."
        rh._throttled_update_live_json(force=True)
        assert "Sto analizzando la formula..." in rh.live_json_html.content

        # 3. Progressive tool execution
        tool_ev = ToolEvent(tool_name="math_sqrt", depth=0, skill_path="/")
        rh.add_tool_event(tool_ev)
        assert len(rh.live_tool_events) == 1
        assert any(
            m.get("source") == "live_tool_event" and m.get("tool_name") == "math_sqrt"
            for m in rh.raw_json_data
        )

        # 4. Progressive skill event
        skill = SkillNode(
            name="calculator", path="/calculator", depth=1, status="active"
        )
        rh.handle_live_skill_event(skill, "opened")
        assert "/calculator" in rh.live_skill_nodes

        # 5. Progressive content tokens
        rh.stream_token("La radice ")
        rh.stream_token("di 16 è 4.")
        assert rh.live_response_text == "La radice di 16 è 4."
        rh._throttled_update_live_json(force=True)
        assert "La radice di 16 è 4." in rh.live_json_html.content

        # 6. Finalize live turn
        rh.finalize_live_turn()
        assert rh.is_streaming is False
        assert rh.live_status_badge.visible is False


def test_raw_history_expansion_persistence_across_renders(ui_context: Client) -> None:
    with ui_context:
        rh = RawHistoryComponent()
        rh.build()

        sub_skill = SkillNode(
            name="my-skill", path="/my-skill", depth=1, status="completed"
        )
        session = ChatSession(title="Persistence Test")
        session.messages = [
            ChatMessage(role="user", content="Avvia"),
            ChatMessage(role="assistant", content="Fine", skills=[sub_skill]),
        ]

        rh.render_session(session)
        assert len(rh.expansions) >= 1
        # Initially closed
        assert rh.expansions[0].value is False

        # User expands my-skill
        rh.expansions[0].value = True

        # Re-render session (e.g. after stream completion)
        rh.render_session(session)
        assert len(rh.expansions) >= 1
        # Preserved open state!
        assert rh.expansions[0].value is True


def test_chat_view_streaming_integration_with_raw_history(ui_context: Client) -> None:
    with ui_context:
        session = session_manager.get_current_session()
        session.messages = [ChatMessage(role="user", content="Ciao")]

        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
        cv.build()

        # Start stream
        cv.start_assistant_stream()
        assert cv.raw_history_view.is_streaming is True

        # Stream tokens
        cv.stream_token("Ecco la risposta live")
        assert cv.raw_history_view.live_response_text == "Ecco la risposta live"

        # Finalize stream
        cv.finalize_stream()
        assert cv.raw_history_view.is_streaming is False


def test_repeated_skill_disclosures_are_scoped_to_invocation_and_session(ui_context):
    first = SkillNode(name="repeat", path="/repeat", status="completed")
    nested = SkillNode(name="nested", path="/repeat/nested", status="completed")
    second = SkillNode(
        name="repeat", path="/repeat", status="error", sub_skills=[nested]
    )
    session = ChatSession(
        messages=[ChatMessage(role="assistant", skills=[first, second])]
    )
    with ui_context:
        view = RawHistoryComponent()
        view.build()
        view.render_session(session)
        view.expansions[0].value = True
        view.render_session(session)
        assert [exp.value for exp in view.expansions] == [True, False, False]
        labels = [
            element.text
            for element in ui_context.elements.values()
            if isinstance(element, ui.label)
        ]
        assert "ERROR" in labels
        view.toggle_all(True)
        view.render_session(session)
        assert all(exp.value for exp in view.expansions)
        other_session = session.model_copy(update={"id": "another-chat"})
        view.render_session(other_session)
        assert not any(exp.value for exp in view.expansions)
        view.render_session(session)
        view.toggle_all(False)
        view.render_session(session)
        assert not any(exp.value for exp in view.expansions)
