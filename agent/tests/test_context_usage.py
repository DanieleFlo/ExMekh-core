from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

import exmekh_agent.exmekh_bridge.providers.context_usage as cu
from exmekh_agent.config import AppConfig, config
from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    check_provider_health,
    extract_model_context_length,
)
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    _annotate_skill_tree_recursive,
    _apply_skill_compaction_to_scope,
    _is_skill_tree_entry_open,
    _resolve_system_prompt_for_session,
    _serialize_tool_calls_for_counting,
    _slice_already_contains_inner,
    build_compacted_context_messages,
    compute_raw_skill_compaction_tokens,
    compute_session_context_usage,
    compute_skill_node_closure_tokens,
    count_message_dict_tokens,
    count_text_tokens,
    find_skill_opening_index,
    format_compact_tokens,
    format_context_percentage,
    format_skill_closure_tokens,
    get_qwen_tokenizer,
    populate_skill_closure_tokens,
)
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.history.raw_history_view import RawHistoryComponent


@pytest.mark.parametrize("scope", ["root", "skill"])
@pytest.mark.parametrize(
    "event_type,status,has_result",
    [
        ("tool_start", "completed", False),
        ("execution", "error", True),
        ("tool_end", "error", True),
        ("execution", "running", False),
    ],
)
def test_synthesized_context_uses_the_event_lifecycle(
    scope, event_type, status, has_result
):
    event = ToolEvent(tool_name="probe", event_type=event_type, status=status)
    messages = (
        cu._build_turn_messages_for_assistant(
            ChatMessage(role="assistant", tool_events=[event])
        )
        if scope == "root"
        else cu._build_messages_from_skill_node(
            SkillNode(name="work", path="/work", depth=1, tool_events=[event])
        )
    )
    assert (
        any(
            item.get("role") == "tool" and item.get("name") == "probe"
            for item in messages
        )
        is has_result
    )


def test_qwen_tokenizer_counts_tokens_and_fallback() -> None:
    cu._tokenizer_instance = None
    tokenizer = get_qwen_tokenizer()
    assert tokenizer is not None
    assert get_qwen_tokenizer() is tokenizer

    count_text_tokens.cache_clear()
    assert count_text_tokens("") == 0
    count = count_text_tokens("Ciao mondo, test conteggio token con QwenTokenizer.")
    assert count > 0
    assert count == tokenizer.count_tokens(
        "Ciao mondo, test conteggio token con QwenTokenizer."
    )

    # Fallback when tokenizer raises an exception
    count_text_tokens.cache_clear()
    with patch.object(cu, "get_qwen_tokenizer") as mock_get_tok:
        mock_get_tok.return_value.count_tokens.side_effect = RuntimeError("offline")
        assert count_text_tokens("12345678") == 2
    count_text_tokens.cache_clear()


def test_format_compact_tokens_and_percentage() -> None:
    assert format_compact_tokens(0) == "0"
    assert format_compact_tokens(42) == "42"
    assert format_compact_tokens(450) == "0.5k"
    assert format_compact_tokens(1380) == "1.4k"
    assert format_compact_tokens(22000) == "22.0k"
    assert format_compact_tokens(32768) == "32.8k"

    assert format_context_percentage(0, 22000) == "0.0%"
    assert format_context_percentage(1380, 32768) == "4.2%"
    assert format_context_percentage(11000, 22000) == "50.0%"
    assert format_context_percentage(100, 0) == "0.0%"


def test_serialize_and_count_message_dict_tokens() -> None:
    assert _serialize_tool_calls_for_counting(None) == ""
    assert _serialize_tool_calls_for_counting("null") == ""
    assert _serialize_tool_calls_for_counting("   ") == ""
    assert _serialize_tool_calls_for_counting('[{"id": "1"}]') == '[{"id": "1"}]'
    assert _serialize_tool_calls_for_counting([{"id": "1"}]) == '[{"id": "1"}]'

    assert count_message_dict_tokens({"role": "assistant", "content": ""}) == 0
    assert (
        count_message_dict_tokens(
            {
                "role": "assistant",
                "name": "bot",
                "content": "Risposta",
                "tool_calls": [{"id": "c1"}],
                "reasoning_content": "Pensiero",
            }
        )
        > 0
    )


def test_config_context_length_management(tmp_path) -> None:
    cfg = AppConfig()
    cfg.default_model = "model-a"
    cfg.context_length = 16000
    cfg.model_context_lengths.clear()

    # Fallback to cfg.context_length when model_context_lengths is empty
    assert cfg.get_context_length("model-a") == 16000
    assert cfg.get_context_length("") == 16000

    # Ignore invalid set_model_context_length
    cfg.set_model_context_length("", 32000)
    cfg.set_model_context_length("model-a", -5)
    assert cfg.model_context_lengths == {}

    # Set context length for another model and default model
    cfg.set_model_context_length("model-b", 64000)
    assert cfg.get_context_length("model-b") == 64000
    # Unknown models must not inherit another model's advertised context.
    assert cfg.get_context_length("model-a") == 16000

    cfg.set_model_context_length("model-a", 22000)
    assert cfg.context_length == 22000
    assert cfg.get_context_length("model-a") == 22000

    # Update syncs context_length when default_model changes to a known model
    with patch.object(cfg, "save_to_env"):
        cfg.update(default_model="model-b")
        assert cfg.context_length == 64000
        cfg.update(default_model="unknown-model")
        assert cfg.default_model == "unknown-model"


@pytest.mark.asyncio
async def test_inference_engine_context_length_detection() -> None:
    assert extract_model_context_length({"context_length": 22000}) == 22000
    assert (
        extract_model_context_length({"top_provider": {"context_length": 65536}})
        == 65536
    )
    assert extract_model_context_length({"max_model_len": 131072}) == 131072
    assert extract_model_context_length({"top_provider": "invalid"}) is None
    assert extract_model_context_length({}) is None

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "Qwen3.8-27B-UD-Q4_K_S",
                "context_length": 22000,
                "top_provider": {"context_length": 22000},
            },
            {
                "id": "Model-Without-Context-Length",
            },
            {
                "id": "",
            },
        ],
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        is_alive, models = await check_provider_health(
            "http://localhost:8080/v1", "prova"
        )
        assert is_alive is True
        assert "Qwen3.8-27B-UD-Q4_K_S" in models
        assert "Model-Without-Context-Length" in models
        assert config.get_context_length("Qwen3.8-27B-UD-Q4_K_S") == 22000


def test_skill_compaction_helpers_and_edge_cases() -> None:
    # _is_skill_tree_entry_open branches
    assert _is_skill_tree_entry_open({"status": "completed"}) is False
    assert _is_skill_tree_entry_open({"status": "error"}) is False
    assert _is_skill_tree_entry_open({"status": "active"}) is True
    assert _is_skill_tree_entry_open({"output": "done"}) is False
    assert _is_skill_tree_entry_open({"end_index": 3}) is False

    node_active = SkillNode(name="sk1", status="active")
    node_done = SkillNode(name="sk2", status="completed")
    nodes_map = {"sk1": node_active, "sk2": node_done}
    assert _is_skill_tree_entry_open({"skill": "sk1"}, nodes_map) is True
    assert _is_skill_tree_entry_open({"skill": "sk2"}, nodes_map) is False
    assert _is_skill_tree_entry_open({"skill": "unknown"}, nodes_map) is True

    # find_skill_opening_index branches
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": "not-json"},
        {"role": "assistant", "tool_calls": ["not-a-dict"]},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_123", "name": "direct-skill"}],
        },
    ]
    assert find_skill_opening_index(msgs, "direct-skill", None) == 3
    assert find_skill_opening_index(msgs, "other", "call_123") == 3
    assert find_skill_opening_index(msgs, "missing", "call_999") == -1
    assert find_skill_opening_index(msgs, "direct-skill", "call_999") == -1

    # _slice_already_contains_inner branches
    assert _slice_already_contains_inner([], 0, [{"role": "tool", "content": "a"}]) == 0
    seq = [
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "inner1"},
        {"role": "assistant", "content": "after"},
    ]
    assert (
        _slice_already_contains_inner(
            seq,
            0,
            [
                {"role": "tool", "content": "inner1"},
                {"role": "tool", "content": "mismatch"},
            ],
        )
        == 1
    )

    # _apply_skill_compaction_to_scope when closed skill had uncompacted inner messages in scope
    uncompacted_scope = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c_sk", "function": {"name": "my-skill"}}],
        },
        {"role": "tool", "content": "skill instructions"},
        {"role": "assistant", "content": "inner step"},
    ]
    compacted = _apply_skill_compaction_to_scope(
        uncompacted_scope,
        [
            {
                "skill": "my-skill",
                "opening_tool_call_id": "c_sk",
                "output": "Compacted final output",
                "history": uncompacted_scope,
            }
        ],
    )
    assert len(compacted) == 2
    assert compacted[1]["content"] == "Compacted final output"

    # _apply_skill_compaction_to_scope when open skill is not matched by call_id (fallback append)
    fallback_open = _apply_skill_compaction_to_scope(
        [{"role": "assistant", "content": "starting"}],
        [
            {
                "skill": "unmatched-open-skill",
                "output": None,
                "history": [{"role": "tool", "content": "open skill body"}],
            }
        ],
    )
    assert any(m.get("content") == "open skill body" for m in fallback_open)


def test_skill_compaction_closed_vs_open_in_skill_history_tree() -> None:
    """Verify that closed skills only count outer history (compacted output),
    while open skills include all inner skill messages.
    """
    inner_skill_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": '[{"id": "call_open_skill", "type": "function", "function": {"name": "deep-thinking", "arguments": {}}}]',
        },
        {
            "role": "tool",
            "name": "deep-thinking",
            "content": "# Deep Thinking Instructions\n"
            + ("Detailed rules and catalog " * 40),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": '[{"id": "call_web", "type": "function", "function": {"name": "read-web-page", "arguments": {"url": "https://example.com"}}}]',
        },
        {
            "role": "tool",
            "name": "read-web-page",
            "content": "Very long web page body content inside the skill " * 80,
        },
    ]

    outer_raw_when_closed = [
        {"role": "system", "content": "System prompt di test"},
        {"role": "user", "content": "Fai una ricerca approfondita"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": '[{"id": "call_open_skill", "type": "function", "function": {"name": "deep-thinking", "arguments": {}}}]',
        },
        {
            "role": "tool",
            "name": "deep-thinking",
            "content": "<header><name>deep-thinking</name><type>DeepSkill</type><id>0_0_1</id></header>\nSintesi compatta.",
        },
        {
            "role": "assistant",
            "content": "Ecco la risposta finale sintetizzata per l'utente.",
        },
    ]

    # 1. Session where skill is OPEN (output=None, status="active")
    open_session = ChatSession(
        model="Qwen3.8-27B-UD-Q4_K_S",
        system_prompt="System prompt di test",
        messages=[
            ChatMessage(role="user", content="Fai una ricerca approfondita"),
            ChatMessage(
                role="assistant",
                content="Streaming in progress...",
                skills=[
                    SkillNode(
                        name="deep-thinking",
                        path="/deep-thinking",
                        depth=1,
                        status="active",
                    )
                ],
                raw_history=[
                    {"role": "system", "content": "System prompt di test"},
                    {"role": "user", "content": "Fai una ricerca approfondita"},
                    *inner_skill_messages,
                ],
                skill_history_tree=[
                    {
                        "skill": "deep-thinking",
                        "opening_tool_call_id": "call_open_skill",
                        "output": None,
                        "history": inner_skill_messages,
                        "children": [],
                    }
                ],
            ),
        ],
    )

    # 2. Session where skill is CLOSED (output is present, status="completed")
    closed_session = ChatSession(
        model="Qwen3.8-27B-UD-Q4_K_S",
        system_prompt="System prompt di test",
        messages=[
            ChatMessage(role="user", content="Fai una ricerca approfondita"),
            ChatMessage(
                role="assistant",
                content="Ecco la risposta finale sintetizzata per l'utente.",
                skills=[
                    SkillNode(
                        name="deep-thinking",
                        path="/deep-thinking",
                        depth=1,
                        status="completed",
                    )
                ],
                raw_history=outer_raw_when_closed,
                skill_history_tree=[
                    {
                        "skill": "deep-thinking",
                        "opening_tool_call_id": "call_open_skill",
                        "output": "<header><name>deep-thinking</name><type>DeepSkill</type><id>0_0_1</id></header>\nSintesi compatta.",
                        "end_index": 5,
                        "history": inner_skill_messages,
                        "children": [],
                    }
                ],
            ),
        ],
    )

    open_tokens = compute_session_context_usage(open_session).used_tokens
    closed_tokens = compute_session_context_usage(closed_session).used_tokens

    assert open_tokens > closed_tokens
    closed_msgs = build_compacted_context_messages(closed_session)
    assert not any(
        "Very long web page body content inside the skill" in str(m.get("content", ""))
        for m in closed_msgs
    )
    open_msgs = build_compacted_context_messages(open_session)
    assert any(
        "Very long web page body content inside the skill" in str(m.get("content", ""))
        for m in open_msgs
    )


def test_nested_matryoshka_skills_and_synthesized_turns() -> None:
    closed_child_sub_skill = SkillNode(
        name="closed-sub-skill",
        path="/parent-skill/closed-sub-skill",
        depth=2,
        status="completed",
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/parent-skill/closed-sub-skill",
                depth=2,
                content="Reasoning interno della sub-skill chiusa che NON deve pesare "
                * 30,
            )
        ],
    )
    open_child_sub_skill = SkillNode(
        name="open-sub-skill",
        path="/parent-skill/open-sub-skill",
        depth=2,
        status="active",
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/parent-skill/open-sub-skill",
                depth=2,
                content="",
            ),
            ReasoningBlock(
                skill_path="/parent-skill/open-sub-skill",
                depth=2,
                content="Reasoning interno della sub-skill APERTA che DEVE essere contato "
                * 20,
            ),
        ],
        tool_events=[
            ToolEvent(
                tool_name="running-tool",
                depth=2,
                skill_path="/parent-skill/open-sub-skill",
                status="running",
                event_type="tool_start",
            ),
            ToolEvent(
                tool_name="done-tool",
                depth=2,
                skill_path="/parent-skill/open-sub-skill",
                status="completed",
                event_type="tool_end",
            ),
        ],
    )
    parent_skill = SkillNode(
        name="parent-skill",
        path="/parent-skill",
        depth=1,
        status="active",
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/parent-skill",
                depth=1,
                content="Reasoning del parent skill aperto.",
            )
        ],
        sub_skills=[closed_child_sub_skill, open_child_sub_skill],
    )

    session = ChatSession(
        system_prompt="",
        agent_mode="quick_chat",
        messages=[
            ChatMessage(role="system", content="Custom system message"),
            ChatMessage(role="user", content="Esegui skill annidate"),
            ChatMessage(
                role="assistant",
                content="Risposta parziale",
                reasoning_blocks=[
                    ReasoningBlock(skill_path="/", depth=0, content="Root thinking")
                ],
                tool_events=[
                    ToolEvent(
                        tool_name="root-tool-done",
                        depth=0,
                        skill_path="/",
                        status="completed",
                    ),
                    ToolEvent(
                        tool_name="root-tool-running",
                        depth=0,
                        skill_path="/",
                        status="running",
                        event_type="tool_start",
                    ),
                    ToolEvent(
                        tool_name="inner-tool-ignored-at-root",
                        depth=1,
                        skill_path="/parent-skill",
                        status="completed",
                    ),
                ],
                skills=[parent_skill],
            ),
        ],
    )

    visible_msgs = build_compacted_context_messages(session)
    joined_content = "\n".join(str(m.get("content", "")) for m in visible_msgs)
    assert "Custom system message" in joined_content
    assert "Root thinking" in joined_content
    assert "Reasoning del parent skill aperto." in joined_content
    assert (
        "Reasoning interno della sub-skill APERTA che DEVE essere contato"
        in joined_content
    )
    assert (
        "Reasoning interno della sub-skill chiusa che NON deve pesare"
        not in joined_content
    )

    # Synthesized turn with skill_history_tree (without raw_history)
    session_tree_only = ChatSession(
        agent_mode="deep_agent",
        system_prompt="",
        messages=[
            ChatMessage(role="user", content="Domanda"),
            ChatMessage(
                role="assistant",
                content="",
                reasoning_blocks=[
                    ReasoningBlock(skill_path="/s1", depth=1, content="Inner block")
                ],
                skill_history_tree=[
                    {
                        "skill": "s1",
                        "output": None,
                        "history": [{"role": "tool", "content": "tree-only-body"}],
                    }
                ],
            ),
        ],
    )
    with patch(
        "exmekh_agent.exmekh_bridge.providers.context_usage.load_real_hds_system_prompt",
        return_value="Real HDS Prompt",
    ):
        tree_msgs = build_compacted_context_messages(session_tree_only)
        assert tree_msgs[0]["content"] == "Real HDS Prompt"
        assert any(m.get("content") == "tree-only-body" for m in tree_msgs)

    # find_skill_opening_index with tool_calls=None
    assert (
        find_skill_opening_index(
            [{"role": "assistant", "tool_calls": None}], "s1", None
        )
        == -1
    )

    # _apply_skill_compaction_to_scope when closed skill has no call_id
    compacted_no_cid = _apply_skill_compaction_to_scope(
        [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "sk-no-cid"}}],
            },
            {"role": "tool", "content": "inner-step"},
        ],
        [
            {
                "skill": "sk-no-cid",
                "output": "done-no-cid",
                "history": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "sk-no-cid"}}],
                    },
                    {"role": "tool", "content": "inner-step"},
                ],
            }
        ],
    )
    assert compacted_no_cid[1]["content"] == "done-no-cid"
    assert "tool_call_id" not in compacted_no_cid[1]


def test_collector_sub_agent_and_system_prompt_resolution() -> None:
    empty_session = ChatSession()
    assert build_compacted_context_messages(empty_session) == []

    # Collector snapshot during live turn
    collector = TurnStreamCollector()
    collector.append_token("Live token")
    collector.append_reasoning("Live reasoning")
    session_with_collector = ChatSession(
        agent_mode="quick_chat",
        messages=[ChatMessage(role="user", content="Domanda live")],
    )
    msgs_live = build_compacted_context_messages(
        session_with_collector, collector=collector
    )
    assert any(m.get("content") == "Live token" for m in msgs_live)

    # Sub-agent conversation context usage & raw_history with empty msg.content & session.system_prompt
    sub_conv = SubAgentConversation(
        path="main/coding-agent",
        messages=[
            ChatMessage(role="user", content="Task sub-agent"),
            ChatMessage(
                role="assistant",
                content="",
                raw_history=[{"role": "assistant", "content": "Done"}],
            ),
        ],
    )
    session_with_sys_prompt = ChatSession(
        system_prompt="Prompt di sistema della sessione",
        messages=sub_conv.messages,
    )
    sub_usage = compute_session_context_usage(
        session_with_sys_prompt, sub_agent=sub_conv
    )
    assert sub_usage.used_tokens > 0

    # Zero context_length edge case
    with patch.object(config, "get_context_length", return_value=0):
        usage = compute_session_context_usage(session_with_collector)
        assert usage.context_length == 0
        assert usage.percentage == 0.0
        assert compute_session_context_usage(session_with_collector).estimated

    # _resolve_system_prompt_for_session fallback when deep_agent prompt returns empty
    with patch(
        "exmekh_agent.exmekh_bridge.providers.context_usage.load_real_hds_system_prompt",
        return_value="",
    ):
        fallback_prompt = _resolve_system_prompt_for_session(
            ChatSession(agent_mode="deep_agent", system_prompt=""),
            [ChatMessage(role="user", content="test")],
        )
        assert "AI assistant" in fallback_prompt


def test_input_bar_context_display_and_updates() -> None:
    ib = InputBarComponent(on_send=lambda _: None, on_stop=lambda: None)
    with Client(page("/test-context-input-bar")), ui.column():
        ib.build()

    assert ib.model_label is not None
    assert ib.context_percent_label is not None
    assert ib.context_tokens_label is not None

    ib.update_context_info(
        model_name="Qwen3.8-27B-UD-Q4_K_S",
        used_tokens=1380,
        context_length=32768,
    )
    assert ib.model_label.text == "Qwen3.8-27B-UD-Q4_K_S"
    assert ib.context_percent_label.text == "4.2%"
    assert ib.context_tokens_label.text == "1.4k/32.8k"
    assert "text-emerald-400" in ib.context_percent_label._classes

    # Medium usage (amber)
    ib.update_context_info(
        model_name="Qwen3.8-27B-UD-Q4_K_S",
        used_tokens=15000,
        context_length=22000,
    )
    assert ib.context_percent_label.text == "68.2%"
    assert ib.context_tokens_label.text == "15.0k/22.0k"
    assert "text-amber-400" in ib.context_percent_label._classes

    # High usage (rose)
    ib.update_context_info(
        model_name="Qwen3.8-27B-UD-Q4_K_S",
        used_tokens=20000,
        context_length=22000,
    )
    assert ib.context_percent_label.text == "90.9%"
    assert ib.context_tokens_label.text == "20.0k/22.0k"
    assert "text-rose-400" in ib.context_percent_label._classes


def test_skill_closure_tokens_computation_and_ui_rendering() -> None:
    assert isinstance(cu.load_real_hds_system_prompt(), str)
    assert format_skill_closure_tokens(4800, 1200) == "4.8k → 1.2k token"
    assert format_skill_closure_tokens(850, 320) == "0.8k → 0.3k token"
    assert format_skill_closure_tokens(85, 32) == "85 → 32 token"

    # 1. Pre-populated raw skill dict returns cached values immediately
    cached_sk = {
        "skill": "cached-skill",
        "tokens_before_close": 3500,
        "tokens_after_close": 900,
    }
    assert compute_raw_skill_compaction_tokens(cached_sk) == (3500, 900)

    # 2. Raw skill dict without prefix_messages and with non-standard history fallback
    fallback_sk: dict[str, object] = {
        "skill": "fallback-skill",
        "output": "risultato sintetico",
        "history": [
            {
                "role": "assistant",
                "content": "Ragionamento interno lungo " * 20,
            }
        ],
    }
    b_tok, a_tok = compute_raw_skill_compaction_tokens(fallback_sk)
    assert b_tok > a_tok
    assert fallback_sk["tokens_before_close"] == b_tok
    assert fallback_sk["tokens_after_close"] == a_tok

    # 3. compute_skill_node_closure_tokens with nested sub_skills (active and completed)
    closed_sub = SkillNode(
        name="sub-done",
        path="/main-skill/sub-done",
        depth=2,
        status="completed",
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/main-skill/sub-done",
                depth=2,
                content="Sub skill reasoning " * 15,
            )
        ],
    )
    open_sub = SkillNode(
        name="sub-active",
        path="/main-skill/sub-active",
        depth=2,
        status="active",
    )
    parent_node = SkillNode(
        name="main-skill",
        path="/main-skill",
        depth=1,
        status="completed",
        reasoning_blocks=[
            ReasoningBlock(
                skill_path="/main-skill",
                depth=1,
                content="Parent skill reasoning " * 25,
            )
        ],
        tool_events=[
            ToolEvent(
                tool_name="run-search",
                skill_path="/main-skill",
                depth=1,
                status="completed",
            )
        ],
        sub_skills=[closed_sub, open_sub],
    )
    p_before, p_after = compute_skill_node_closure_tokens(parent_node, base_tokens=200)
    assert p_before > p_after
    assert parent_node.tokens_before_close == p_before
    assert parent_node.tokens_after_close == p_after
    assert closed_sub.tokens_before_close is not None
    assert closed_sub.tokens_after_close is not None
    assert open_sub.tokens_before_close is None
    # Calling again returns the already-computed values
    assert compute_skill_node_closure_tokens(parent_node) == (p_before, p_after)

    # 4. _annotate_skill_tree_recursive with empty prior_messages and unmatched opening index
    unmatched_tree = [
        {
            "skill": "",
            "output": "ok",
            "history": [{"role": "tool", "content": "step"}],
        }
    ]
    matched_map: dict[str, tuple[int, int]] = {}
    _annotate_skill_tree_recursive([], unmatched_tree, [], matched_map)
    assert unmatched_tree[0]["tokens_before_close"] > 0

    # 5. populate_skill_closure_tokens on empty session, session with raw_history+tree, and live collector
    populate_skill_closure_tokens(ChatSession())

    session = ChatSession(
        agent_mode="quick_chat",
        messages=[
            ChatMessage(role="user", content="Esegui analisi"),
            ChatMessage(
                role="assistant",
                content="Fatto",
                raw_history=[
                    {"role": "user", "content": "Esegui analisi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_s1",
                                "function": {"name": "skill-one", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_s1",
                        "content": "Sintesi s1",
                    },
                ],
                skill_history_tree=[
                    {
                        "skill": "skill-one",
                        "opening_tool_call_id": "call_s1",
                        "output": "Sintesi s1",
                        "history": [
                            {"role": "user", "content": "Esegui analisi"},
                            {
                                "role": "assistant",
                                "content": "Ragionamento interno molto esteso " * 30,
                            },
                            {"role": "tool", "content": "Output intermedio " * 20},
                        ],
                        "children": [
                            {
                                "skill": "nested-closed",
                                "output": "Done nested",
                                "history": [
                                    {
                                        "role": "assistant",
                                        "content": "Inner nested " * 10,
                                    }
                                ],
                            },
                            {
                                "skill": "nested-open",
                                "output": None,
                                "history": [{"role": "assistant", "content": "Open"}],
                            },
                        ],
                    }
                ],
                skills=[
                    SkillNode(
                        name="skill-one",
                        path="/skill-one",
                        depth=1,
                        status="completed",
                        sub_skills=[
                            SkillNode(
                                name="unmatched-sub",
                                path="",
                                depth=2,
                                status="completed",
                            ),
                            SkillNode(
                                name="active-sub",
                                path="/skill-one/active-sub",
                                depth=2,
                                status="active",
                            ),
                        ],
                    )
                ],
            ),
        ],
    )
    collector = TurnStreamCollector()
    collector_skill_open = SkillNode(
        name="live-skill", path="/live-skill", depth=1, status="active"
    )
    collector.append_skill(collector_skill_open, action="opened")
    collector_skill_closed = SkillNode(
        name="live-skill",
        path="/live-skill",
        depth=1,
        status="completed",
        tokens_before_close=2400,
        tokens_after_close=800,
    )
    collector.append_skill(collector_skill_closed, action="completed")
    assert collector.skills[0].tokens_before_close == 2400
    assert collector.skills[0].tokens_after_close == 800

    # Also test collector with skill_history_tree & empty raw_history
    collector.set_raw_history(
        [],
        [
            {
                "skill": "live-skill",
                "output": "Done live",
                "history": [{"role": "assistant", "content": "Live inner " * 15}],
            }
        ],
    )
    populate_skill_closure_tokens(session, collector=collector)
    first_skill = session.messages[1].skills[0]
    assert first_skill.tokens_before_close is not None
    assert first_skill.tokens_after_close is not None
    assert first_skill.tokens_before_close > first_skill.tokens_after_close

    # Sub-agent populate_skill_closure_tokens branch
    sub_conv = SubAgentConversation(
        path="main/sub",
        messages=session.messages,
    )
    populate_skill_closure_tokens(session, sub_agent=sub_conv)

    # 6. Render closed skill in ChatViewComponent and RawHistoryComponent
    with Client(page("/test-context-skill-rendering")), ui.column():
        chat_view = ChatViewComponent(on_suggested_prompt=lambda _: None)
        chat_view._render_skill_node(first_skill)
        raw_view = RawHistoryComponent()
        raw_view._render_skill_compaction(
            session.messages[1].skill_history_tree[0], depth=1, is_live=False
        )
