from __future__ import annotations

import math
from collections.abc import Generator

import pytest
from exmekh_lib.utils.xml import stringify_tool_result
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.history.skill_history import (
    inner_skill_history_messages,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.ui.graph.graph_data import (
    FIXED_NODE_SPACING_X,
    FIXED_SUB_AGENT_NODE_SPACING,
    NODE_COLORS,
    GraphHistoryLine,
    GraphNode,
    GraphSkillBranch,
    _build_nodes_for_messages,
    _build_tool_triplet_events,
    _ensure_interaction_nodes_on_main,
    _ensure_sub_agent_branches_do_not_cross,
    _extract_nodes_from_raw_scope,
    _extract_real_system_prompt_event,
    _layout_main_horizontal_history,
    _layout_sub_agent_constellation_line,
    _segments_cross,
    build_graph_data_from_session,
    check_sub_agent_branches_do_not_cross,
)
from exmekh_agent.ui.graph.graph_svg import render_star_svg_html
from exmekh_agent.ui.graph.graph_view import GraphViewComponent
from exmekh_agent.ui.history.raw_history_view import load_real_hds_system_prompt


@pytest.fixture(autouse=True)
def graph_ui_context() -> Generator[None]:
    client = Client(page("/test-graph-view"))
    with client:
        yield


@pytest.mark.parametrize(
    "content, cancelled",
    [
        (stringify_tool_result({"status": "cancelled", "executed": False}), True),
        (stringify_tool_result({"status": "cancelled", "executed": True}), False),
        (stringify_tool_result({"status": "completed", "executed": False}), False),
        ("<other><status>cancelled</status><executed>false</executed></other>", False),
        ("<tool-result/>", False),
        ("ordinary tool output", False),
    ],
)
def test_undispatched_call_receipt_does_not_invent_an_execution(content, cancelled):
    events = _build_tool_triplet_events(
        tool_name="bash",
        tool_call_id="cancelled-call",
        arguments={},
        assistant_content="",
        tool_event=None,
        tool_response_content=content,
        skill_path="/file-processing",
        depth=1,
    )
    stages = [event["event_stage"] for event in events]
    if cancelled:
        assert stages == ["tool_call", "tool_response"]
        assert events[-1]["status"] == "cancelled"
        assert events[-1]["executed"] is False
    else:
        assert stages == ["tool_call", "tool_execution", "tool_response"]


@pytest.mark.parametrize("tool_header_count", [1, 3])
@pytest.mark.parametrize("has_response", [False, True], ids=["live", "completed"])
def test_skill_history_keeps_steering_before_the_first_inner_response(
    tool_header_count: int, has_response: bool
) -> None:
    prefix = [{"role": "assistant", "content": "Parent opening call"}] + [
        {"role": "tool", "content": f"Opening batch result {index}"}
        for index in range(tool_header_count)
    ]
    inner = [
        {"role": "user", "content": "First guidance"},
        {"role": "user", "content": "Second guidance"},
    ]
    if has_response:
        inner.append({"role": "assistant", "content": "Guided response"})
    assert inner_skill_history_messages(prefix + inner) == inner


def test_main_history_is_horizontal_and_passes_through_center() -> None:
    session = ChatSession(
        id="chat-horiz",
        title="Chat Orizzontale",
        messages=[
            ChatMessage(role="user", content="Ciao, usa un tool"),
            ChatMessage(
                role="assistant",
                content="Ecco il risultato finale.",
                tool_events=[
                    ToolEvent(
                        tool_call_id="call_1",
                        tool_name="search-web",
                        status="completed",
                        depth=0,
                        skill_path="/",
                        duration_seconds=1.25,
                    )
                ],
            ),
        ],
    )
    graph = build_graph_data_from_session(session)
    assert len(graph.histories) == 1
    main = graph.histories[0]
    assert main.is_main is True
    assert main.angle_rad == 0.0
    assert main.angle_deg == 0.0
    assert main.y1 == graph.center_y == 225.0
    assert main.y2 == graph.center_y == 225.0
    assert main.x1 < graph.center_x < main.x2

    # All depth-0 nodes on main history lie on the horizontal line y = 225.0
    # System Prompt and Utente are separated into distinct nodes!
    assert len(main.nodes) == 4
    assert [n.kind for n in main.nodes] == [
        "system",
        "user",
        "tool_bundle",
        "assistant",
    ]
    assert all(n.y == 225.0 for n in main.nodes)
    # System Prompt node contains ONLY the real HDS system prompt (not an invented placeholder)
    sys_node = main.nodes[0]
    user_node = main.nodes[1]
    assert len(sys_node.events) == 1
    assert sys_node.events[0]["role"] == "system"
    assert "collaborative assistant" in sys_node.events[0]["content"]
    assert (
        "You are a friendly, accurate, and professional AI assistant."
        not in sys_node.events[0]["content"]
    )
    # Utente node contains ONLY the user message event
    assert len(user_node.events) == 1
    assert user_node.events[0]["role"] == "user"
    assert user_node.events[0]["content"] == "Ciao, usa un tool"
    # Colors are distinct per node role/type
    assert main.nodes[0].color == NODE_COLORS["system"]
    assert main.nodes[1].color == NODE_COLORS["user"]
    assert main.nodes[2].color == NODE_COLORS["tool_bundle"]
    assert main.nodes[3].color == NODE_COLORS["assistant"]
    assert len({n.color for n in main.nodes}) == 4


def test_tool_node_groups_call_execution_and_response_events() -> None:
    session = ChatSession(
        id="chat-triplet",
        title="Tool Triplet",
        messages=[
            ChatMessage(role="user", content="Verifica mock-tool"),
            ChatMessage(
                role="assistant",
                content="Fatto.",
                tool_events=[
                    ToolEvent(
                        tool_call_id="call_mock_42",
                        tool_name="mock-tool",
                        event_type="tool_end",
                        status="completed",
                        depth=0,
                        skill_path="/",
                        duration_seconds=0.45,
                    )
                ],
                raw_history=[
                    {"role": "system", "content": "Real System Prompt from Provider"},
                    {"role": "user", "content": "Verifica mock-tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_mock_42",
                                "type": "function",
                                "function": {
                                    "name": "mock-tool",
                                    "arguments": {"query": "test"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "mock-tool",
                        "tool_call_id": "call_mock_42",
                        "content": '{"ok": true}',
                    },
                    {"role": "assistant", "content": "Fatto."},
                ],
            ),
        ],
    )
    graph = build_graph_data_from_session(session)
    main = graph.histories[0]
    assert main.nodes[0].kind == "system"
    assert main.nodes[0].events[0]["content"] == "Real System Prompt from Provider"
    assert main.nodes[1].kind == "user"
    assert main.nodes[1].events[0]["content"] == "Verifica mock-tool"

    tool_node = main.nodes[2]
    assert tool_node.kind == "tool_bundle"
    assert len(tool_node.events) == 3
    stages = [ev["event_stage"] for ev in tool_node.events]
    assert stages == [
        "tool_call",
        "tool_execution",
        "tool_response",
    ]
    assert tool_node.events[0]["tool_name"] == "mock-tool"
    assert tool_node.events[1]["duration_seconds"] == 0.45
    assert tool_node.events[2]["content"] == '{"ok": true}'


def test_closed_skill_nodes_remain_with_counterclockwise_offset() -> None:
    nested_sub = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="completed",
        start_time=3,
        tool_events=[
            ToolEvent(
                tool_name="nested-sub-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
                timestamp=4,
            )
        ],
    )
    root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        start_time=1,
        tool_events=[
            ToolEvent(
                tool_name="mock-tool", depth=1, skill_path="/mock-skill", timestamp=2
            )
        ],
        sub_skills=[nested_sub],
    )
    session = ChatSession(
        id="chat-skill-ccw",
        title="Skill Offset Antiorario",
        messages=[
            ChatMessage(role="user", content="Usa skill"),
            ChatMessage(
                role="assistant",
                content="Skill chiusa e completata.",
                skills=[root_skill],
            ),
        ],
    )
    graph = build_graph_data_from_session(session)
    main = graph.histories[0]

    # System (depth 0), User (depth 0), mock-skill (depth 1), mock-tool (depth 1),
    # nested-sub-skill (depth 2), nested-sub-tool (depth 2), Assistant (depth 0)
    assert [n.depth for n in main.nodes] == [0, 0, 1, 1, 2, 2, 0]

    sys_node = main.nodes[0]
    user_node = main.nodes[1]
    depth1_node = main.nodes[2]
    depth1_tool_node = main.nodes[3]
    depth2_node = main.nodes[4]
    depth2_tool_node = main.nodes[5]
    final_assistant_node = main.nodes[-1]

    # Main history is horizontal at y = 225.0.
    # Counterclockwise step on the main history goes 1 step DOWNWARD (in basso, y > 225.0).
    assert sys_node.y == 225.0
    assert user_node.y == 225.0
    assert final_assistant_node.y == 225.0
    assert depth1_node.y > user_node.y
    assert depth2_node.y > depth1_node.y
    # All nodes in the same skill_path have the exact same offset
    assert depth1_node.y == depth1_tool_node.y
    assert depth2_node.y == depth2_tool_node.y
    assert len(main.skill_branches) == 1
    assert main.skill_branches[0].skill_name == "mock-skill"


def test_multiple_histories_intersect_at_interaction_points_forming_constellation() -> (
    None
):
    session = ChatSession(
        id="chat-constellation",
        title="Costellazione Sub-Agent",
        messages=[
            ChatMessage(role="user", content="Delega ai sub-agent"),
            ChatMessage(role="assistant", content="Tutti i sub-agent hanno risposto."),
        ],
        sub_agent_conversations=[
            SubAgentConversation(
                id="sub-1",
                path="main/coding-agent",
                order=1,
                status="completed",
                messages=[
                    ChatMessage(role="user", content="Task 1"),
                    ChatMessage(role="assistant", content="Done 1"),
                ],
            ),
            SubAgentConversation(
                id="sub-2",
                path="main/research-agent",
                order=1,
                status="completed",
                messages=[
                    ChatMessage(role="user", content="Task 2"),
                    ChatMessage(role="assistant", content="Done 2"),
                ],
            ),
            SubAgentConversation(
                id="sub-3",
                path="main/verifier-agent",
                order=1,
                status="completed",
                messages=[
                    ChatMessage(role="assistant", content="Done 3"),
                ],
            ),
        ],
    )
    graph = build_graph_data_from_session(session)
    assert len(graph.histories) == 4
    main = graph.histories[0]
    # Main history is strictly horizontal (0 deg)
    assert main.angle_deg == 0.0

    # Sub-agent coding-agent has its own real HDS system prompt in a separate 'system' node
    coding_sub_line = graph.histories[1]
    assert coding_sub_line.nodes[0].kind == "system"
    assert "coding agent" in coding_sub_line.nodes[0].events[0]["content"].lower()
    assert coding_sub_line.nodes[1].kind == "user"

    # Interaction nodes on the main history where each sub-agent is called
    interaction_nodes = [n for n in main.nodes if n.kind == "sub_agent_bundle"]
    assert len(interaction_nodes) == 3

    # Each sub-agent history line intersects the parent history at its exact point of interaction
    for idx, sub_line in enumerate(graph.histories[1:]):
        inter_node = interaction_nodes[idx]
        assert math.isclose(sub_line.interaction_x, inter_node.x, abs_tol=0.1)
        assert math.isclose(sub_line.interaction_y, inter_node.y, abs_tol=0.1)
        # Verify point (interaction_x, interaction_y) lies directly on the straight line (x1,y1)-(x2,y2)
        cross = (sub_line.x2 - sub_line.x1) * (sub_line.interaction_y - sub_line.y1) - (
            sub_line.y2 - sub_line.y1
        ) * (sub_line.interaction_x - sub_line.x1)
        line_len = math.hypot(sub_line.x2 - sub_line.x1, sub_line.y2 - sub_line.y1)
        assert abs(cross) / line_len < 0.05

    # Sub-agent lines ordered by interaction_x have monotonically decreasing angles so they NEVER cross each other
    sub_lines_sorted = sorted(graph.histories[1:], key=lambda h: h.interaction_x)
    for left_h, right_h in zip(sub_lines_sorted, sub_lines_sorted[1:], strict=False):
        assert left_h.interaction_x < right_h.interaction_x
        assert left_h.angle_deg > right_h.angle_deg
        # Even at their outer tips (x2), the left sub-agent stays strictly to the left of the right sub-agent
        assert left_h.x2 < right_h.x2

    svg_str = render_star_svg_html(graph)
    assert 'stroke-dasharray="5,5"' in svg_str
    assert 'class="graph-interaction-point"' in svg_str
    assert "graph-sub-parallel-out" in svg_str
    assert "graph-sub-parallel-in" in svg_str
    assert "graph-sub-arrow-out" in svg_str
    assert "graph-sub-arrow-in" in svg_str


def test_raw_skill_history_tree_extraction_and_loading_state_before_completion() -> (
    None
):
    session_a = ChatSession(
        id="chat-a",
        title="Chat A",
        messages=[
            ChatMessage(role="user", content="Domanda A"),
            ChatMessage(
                role="assistant",
                content="Risposta A",
                reasoning_blocks=[
                    ReasoningBlock(
                        skill_path="/",
                        depth=0,
                        generation_index=1,
                        content="Pensiero iniziale",
                    )
                ],
                raw_history=[
                    {"role": "system", "content": "Real Main System Prompt A"},
                    {"role": "user", "content": "Domanda A"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_sk_1",
                                "type": "function",
                                "function": {"name": "deep-thinking", "arguments": {}},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "deep-thinking",
                        "tool_call_id": "call_sk_1",
                        "content": "Output skill",
                    },
                    {"role": "assistant", "content": "Risposta A"},
                ],
                skill_history_tree=[
                    {
                        "skill": "deep-thinking",
                        "opening_tool_call_id": "call_sk_1",
                        "output": "Output skill",
                        "history": [
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_sk_1",
                                        "type": "function",
                                        "function": {
                                            "name": "deep-thinking",
                                            "arguments": {},
                                        },
                                    }
                                ],
                            },
                            {"role": "tool", "content": "Catalog deep-thinking"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_inner_web",
                                        "type": "function",
                                        "function": {
                                            "name": "search-web",
                                            "arguments": {"query": "news"},
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "name": "search-web",
                                "tool_call_id": "call_inner_web",
                                "content": '{"results": []}',
                            },
                        ],
                        "children": [],
                    }
                ],
            ),
        ],
    )
    session_b = ChatSession(
        id="chat-b",
        title="Chat B",
        messages=[ChatMessage(role="user", content="Solo un messaggio in B")],
    )

    comp = GraphViewComponent()
    with ui.column():
        comp.build()

        # 1. While response is being generated, show loading animation and do not build graph yet
        comp.render_session(session_a, is_generating=True)
        assert comp.is_generating is True
        assert comp.current_graph_data is None
        assert "Generating response" in (
            comp.status_label.text if comp.status_label else ""
        )

        # 2. Once response finishes (is_generating=False), build the complete constellation graph
        comp.render_session(session_a, is_generating=False)
        assert comp.is_generating is False
        assert comp.current_graph_data is not None
        # System + User + Skill: deep-thinking (depth 1) + Tool: search-web (depth 1) + Assistant (depth 0)
        assert comp.current_graph_data.total_nodes == 5
        final_node = comp.current_graph_data.histories[0].nodes[-1]
        assert final_node.kind == "assistant"
        assert len(final_node.events) == 1
        assert final_node.events[0]["role"] == "assistant"
        assert all(ev.get("role") != "reasoning" for ev in final_node.events)
        skill_nodes = [
            n for n in comp.current_graph_data.histories[0].nodes if n.depth > 0
        ]
        assert len(skill_nodes) == 2
        assert all(n.y > 225.0 for n in skill_nodes)
        assert len({n.y for n in skill_nodes}) == 1

        # Select a node in Chat A
        target_id = skill_nodes[1].id
        comp.select_node(target_id)
        assert comp.get_selected_node_id("chat-a") == target_id

        # Switch to Chat B -> independent graph (System + User = 2 nodes) and independent selection
        comp.render_session(session_b)
        assert comp.current_graph_data.session_id == "chat-b"
        assert comp.current_graph_data.total_nodes == 2
        assert comp.get_selected_node_id("chat-b") is None
        assert comp.get_selected_node_id("chat-a") == target_id

        svg_str = render_star_svg_html(comp.current_graph_data)
        assert "graph-star-svg" in svg_str
        assert "MAIN HISTORY" in svg_str


def test_graph_helpers_and_extraction_edge_cases() -> None:
    # 1. load_real_hds_system_prompt with non-existent HDS and non-existent capabilities path returns empty string
    assert (
        load_real_hds_system_prompt(
            capabilities_path_str="C:/non-existent-caps-dir-12345",
            name_hds="non-existent-hds-xyz",
        )
        == ""
    )

    # 2. No inner messages when only the opening call and tool header exist
    assert inner_skill_history_messages([{"role": "assistant"}, {"role": "tool"}]) == []

    # 3. _build_tool_triplet_events with non-empty assistant_content and tool_event=None
    triplet = _build_tool_triplet_events(
        tool_name="search-web",
        tool_call_id="tc_99",
        arguments=None,
        assistant_content="Chiamo search-web ora",
        tool_event=None,
        tool_response_content="OK",
        skill_path="/",
        depth=0,
    )
    assert triplet[0]["content"] == "Chiamo search-web ora"
    assert triplet[1]["status"] == "completed"

    # 4. _extract_nodes_from_raw_scope edge cases:
    # - assistant without tool_calls
    # - _match_tool_event matching by tool_call_id, then skipping used_tool_event_ids, then matching by tool_name, then returning None
    # - matching skill_tree by skill name when opening_tool_call_id does not match
    # - matching structured_skills (elif matched_struct_skill is not None)
    # - call_sub_agent and exit_agent tool calls
    ev_by_id = ToolEvent(id="ev1", tool_call_id="tc_1", tool_name="tool-a")
    ev_by_name = ToolEvent(id="ev2", tool_call_id="other_id", tool_name="tool-b")
    ev_unused = ToolEvent(id="ev3", tool_call_id="unused_id", tool_name="unused-tool")
    struct_skill = SkillNode(
        name="struct-only-skill", path="/struct-only-skill", depth=1
    )

    raw_slice = [
        {
            "role": "user",
            "content": "Non-assistant item skipped by _extract_nodes_from_raw_scope",
        },
        {"role": "assistant", "content": "No tools here"},
        {
            "role": "assistant",
            "content": "Calling multiple tools",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "tool-a", "arguments": {}},
                },
                {
                    "id": "tc_2",
                    "type": "function",
                    "function": {"name": "tool-b", "arguments": {}},
                },
                {
                    "id": "tc_3",
                    "type": "function",
                    "function": {"name": "unmatched-tool", "arguments": {}},
                },
                {
                    "id": "tc_4",
                    "type": "function",
                    "function": {"name": "skill-by-name", "arguments": {}},
                },
                {
                    "id": "tc_5",
                    "type": "function",
                    "function": {"name": "struct-only-skill", "arguments": {}},
                },
                {
                    "id": "tc_6",
                    "type": "function",
                    "function": {
                        "name": "call_sub_agent",
                        "arguments": {"sub_agent": "main/coding-agent"},
                    },
                },
                {
                    "id": "tc_7",
                    "type": "function",
                    "function": {"name": "exit_agent", "arguments": {}},
                },
            ],
        },
        {"role": "tool", "content": "resp 1"},
        {"role": "tool", "content": "resp 2"},
        {"role": "tool", "tool_call_id": "tc_unmatched", "content": "resp with id"},
    ]
    nodes: list[GraphNode] = []
    _extract_nodes_from_raw_scope(
        raw_slice,
        skill_tree=[
            {
                "skill": "skill-by-name",
                "opening_tool_call_id": "different_id",
                "history": [],
            }
        ],
        structured_skills=[struct_skill],
        tool_events=[ev_by_id, ev_by_name, ev_unused],
        history_id="main",
        history_label="Main",
        depth=0,
        skill_path="/",
        skill_name=None,
        nodes=nodes,
    )
    kinds = [n.kind for n in nodes]
    assert "sub_agent_bundle" in kinds
    assert "skill_bundle" in kinds
    assert "tool_bundle" in kinds


def test_system_prompt_sources_and_message_builder_branches() -> None:
    # 1. _extract_real_system_prompt_event from fallback_raw_history
    ev_fallback = _extract_real_system_prompt_event(
        [
            ChatMessage(
                role="user",
                content="Hi",
                raw_history=[{"role": "user", "content": "Hi"}],
            )
        ],
        fallback_raw_history=[
            {"role": "user", "content": "Hi"},
            {"role": "developer", "content": "Developer fallback prompt"},
        ],
    )
    assert ev_fallback is not None
    assert ev_fallback["content"] == "Developer fallback prompt"

    # 2. _extract_real_system_prompt_event from ChatMessage(role="system")
    ev_msg_sys = _extract_real_system_prompt_event(
        [ChatMessage(role="system", content="Message level system prompt")],
        capabilities_path_str="C:/non-existent-caps-dir-12345",
        name_hds="non-existent-hds-xyz",
    )
    assert ev_msg_sys is not None
    assert ev_msg_sys["content"] == "Message level system prompt"

    # 3. _extract_real_system_prompt_event from explicit_system_prompt
    ev_explicit = _extract_real_system_prompt_event(
        [ChatMessage(role="user", content="Hello")],
        explicit_system_prompt="Explicit session system prompt",
        capabilities_path_str="C:/non-existent-caps-dir-12345",
        name_hds="non-existent-hds-xyz",
    )
    assert ev_explicit is not None
    assert ev_explicit["content"] == "Explicit session system prompt"

    # 4. _extract_real_system_prompt_event returning None when no source has a prompt
    assert (
        _extract_real_system_prompt_event(
            [ChatMessage(role="user", content="Hello")],
            capabilities_path_str="C:/non-existent-caps-dir-12345",
            name_hds="non-existent-hds-xyz",
        )
        is None
    )

    # 5. _build_nodes_for_messages with empty messages & no fallback -> returns []
    assert _build_nodes_for_messages([], history_id="h", history_label="H") == []

    # 6. _build_nodes_for_messages for sub-agent with fallback_raw_history user message and without user message
    sub_nodes = _build_nodes_for_messages(
        [ChatMessage(role="assistant", content="Sub done")],
        history_id="sub",
        history_label="Sub",
        capabilities_path_str="C:/non-existent-caps-dir-12345",
        name_hds="non-existent-hds-xyz",
        fallback_raw_history=[
            {"role": "assistant", "content": "skip"},
            {"role": "user", "content": "Sub user prompt from raw"},
        ],
        is_sub_agent=True,
    )
    assert sub_nodes[0].kind == "user"
    assert "Sub user prompt from raw" in sub_nodes[0].events[0]["content"]

    sub_nodes_no_raw_user = _build_nodes_for_messages(
        [ChatMessage(role="assistant", content="Sub fallback message content")],
        history_id="sub2",
        history_label="Sub2",
        capabilities_path_str="C:/non-existent-caps-dir-12345",
        name_hds="non-existent-hds-xyz",
        fallback_raw_history=[{"role": "assistant", "content": "only assistant"}],
        is_sub_agent=True,
    )
    assert sub_nodes_no_raw_user[0].kind == "user"
    assert (
        "Sub fallback message content" in sub_nodes_no_raw_user[0].events[0]["content"]
    )

    # 7. _build_nodes_for_messages with system message, multi-turn prev_raw prefix slicing,
    # duplicate tool_events deduplication, skill_path inference, and assistant error state
    multi_msgs = [
        ChatMessage(role="system", content="Sys"),
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(
            role="assistant",
            content="Resp 1",
            raw_history=[
                {"role": "user", "content": "Turn 1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "t1", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "name": "t1", "tool_call_id": "c1", "content": "ok1"},
                {"role": "assistant", "content": "Resp 1"},
            ],
        ),
        ChatMessage(
            role="assistant",
            content="Resp 2",
            raw_history=[
                {"role": "user", "content": "Turn 1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "t1", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "name": "t1", "tool_call_id": "c1", "content": "ok1"},
                {"role": "assistant", "content": "Resp 1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "t2", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "name": "t2", "tool_call_id": "c2", "content": "ok2"},
                {"role": "assistant", "content": "Resp 2"},
            ],
        ),
        ChatMessage(
            role="assistant",
            content="Error turn",
            error="Timeout error",
            tool_events=[
                ToolEvent(
                    id="e1", tool_call_id="dup_id", tool_name="my-skill", skill_path="/"
                ),
                ToolEvent(
                    id="e2", tool_call_id="dup_id", tool_name="my-skill", skill_path="/"
                ),
                ToolEvent(
                    id="e3",
                    tool_call_id="inner_id",
                    tool_name="inner-tool",
                    skill_path="/my-skill",
                ),
            ],
        ),
    ]
    built = _build_nodes_for_messages(multi_msgs, history_id="m", history_label="M")
    assert any(n.kind == "error" for n in built)
    assert any(n.skill_path == "/my-skill" and n.kind == "skill_bundle" for n in built)


def test_layout_constellation_and_component_edge_cases(monkeypatch) -> None:
    # 1. _ensure_interaction_nodes_on_main when a sub_agent_bundle node already exists on main_nodes
    existing_sub = GraphNode(
        id="main_sub_0",
        history_id="main",
        history_label="Main",
        step_index=0,
        kind="sub_agent_bundle",
        label="Sub",
        short_label="Sub",
        color=NODE_COLORS["sub_agent_bundle"],
    )
    sub_conv = SubAgentConversation(
        id="sub-existing", path="main/coding-agent", order=1
    )
    inter_map = _ensure_interaction_nodes_on_main([existing_sub], [sub_conv])
    assert inter_map["sub-existing"] is existing_sub
    assert existing_sub.sub_agent_target_id == "sub-existing"

    # 2. _apply_skill_offsets_and_branches when skill nodes are at index 0 and at the end (j == n)
    line_only_skill = GraphHistoryLine(
        id="skill_only",
        label="Skill Only",
        nodes=[
            GraphNode(
                id="s0",
                history_id="skill_only",
                history_label="Skill Only",
                step_index=0,
                kind="skill_bundle",
                label="Skill",
                short_label="Skill",
                color=NODE_COLORS["skill_bundle"],
                depth=1,
                skill_path="/only-skill",
                skill_name="only-skill",
            )
        ],
    )
    _layout_main_horizontal_history(line_only_skill)
    assert line_only_skill.nodes[0].base_x == 500.0
    assert len(line_only_skill.skill_branches) == 1

    # 3. _layout_sub_agent_constellation_line with 0 nodes and 1 node, and obtuse/vertical angles
    empty_sub = GraphHistoryLine(id="empty_sub", label="Empty Sub", nodes=[])
    _layout_sub_agent_constellation_line(
        empty_sub, anchor_x=500.0, anchor_y=225.0, angle_deg=90.0
    )
    single_sub = GraphHistoryLine(
        id="single_sub",
        label="Single Sub",
        nodes=[existing_sub.model_copy()],
    )
    _layout_sub_agent_constellation_line(
        single_sub, anchor_x=500.0, anchor_y=225.0, angle_deg=135.0
    )
    assert single_sub.nodes[0].base_x < 500.0

    # 4. build_graph_data_from_session with live_message, single sub-agent, and deep nested sub-agents
    session_deep = ChatSession(
        id="deep-constellation",
        title="Deep Constellation",
        messages=[ChatMessage(role="user", content="Start")],
        sub_agent_conversations=[
            SubAgentConversation(
                id="sub-root-left",
                path="main/coding-agent",
                order=1,
                messages=[ChatMessage(role="assistant", content="Root left")],
            ),
            SubAgentConversation(
                id="sub-root-right",
                path="main/research-agent",
                order=2,
                messages=[ChatMessage(role="assistant", content="Root right")],
            ),
            SubAgentConversation(
                id="sub-nested-left",
                path="main/coding-agent/child-a",
                order=3,
                messages=[ChatMessage(role="assistant", content="Nested left")],
            ),
            SubAgentConversation(
                id="sub-nested-right",
                path="main/research-agent/child-b",
                order=4,
                messages=[],
            ),
            SubAgentConversation(
                id="sub-nested-orphan",
                path="main/unknown-parent/child-c",
                order=5,
                messages=[],
            ),
        ],
    )
    graph_deep = build_graph_data_from_session(
        session_deep,
        live_message=ChatMessage(role="assistant", content="Live final"),
    )
    assert graph_deep.find_node(None) is None
    assert graph_deep.find_node("non-existent-id") is None
    assert graph_deep.find_node(graph_deep.histories[0].nodes[0].id) is not None

    # Single sub-agent case (m_total == 1 -> 82.0 deg) and empty branch points in SVG
    session_single_sub = ChatSession(
        id="single-sub-session",
        title="Single Sub",
        messages=[ChatMessage(role="user", content="Hi")],
        sub_agent_conversations=[
            SubAgentConversation(
                id="only-sub",
                path="main/coding-agent",
                order=1,
                messages=[ChatMessage(role="assistant", content="Done")],
            )
        ],
    )
    graph_single = build_graph_data_from_session(session_single_sub)
    assert graph_single.histories[1].angle_deg == 82.0
    graph_single.histories[0].skill_branches.append(
        GraphSkillBranch(
            skill_name="degenerate",
            skill_path="/deg",
            depth=1,
            color="#fff",
            points=[(0.0, 0.0)],
        )
    )
    svg_out = render_star_svg_html(graph_single)
    assert "degenerate" not in svg_out

    # 5. GraphViewComponent uninitialized guard clauses & selection of system, user, and assistant nodes
    unbuilt = GraphViewComponent()
    unbuilt.select_node("any")
    unbuilt.render_session(session_single_sub)
    unbuilt._render_loading_animation(session_single_sub)
    unbuilt._render_stage_and_menu()
    unbuilt._render_bottom_horizontal_menu(None)

    copied_payloads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ui.clipboard, "write", lambda text: copied_payloads.append(("clip", text))
    )
    monkeypatch.setattr(
        ui, "notify", lambda msg, **kw: copied_payloads.append(("notify", msg))
    )

    comp = GraphViewComponent()
    with ui.column():
        comp.build()
        # Render empty session (total_nodes == 0)
        empty_sess = ChatSession(id="empty-s", title="Empty", messages=[])
        comp.render_session(empty_sess)
        assert comp.current_graph_data is not None
        assert comp.current_graph_data.total_nodes == 0

        # Pre-populate a stale selected_node_id that doesn't exist, then render_session
        comp.selected_node_by_session[session_single_sub.id] = "stale-id"
        comp.render_session(session_single_sub)
        assert comp.get_selected_node_id(session_single_sub.id) is None

        # Select system node (depth 0), user node (depth 0), and assistant node (depth 0)
        for node in comp.current_graph_data.histories[0].nodes:
            comp.select_node(node.id)
            assert comp.get_selected_node_id(session_single_sub.id) == node.id

        # Test _copy_json helper
        GraphViewComponent._copy_json('{"ok": true}', "Copiato")
        assert len(copied_payloads) == 2

        # Test status_label=None branch during generating and completed states
        comp.status_label = None
        comp.render_session(session_single_sub, is_generating=True)
        comp.select_node("ignored-while-generating")
        comp.render_session(session_single_sub, is_generating=False)


def test_fixed_node_distance_and_horizontal_scroll_expansion() -> None:
    """Verify that the distance between consecutive nodes on the main history is fixed
    (FIXED_NODE_SPACING_X) regardless of node count, that the horizontal width of the
    graph is not fixed and increases with the number of nodes, and that the stage is
    horizontally scrollable.
    """
    short_session = ChatSession(
        id="short-graph",
        title="Short Graph",
        messages=[ChatMessage(role="user", content="Domanda breve")],
    )
    long_tool_events = [
        ToolEvent(
            id=f"ev_{i}",
            tool_call_id=f"call_{i}",
            tool_name=f"tool-step-{i}",
            status="completed",
            depth=0,
            skill_path="/",
        )
        for i in range(14)
    ]
    long_session = ChatSession(
        id="long-graph",
        title="Long Scrollable Graph",
        messages=[
            ChatMessage(role="user", content="Esegui molti tool in sequenza"),
            ChatMessage(
                role="assistant",
                content="Completati tutti i 14 tool.",
                tool_events=long_tool_events,
            ),
        ],
    )

    short_graph = build_graph_data_from_session(short_session)
    long_graph = build_graph_data_from_session(long_session)

    short_nodes = short_graph.histories[0].nodes
    long_nodes = long_graph.histories[0].nodes
    assert len(short_nodes) == 2
    assert len(long_nodes) == 17

    # 1. Fixed node-to-node horizontal distance across both short and long sessions
    for prev_n, next_n in zip(short_nodes, short_nodes[1:], strict=False):
        assert math.isclose(
            next_n.base_x - prev_n.base_x, FIXED_NODE_SPACING_X, abs_tol=0.05
        )

    for prev_n, next_n in zip(long_nodes, long_nodes[1:], strict=False):
        assert math.isclose(
            next_n.base_x - prev_n.base_x, FIXED_NODE_SPACING_X, abs_tol=0.05
        )

    # 2. Horizontal size (graph.width and main line length) is NOT fixed and grows as distance increases
    short_line_span = short_graph.histories[0].x2 - short_graph.histories[0].x1
    long_line_span = long_graph.histories[0].x2 - long_graph.histories[0].x1
    assert long_line_span > short_line_span
    assert long_graph.width > short_graph.width
    assert long_graph.width >= (len(long_nodes) - 1) * FIXED_NODE_SPACING_X

    # 3. Stage container is horizontally scrollable and canvas matches graph_data.width in pixels
    comp = GraphViewComponent()
    with ui.column():
        comp.build()
        assert comp.stage_container is not None
        assert "overflow-x-auto" in comp.stage_container.classes
        comp.render_session(long_session)
        svg_str = render_star_svg_html(long_graph)
        assert (
            f'viewBox="0 0 {int(long_graph.width)} {int(long_graph.height)}"' in svg_str
        )
        assert f'width="{int(long_graph.width)}"' in svg_str


def test_sub_agent_branches_non_crossing_and_collision_resolution() -> None:
    """Verify segment crossing detection, endpoint sharing, and non-crossing collision resolution."""
    # 1. Crossing line segments
    assert _segments_cross((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)) is True

    # 2. Parallel / non-intersecting line segments
    assert _segments_cross((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0)) is False

    # 3. Segments sharing an attachment endpoint (e.g., child connected to parent)
    assert _segments_cross((5.0, 5.0), (10.0, 10.0), (5.0, 5.0), (0.0, 10.0)) is False

    # 4. Create colliding lines and verify _ensure_sub_agent_branches_do_not_cross resolves them
    line_a = GraphHistoryLine(
        id="sub_a",
        label="Sub A",
        interaction_x=200.0,
        interaction_y=225.0,
        angle_deg=45.0,
        x1=200.0,
        y1=225.0,
        x2=400.0,
        y2=25.0,
        nodes=[
            GraphNode(
                id="na",
                history_id="sub_a",
                history_label="A",
                step_index=0,
                kind="user",
                label="A",
                short_label="A",
                color="#fff",
            )
        ],
    )
    line_b = GraphHistoryLine(
        id="sub_b",
        label="Sub B",
        parent_history_id="sub_a",
        interaction_x=400.0,
        interaction_y=225.0,
        angle_deg=135.0,
        x1=400.0,
        y1=225.0,
        x2=200.0,
        y2=25.0,
        nodes=[
            GraphNode(
                id="nb",
                history_id="sub_b",
                history_label="B",
                step_index=0,
                kind="user",
                label="B",
                short_label="B",
                color="#fff",
            )
        ],
    )
    test_histories = [line_a, line_b]
    assert len(check_sub_agent_branches_do_not_cross(test_histories)) == 1

    # Apply resolution
    _ensure_sub_agent_branches_do_not_cross(test_histories)
    assert check_sub_agent_branches_do_not_cross(test_histories) == []

    # Also test resolution when target has angle < 90.0
    line_c = GraphHistoryLine(
        id="sub_c",
        label="Sub C",
        parent_history_id="main",
        interaction_x=200.0,
        interaction_y=225.0,
        angle_deg=45.0,
        x1=200.0,
        y1=225.0,
        x2=400.0,
        y2=25.0,
        nodes=[
            GraphNode(
                id="nc",
                history_id="sub_c",
                history_label="C",
                step_index=0,
                kind="user",
                label="C",
                short_label="C",
                color="#fff",
            )
        ],
    )
    line_d = GraphHistoryLine(
        id="sub_d",
        label="Sub D",
        parent_history_id="main",
        interaction_x=400.0,
        interaction_y=225.0,
        angle_deg=85.0,
        x1=400.0,
        y1=225.0,
        x2=200.0,
        y2=25.0,
        nodes=[
            GraphNode(
                id="nd",
                history_id="sub_d",
                history_label="D",
                step_index=0,
                kind="user",
                label="D",
                short_label="D",
                color="#fff",
            )
        ],
    )
    test_histories_cd = [line_c, line_d]
    assert len(check_sub_agent_branches_do_not_cross(test_histories_cd)) == 1
    _ensure_sub_agent_branches_do_not_cross(test_histories_cd)
    assert check_sub_agent_branches_do_not_cross(test_histories_cd) == []


def test_vertical_fixed_node_distance_and_dynamic_canvas_height() -> None:
    """Verify that nodes along sub-agent rays have fixed Euclidean distance, that the canvas
    height grows dynamically with many nodes and deep skills, and that the stage container
    enables vertical scrolling.
    """
    sub_skill = SkillNode(
        name="inner-sub-skill",
        path="/inner-sub-skill",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(tool_name="sub-tool", depth=1, skill_path="/inner-sub-skill")
        ],
    )
    sub_messages = [
        ChatMessage(role="user", content=f"Step {i}")
        if i % 2 == 0
        else ChatMessage(
            role="assistant", content=f"Resp {i}", skills=[sub_skill] if i == 1 else []
        )
        for i in range(12)
    ]
    session_tall = ChatSession(
        id="tall-sub-session",
        title="Tall Sub Agent Constellation",
        messages=[ChatMessage(role="user", content="Start tall test")],
        sub_agent_conversations=[
            SubAgentConversation(
                id="tall-sub",
                path="main/coding-agent",
                order=1,
                messages=sub_messages,
            )
        ],
    )

    graph_tall = build_graph_data_from_session(session_tall)
    sub_line = graph_tall.histories[1]
    assert len(sub_line.nodes) >= 12

    # 1. Fixed distance between consecutive nodes along the sub-agent ray
    for prev_n, next_n in zip(sub_line.nodes, sub_line.nodes[1:], strict=False):
        # Ignore cross-skill jump between depth 0 and depth 1, check same-depth nodes
        if prev_n.depth == next_n.depth:
            dist = math.hypot(
                next_n.base_x - prev_n.base_x, next_n.base_y - prev_n.base_y
            )
            assert math.isclose(dist, FIXED_SUB_AGENT_NODE_SPACING, abs_tol=0.1)

    # 2. Canvas height grew dynamically beyond the default 500px to accommodate tall ray and skill branch
    assert graph_tall.height > 500.0
    assert len(sub_line.skill_branches) >= 1

    # 3. Sub-agent branches never cross
    assert check_sub_agent_branches_do_not_cross(graph_tall.histories) == []

    # 4. GraphViewComponent stage has both overflow-x-auto and overflow-y-auto
    comp = GraphViewComponent()
    with ui.column():
        comp.build()
        assert comp.stage_container is not None
        assert "overflow-x-auto" in comp.stage_container.classes
        assert "overflow-y-auto" in comp.stage_container.classes
        comp.render_session(session_tall)
        svg_str = render_star_svg_html(graph_tall)
        assert (
            f'viewBox="0 0 {int(graph_tall.width)} {int(graph_tall.height)}"' in svg_str
        )
        assert f'height="{int(graph_tall.height)}"' in svg_str


def test_graph_view_renders_nodes_with_windows_paths_and_unicode_escapes() -> None:
    r"""Verify that node labels containing Windows backslash paths (e.g. C:\Users...)
    or quotes do not trigger ast.literal_eval unicode decode errors.
    """
    session = ChatSession(
        id="session-windows-paths",
        title="Windows Paths Session",
        messages=[
            ChatMessage(
                role="user",
                content=r"Analyze file in C:\Users\danie\Desktop\exmekh_lib\ExMekh Agent",
            ),
            ChatMessage(
                role="assistant",
                content=r"Nella directory `C:\Users\danie\Desktop\exmekh_lib\ExMekh Agent` ho trovato i file.",
            ),
        ],
    )
    comp = GraphViewComponent()
    with ui.column():
        comp.build()
        comp.render_session(session)
        assert comp.current_graph_data is not None
        assert comp.current_graph_data.total_nodes >= 2
        for node in comp.current_graph_data.histories[0].nodes:
            comp.select_node(node.id)
            assert comp.get_selected_node_id(session.id) == node.id


def test_graph_view_renders_steering_node_with_user_message() -> None:
    raw_history = [
        {"role": "system", "content": "System Prompt Content"},
        {"role": "user", "content": "Start prompt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "bash", "arguments": {"cmd": "ls"}},
                }
            ],
        },
        {
            "role": "tool",
            "name": "bash",
            "tool_call_id": "call-1",
            "content": "file1.txt",
        },
        {"role": "user", "content": "Change directory to src and search again"},
        {"role": "assistant", "content": "Here are the files in src."},
    ]
    session = ChatSession(
        id="session-steering-test",
        title="Steering Test Session",
        messages=[
            ChatMessage(role="user", content="Start prompt"),
            ChatMessage(
                role="assistant",
                content="Here are the files in src.",
                raw_history=raw_history,
                raw_history_start=2,
                steering_messages=[
                    QueuedMessage(
                        text="Change directory to src and search again",
                        display_content="Steering instruction",
                    )
                ],
            ),
        ],
    )
    graph_data = build_graph_data_from_session(session)
    main = graph_data.histories[0]
    node_kinds = [n.kind for n in main.nodes]
    assert node_kinds == [
        "system",
        "user",
        "tool_bundle",
        "steering",
        "assistant",
    ]

    steering_node = main.nodes[3]
    assert steering_node.kind == "steering"
    assert steering_node.color == NODE_COLORS["steering"]
    assert steering_node.color == "#fb923c"
    assert "Steering: Change directory to" in steering_node.label
    assert steering_node.short_label == "Steering"
    assert steering_node.depth == 0
    assert steering_node.skill_path == "/"
    assert len(steering_node.events) == 1

    ev = steering_node.events[0]
    assert ev["role"] == "user"
    assert ev["event_stage"] == "steering_message"
    assert ev["stage_label"] == "Steering Message"
    assert ev["content"] == "Change directory to src and search again"
    assert ev["display_content"] == "Steering instruction"
    assert ev["message_id"] == session.messages[1].steering_messages[0].id

    # Test rendering in GraphViewComponent and selecting steering node
    comp = GraphViewComponent()
    with ui.column():
        comp.build()
        comp.render_session(session)
        comp.select_node(steering_node.id)
        assert comp.get_selected_node_id(session.id) == steering_node.id


def test_graph_view_renders_steering_fallback_without_raw_history() -> None:
    session = ChatSession(
        id="session-steering-fallback",
        title="Steering Fallback Session",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(
                role="assistant",
                content="Done",
                tool_events=[ToolEvent(tool_name="web_search", tool_call_id="call-ws")],
                steering_messages=[
                    QueuedMessage(
                        text="Focus on python 3.13",
                        display_content="Focus on 3.13 preview",
                    ),
                    QueuedMessage(
                        text="Focus on python 3.13",
                        display_content="Second message with the same text",
                    ),
                    QueuedMessage(
                        text="Third text without display",
                    ),
                ],
            ),
        ],
    )
    graph_data = build_graph_data_from_session(session)
    main = graph_data.histories[0]
    assert [n.kind for n in main.nodes] == [
        "system",
        "user",
        "tool_bundle",
        "steering",
        "steering",
        "steering",
        "assistant",
    ]
    st_node = next(n for n in main.nodes if n.kind == "steering")
    assert st_node.events[0]["content"] == "Focus on python 3.13"
    assert st_node.events[0]["display_content"] == "Focus on 3.13 preview"


def test_graph_data_raw_scope_steering_with_message_id_and_display_content() -> None:
    raw_slice = [
        {"role": "unknown_role", "content": "ignored"},
        {"role": "user", "content": "Inspect this code", "message_id": "mid-99"},
        {"role": "user", "content": "Second guidance with no display"},
    ]
    nodes: list[GraphNode] = []
    qm1 = QueuedMessage(text="Different guidance", display_content="Diff")
    qm2 = QueuedMessage(text="Inspect this code", display_content="Display preview")
    qm3 = QueuedMessage(text="Second guidance with no display", display_content=None)
    _extract_nodes_from_raw_scope(
        raw_slice,
        skill_tree=[],
        structured_skills=[],
        tool_events=[],
        history_id="test_hist",
        history_label="Test",
        depth=0,
        skill_path="/",
        skill_name=None,
        nodes=nodes,
        message=ChatMessage(role="assistant", steering_messages=[qm1, qm2, qm3]),
    )
    assert len(nodes) == 2
    assert nodes[0].kind == "steering"
    assert nodes[0].events[0]["display_content"] == "Display preview"
    assert nodes[0].events[0]["timestamp"] == qm2.created_at
    assert "display_content" not in nodes[1].events[0]

    # Also test without matched_sm
    raw_slice_no_match = [
        {"role": "user", "content": "Other guidance", "message_id": "mid-100"},
    ]
    nodes_no_match: list[GraphNode] = []
    _extract_nodes_from_raw_scope(
        raw_slice_no_match,
        skill_tree=[],
        structured_skills=[],
        tool_events=[],
        history_id="test_hist",
        history_label="Test",
        depth=0,
        skill_path="/",
        skill_name=None,
        nodes=nodes_no_match,
        message=ChatMessage(role="assistant", steering_messages=[qm1]),
    )
    assert len(nodes_no_match) == 1
    assert nodes_no_match[0].events[0]["message_id"] == "mid-100"
