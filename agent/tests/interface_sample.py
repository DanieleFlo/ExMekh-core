from __future__ import annotations

from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)


def interface_session() -> ChatSession:
    tool = ToolEvent(
        tool_name="bash",
        tool_call_id="root-call",
        arguments={
            "command": "echo <hello>\n" + "argomento lungo " * 80,
            "timeout": 10,
        },
    )
    nested_tool = ToolEvent(
        tool_name="read",
        skill_path="/worker",
        depth=1,
        arguments={"file": "nested-private.txt"},
    )
    nested_reasoning = ReasoningBlock(
        content="Nested reasoning", depth=1, skill_path="/worker"
    )
    skill = SkillNode(
        name="worker",
        path="/worker",
        depth=1,
        status="completed",
        reasoning_blocks=[nested_reasoning],
        tool_events=[nested_tool],
        sub_skills=[SkillNode(name="nested", path="/worker/nested", depth=2)],
    )
    return ChatSession(
        title="Interfaccia · test",
        system_prompt="System instructions",
        messages=[
            ChatMessage(
                role="user",
                content="Internal attachment markup",
                display_content="Primo messaggio **utente**",
            ),
            ChatMessage(
                role="assistant",
                content="Prima risposta",
                reasoning_content="Root reasoning\nNested reasoning",
                reasoning_blocks=[
                    ReasoningBlock(content="Root reasoning"),
                    nested_reasoning,
                ],
                tool_events=[tool, nested_tool],
                skills=[skill],
                steering_messages=[
                    QueuedMessage(
                        text="Steering text", display_content="Visible steering"
                    )
                ],
                skill_history_tree=[
                    {
                        "skill": "worker",
                        "history": [
                            {"role": "assistant", "content": "Nested transcript ```"}
                        ],
                        "children": [],
                    }
                ],
            ),
            ChatMessage(role="user", content="Secondo messaggio"),
            ChatMessage(
                role="assistant",
                content="Seconda risposta",
                reasoning_content="Legacy reasoning",
                error="cancelled",
            ),
        ],
        sub_agent_conversations=[
            SubAgentConversation(
                id="child",
                path="main/worker",
                status="completed",
                messages=[
                    ChatMessage(role="user", content="Child prompt"),
                    ChatMessage(role="assistant", content="Child reply"),
                ],
                raw_history=[{"role": "assistant", "content": "Child transcript"}],
            ),
            SubAgentConversation(path="main/empty", status="completed"),
        ],
    )
