"""Persist one visible conversation per child call or persistent child node."""

from __future__ import annotations

from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEvent
from exmekh_agent.exmekh_bridge.models import ChatSession, SubAgentConversation


def apply_sub_agent_event(
    session: ChatSession, event: SubAgentEvent
) -> SubAgentConversation:
    if event.action == "started":
        return session.start_sub_agent_conversation(
            conversation_id=event.id,
            path=event.path,
            persistent_history=event.persistent_history,
            user_message=event.user_message,
        )

    conversation = session.find_sub_agent_conversation(event.id)
    if conversation is None:
        raise ValueError(f"Unknown sub-agent conversation: {event.id}")

    if event.action == "token":
        conversation.append_token(event.token)
    elif event.action == "reasoning":
        conversation.append_reasoning(event.token, event.skill_path, event.depth)
    elif event.action == "tool" and event.tool_event:
        conversation.add_tool_event(event.tool_event)
    elif event.action == "completed":
        conversation.complete_call(event.response, event.raw_history)
    elif event.action == "error":
        conversation.fail_call()
    return conversation
