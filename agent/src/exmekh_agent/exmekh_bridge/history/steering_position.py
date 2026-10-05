"""Locate accepted steering messages in the actual ExMekh-Lib execution history."""

from exmekh_agent.exmekh_bridge.models import ChatMessage, SteeringInsertion


def find_steering_insertions(message: ChatMessage) -> list[SteeringInsertion]:
    """Compatibility entry point; the message owns confirmation and reconstruction."""
    return message.confirmed_steering_insertions()
