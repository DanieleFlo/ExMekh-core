"""Locate accepted steering messages in the actual QS execution history."""

from fe.qs_bridge.models import ChatMessage, SteeringInsertion


def find_steering_insertions(message: ChatMessage) -> list[SteeringInsertion]:
    """Compatibility entry point; the message owns confirmation and reconstruction."""
    return message.confirmed_steering_insertions()
