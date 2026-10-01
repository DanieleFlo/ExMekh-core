from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from fe.qs_bridge.models import ChatMessage
from fe.state.session_manager import session_manager
from fe.ui.page_controller import ChatPageController
from fe.ui.sidebar import SidebarComponent


@pytest.mark.parametrize("has_messages", [False, True])
def test_new_quick_chat_notifies_after_deleting_the_button_parent(
    has_messages: bool,
) -> None:
    client = Client(page("/test-new-quick-chat"))
    try:
        with client:
            previous = session_manager.get_current_session()
            if has_messages:
                session_manager.add_message(
                    previous, ChatMessage(role="user", content="Keep this conversation")
                )
            controller = ChatPageController(client)
            sidebar = SidebarComponent(controller.session_switched, controller.new_chat)
            controller.sidebar = sidebar
            sidebar.build()

        for _ in range(2):
            button = next(
                element
                for element in client.elements.values()
                if isinstance(element, ui.button)
                and element._props.get("aria-label") == "New Quick Chat"
            )
            sidebar.selected_sub_agent_id = "previous-sub-agent"
            # NiceGUI runs click handlers in the sender's parent slot.
            with button.parent_slot:
                sidebar._handle_new_chat()

            assert button.is_deleted
            assert sidebar.selected_sub_agent_id is None
            target_id, message_type, payload = client.outbox.messages[-1]
            assert target_id == client.id
            assert message_type == "notify"
            assert payload["message"] == "Conversation ready"
            assert payload["type"] == "positive"
            ready = session_manager.get_current_session()
            assert ready.project_id is None
            assert not ready.messages
            if has_messages:
                assert ready is not previous
                assert previous.messages[0].content == "Keep this conversation"
            else:
                assert ready is previous
    finally:
        client.delete()
