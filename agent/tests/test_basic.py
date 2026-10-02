from pathlib import Path

from fe.config import config
from fe.exmekh_bridge.models import ChatMessage, ChatSession, ReasoningBlock, ToolEvent
from fe.state.session_manager import SessionManager
from fe.ui.chat_view import ChatViewComponent
from fe.ui.header import HeaderComponent
from fe.ui.input_bar import InputBarComponent
from fe.ui.settings_drawer import SettingsDrawerComponent


def test_config_defaults() -> None:
    assert config.base_url.startswith("http")
    assert config.default_model != ""
    assert config.port > 0


def test_models_serialization() -> None:
    event = ToolEvent(tool_name="test_tool", depth=1, skill_path="/sub")
    msg = ChatMessage(role="assistant", content="Hello", tool_events=[event])
    session = ChatSession(title="Test", messages=[msg])

    data = session.model_dump()
    loaded = ChatSession.model_validate(data)

    assert loaded.title == "Test"
    assert len(loaded.messages) == 1
    assert loaded.messages[0].tool_events[0].tool_name == "test_tool"


def test_reasoning_blocks_serialization() -> None:
    b1 = ReasoningBlock(
        skill_path="/", depth=0, generation_index=1, content="Planning turn 1"
    )
    b2 = ReasoningBlock(
        skill_path="/mock-skill",
        depth=1,
        generation_index=2,
        content="Inner skill reasoning",
    )
    b3 = ReasoningBlock(
        skill_path="/", depth=0, generation_index=3, content="Final summary reasoning"
    )

    msg = ChatMessage(
        role="assistant",
        content="Result",
        reasoning_content="Planning turn 1\n\nInner skill reasoning\n\nFinal summary reasoning",
        reasoning_blocks=[b1, b2, b3],
    )
    session = ChatSession(title="Multi-Path Reasoning", messages=[msg])

    data = session.model_dump()
    loaded = ChatSession.model_validate(data)

    assert len(loaded.messages[0].reasoning_blocks) == 3
    assert loaded.messages[0].reasoning_blocks[0].skill_path == "/"
    assert loaded.messages[0].reasoning_blocks[1].skill_path == "/mock-skill"
    assert loaded.messages[0].reasoning_blocks[1].depth == 1
    assert loaded.messages[0].reasoning_blocks[2].generation_index == 3


def test_session_manager(tmp_path: Path) -> None:
    sm = SessionManager(tmp_path)

    sess = sm.create_session("Test Session")
    assert sess.title == "Test Session"
    assert sess.id in sm.sessions

    # Switch
    assert sm.switch_session(sess.id) is not None

    # Delete
    sm.delete_session(sess.id)
    assert sess.id not in sm.sessions


def test_no_euro_or_neurology_icon() -> None:
    ui_dir = Path(__file__).resolve().parents[1] / "src" / "fe" / "ui"
    for py_file in ui_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "neurology" not in content, (
            f"Found 'neurology' (renders euro icon) in {py_file.name}"
        )
        assert "euro" not in content.lower(), (
            f"Found 'euro' icon reference in {py_file.name}"
        )


def test_ui_components_loading_attributes() -> None:
    cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
    assert hasattr(cv, "current_loading_placeholder")
    assert hasattr(cv, "current_live_reasoning_container")

    header = HeaderComponent(
        on_toggle_drawer=lambda: None,
    )
    assert not hasattr(header, "mode_toggle")

    ib = InputBarComponent(on_send=lambda _: None, on_stop=lambda: None)
    assert hasattr(ib, "set_busy")

    sd = SettingsDrawerComponent(on_settings_saved=lambda: None)
    assert hasattr(sd, "save_button")
