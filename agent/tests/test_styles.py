from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from exmekh_agent.ui.styles import (
    CSS_DIR,
    CSS_FILE_PATH,
    CSS_FILENAMES,
    CUSTOM_CSS,
    get_css_content,
    get_css_files,
    load_css,
)


def test_styles_file_and_constants() -> None:
    assert CSS_DIR.is_dir()
    assert CSS_FILE_PATH.is_file()
    for name in CSS_FILENAMES:
        file_path = CSS_DIR / name
        assert file_path.is_file(), f"Expected stylesheet missing: {name}"

    assert "<style>" in CUSTOM_CSS
    assert ":root" in CUSTOM_CSS
    content = get_css_content()
    assert ":root" in content
    assert "--bg-main" in content
    assert ".chat-scroll-container" in content
    assert ".graph-stage" in content
    assert ".exmekh-queue-box" in content
    assert ".input-progress-container" in content
    assert ".nicegui-content" in content
    assert ".chat-message-row" in content
    assert ".chat-poi-track" in content
    assert ".chat-poi-dot" in content


def test_get_css_files_fallbacks(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_css"
    empty_dir.mkdir()
    fallback_file = tmp_path / "fallback.css"
    fallback_file.write_text("/* fallback */", encoding="utf-8")
    missing_file = tmp_path / "missing.css"

    # Fallback to CSS_FILE_PATH when CSS_DIR is missing
    with (
        patch("exmekh_agent.ui.styles.CSS_DIR", tmp_path / "nonexistent_dir"),
        patch("exmekh_agent.ui.styles.CSS_FILE_PATH", fallback_file),
    ):
        assert get_css_files() == [fallback_file]

    # Fallback to CSS_FILE_PATH when CSS_DIR exists but has no matching files
    with (
        patch("exmekh_agent.ui.styles.CSS_DIR", empty_dir),
        patch("exmekh_agent.ui.styles.CSS_FILE_PATH", fallback_file),
    ):
        assert get_css_files() == [fallback_file]

    # Return empty list when CSS_DIR is missing and CSS_FILE_PATH does not exist
    with (
        patch("exmekh_agent.ui.styles.CSS_DIR", tmp_path / "nonexistent_dir"),
        patch("exmekh_agent.ui.styles.CSS_FILE_PATH", missing_file),
    ):
        assert get_css_files() == []

    # Return empty list when CSS_DIR exists without files and CSS_FILE_PATH does not exist
    with (
        patch("exmekh_agent.ui.styles.CSS_DIR", empty_dir),
        patch("exmekh_agent.ui.styles.CSS_FILE_PATH", missing_file),
    ):
        assert get_css_files() == []


def test_get_css_content_missing_file() -> None:
    with (
        patch("exmekh_agent.ui.styles.CSS_DIR", Path("nonexistent_dir")),
        patch("exmekh_agent.ui.styles.CSS_FILE_PATH", Path("nonexistent_styles.css")),
    ):
        assert get_css_content() == ""


def test_load_css_success() -> None:
    with patch("exmekh_agent.ui.styles.ui.add_css") as mock_add_css:
        load_css()
        files = get_css_files()
        assert mock_add_css.call_count == len(files)
        for f in files:
            mock_add_css.assert_any_call(f)


def test_load_css_fallbacks() -> None:
    with (
        patch("exmekh_agent.ui.styles.get_css_files", return_value=[]),
        patch(
            "exmekh_agent.ui.styles.get_css_content", return_value="/* custom css */"
        ),
        patch("exmekh_agent.ui.styles.ui.add_css") as mock_add_css,
    ):
        load_css()
        mock_add_css.assert_called_once_with("/* custom css */")

    with (
        patch("exmekh_agent.ui.styles.get_css_files", return_value=[]),
        patch("exmekh_agent.ui.styles.get_css_content", return_value=""),
        patch("exmekh_agent.ui.styles.ui.add_css") as mock_add_css,
    ):
        load_css()
        mock_add_css.assert_not_called()
