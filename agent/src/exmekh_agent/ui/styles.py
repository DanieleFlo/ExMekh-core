"""CSS Styles and layout theming inspired by OpenAI Codex and minimalist CLI studio aesthetic.

Styles are modularized in the `css/` directory and loaded into NiceGUI via `load_css()` or `ui.add_css()`.
"""

from __future__ import annotations

from pathlib import Path

from markupsafe import Markup
from nicegui import ui

from exmekh_agent.ui.resources import presentation_config, render_html

CSS_DIR: Path = Path(__file__).parent / "resources" / "css"

CSS_FILENAMES: tuple[str, ...] = tuple(presentation_config("styles")["files"])

# Backwards compatibility for single-path references
CSS_FILE_PATH: Path = CSS_DIR / "base.css"


def get_css_files() -> list[Path]:
    """Return the list of existing CSS files in defined order."""
    if CSS_DIR.is_dir():
        files: list[Path] = []
        for name in CSS_FILENAMES:
            path = CSS_DIR / name
            if path.is_file():
                files.append(path)
        if files:
            return files
    if CSS_FILE_PATH.is_file():
        return [CSS_FILE_PATH]
    return []


def get_css_content() -> str:
    """Read and return the raw combined CSS rules from stylesheets."""
    files = get_css_files()
    if not files:
        return ""
    return "\n\n".join(file.read_text(encoding="utf-8") for file in files)


def load_css() -> None:
    """Inject custom CSS stylesheets into the current NiceGUI page."""
    files = get_css_files()
    if files:
        for file_path in files:
            ui.add_css(file_path)
    else:
        # Fallback to in-memory string if files not found
        content = get_css_content()
        if content:
            ui.add_css(content)


# Backwards compatibility for components referencing raw HTML style tags
CUSTOM_CSS: str = render_html("shared/styles.html", content=Markup(get_css_content()))
