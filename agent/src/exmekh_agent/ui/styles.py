"""CSS Styles and layout theming inspired by OpenAI Codex and minimalist CLI studio aesthetic.

Styles are modularized in the `css/` directory and loaded into NiceGUI via `load_css()` or `ui.add_css()`.
"""

from __future__ import annotations

from pathlib import Path

from markupsafe import Markup
from nicegui import ui
from nicegui.client import Client

from exmekh_agent.ui.resources import presentation_config, render_html, ui_text

CSS_DIR: Path = Path(__file__).parent / "resources" / "css"

CSS_FILENAMES: tuple[str, ...] = tuple(presentation_config("styles")["files"])

# Backwards compatibility for single-path references
CSS_FILE_PATH: Path = CSS_DIR / "base.css"

_DARK_PALETTE = (CSS_DIR / "theme-dark.css").read_text(encoding="utf-8")
_THEME_HEAD = render_html(
    "shared/themes.html",
    dark=Markup(_DARK_PALETTE),
    light=Markup((CSS_DIR / "theme-light.css").read_text(encoding="utf-8")),
    script=Markup((CSS_DIR.parent / "js" / "theme.js").read_text(encoding="utf-8")),
)
_THEME_COMMANDS = {
    ui_text("chat.theme_commands.smarmella"): "light",
    ui_text("chat.theme_commands.open"): "light",
    ui_text("chat.theme_commands.close"): "dark",
}


def apply_theme_command(client: Client, text: str) -> bool:
    """Apply a standalone composer command to this browser's theme."""
    theme = _THEME_COMMANDS.get(text.strip().casefold())
    if theme is None:
        return False
    client.run_javascript(f"window.exmekhTheme?.set('{theme}')")
    return True


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
    """Return the default palette and shared rules as a single stylesheet."""
    files = get_css_files()
    if not files:
        return ""
    return "\n\n".join(
        [_DARK_PALETTE, *(file.read_text(encoding="utf-8") for file in files)]
    )


def load_css() -> None:
    """Inject custom CSS stylesheets into the current NiceGUI page."""
    ui.add_head_html(_THEME_HEAD)
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
