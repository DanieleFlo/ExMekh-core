"""Native folder navigation shared by project creation and capability references."""

import tkinter
from pathlib import Path
from tkinter import filedialog

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.state.session_manager import CODE_WORK_ROOT


def pick_folder_dialog(initial_dir: str | Path | None = None) -> str | None:
    """Open the native picker and always release its Tk window."""
    initial_path = Path(initial_dir).expanduser() if initial_dir else CODE_WORK_ROOT
    if not initial_path.is_dir():
        initial_path = initial_path.parent
    initial_directory = str(initial_path.resolve()) if initial_path.is_dir() else None
    root = tkinter.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=initial_directory,
            title=app_text("navigation.new_project_dialog.choose_project_folder"),
            mustexist=True,
        )
        return selected or None
    finally:
        root.destroy()
