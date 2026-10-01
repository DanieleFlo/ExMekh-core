from __future__ import annotations

from pathlib import Path


def format_file_size(size_bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


FILE_ICON_MAP: dict[str, tuple[str, str]] = {
    # Python
    ".py": ("terminal", "text-emerald-400"),
    ".pyw": ("terminal", "text-emerald-400"),
    ".ipynb": ("terminal", "text-emerald-400"),
    # Web & Dev Code
    ".js": ("code", "text-sky-400"),
    ".mjs": ("code", "text-sky-400"),
    ".cjs": ("code", "text-sky-400"),
    ".ts": ("code", "text-sky-400"),
    ".tsx": ("code", "text-sky-400"),
    ".jsx": ("code", "text-sky-400"),
    ".html": ("code", "text-amber-400"),
    ".htm": ("code", "text-amber-400"),
    ".css": ("code", "text-sky-300"),
    ".scss": ("code", "text-pink-400"),
    ".sass": ("code", "text-pink-400"),
    ".less": ("code", "text-blue-400"),
    ".vue": ("code", "text-emerald-400"),
    ".svelte": ("code", "text-orange-400"),
    ".c": ("code", "text-blue-400"),
    ".cpp": ("code", "text-blue-500"),
    ".h": ("code", "text-blue-300"),
    ".hpp": ("code", "text-blue-300"),
    ".cs": ("code", "text-purple-400"),
    ".java": ("code", "text-orange-500"),
    ".go": ("code", "text-cyan-400"),
    ".rs": ("code", "text-amber-600"),
    ".php": ("code", "text-indigo-400"),
    ".rb": ("code", "text-red-400"),
    ".swift": ("code", "text-orange-400"),
    ".kt": ("code", "text-purple-400"),
    # Shell & Scripts
    ".sh": ("terminal", "text-teal-400"),
    ".bash": ("terminal", "text-teal-400"),
    ".zsh": ("terminal", "text-teal-400"),
    ".ps1": ("terminal", "text-blue-400"),
    ".bat": ("terminal", "text-zinc-400"),
    ".cmd": ("terminal", "text-zinc-400"),
    # Config & Structured Data
    ".json": ("data_object", "text-amber-400"),
    ".yaml": ("data_object", "text-amber-300"),
    ".yml": ("data_object", "text-amber-300"),
    ".toml": ("data_object", "text-orange-400"),
    ".xml": ("data_object", "text-teal-400"),
    ".ini": ("tune", "text-zinc-400"),
    ".cfg": ("tune", "text-zinc-400"),
    ".conf": ("tune", "text-zinc-400"),
    ".env": ("security", "text-emerald-400"),
    # Database
    ".sql": ("storage", "text-indigo-400"),
    ".db": ("storage", "text-indigo-400"),
    ".sqlite": ("storage", "text-indigo-400"),
    ".sqlite3": ("storage", "text-indigo-400"),
    # Documents
    ".pdf": ("picture_as_pdf", "text-rose-400"),
    ".md": ("article", "text-sky-400"),
    ".markdown": ("article", "text-sky-400"),
    ".rst": ("article", "text-sky-400"),
    ".txt": ("description", "text-zinc-300"),
    ".log": ("receipt_long", "text-zinc-400"),
    ".doc": ("description", "text-blue-400"),
    ".docx": ("description", "text-blue-400"),
    ".odt": ("description", "text-blue-300"),
    ".rtf": ("description", "text-blue-300"),
    # Spreadsheets
    ".csv": ("table_chart", "text-emerald-400"),
    ".tsv": ("table_chart", "text-emerald-400"),
    ".xls": ("table_chart", "text-emerald-500"),
    ".xlsx": ("table_chart", "text-emerald-500"),
    ".ods": ("table_chart", "text-emerald-400"),
    # Presentations
    ".ppt": ("slideshow", "text-amber-500"),
    ".pptx": ("slideshow", "text-amber-500"),
    ".odp": ("slideshow", "text-amber-500"),
    # Images
    ".png": ("image", "text-purple-400"),
    ".jpg": ("image", "text-purple-400"),
    ".jpeg": ("image", "text-purple-400"),
    ".gif": ("image", "text-pink-400"),
    ".webp": ("image", "text-purple-400"),
    ".svg": ("image", "text-orange-400"),
    ".bmp": ("image", "text-purple-300"),
    ".ico": ("image", "text-amber-400"),
    ".tiff": ("image", "text-purple-400"),
    # Audio
    ".mp3": ("audiotrack", "text-pink-400"),
    ".wav": ("audiotrack", "text-pink-400"),
    ".ogg": ("audiotrack", "text-pink-400"),
    ".m4a": ("audiotrack", "text-pink-400"),
    ".flac": ("audiotrack", "text-pink-400"),
    ".aac": ("audiotrack", "text-pink-400"),
    # Video
    ".mp4": ("videocam", "text-red-400"),
    ".mkv": ("videocam", "text-red-400"),
    ".avi": ("videocam", "text-red-400"),
    ".mov": ("videocam", "text-red-400"),
    ".webm": ("videocam", "text-red-400"),
    # Archives
    ".zip": ("folder_zip", "text-amber-400"),
    ".tar": ("folder_zip", "text-amber-400"),
    ".gz": ("folder_zip", "text-amber-400"),
    ".tgz": ("folder_zip", "text-amber-400"),
    ".7z": ("folder_zip", "text-amber-400"),
    ".rar": ("folder_zip", "text-amber-400"),
    ".bz2": ("folder_zip", "text-amber-400"),
    # Binaries & Libraries
    ".exe": ("memory", "text-zinc-400"),
    ".dll": ("memory", "text-zinc-400"),
    ".so": ("memory", "text-zinc-400"),
    ".dylib": ("memory", "text-zinc-400"),
    ".bin": ("memory", "text-zinc-400"),
    ".whl": ("memory", "text-teal-400"),
    ".iso": ("disc_full", "text-zinc-400"),
}


def get_file_icon(filename: str, mime_type: str = "") -> tuple[str, str, str]:
    """Return (icon_name, color_class, ext_label) tailored for the given filename or MIME type."""
    path = Path(filename)
    suffix = ".env" if path.name.lower() == ".env" else path.suffix.lower()
    if suffix in FILE_ICON_MAP:
        icon, color = FILE_ICON_MAP[suffix]
        return icon, color, suffix

    if mime_type:
        m = mime_type.lower()
        if m.startswith("image/"):
            return "image", "text-purple-400", suffix or ".img"
        if m.startswith("audio/"):
            return "audiotrack", "text-pink-400", suffix or ".audio"
        if m.startswith("video/"):
            return "videocam", "text-red-400", suffix or ".video"
        if "pdf" in m:
            return "picture_as_pdf", "text-rose-400", ".pdf"
        if "text/" in m or "json" in m or "xml" in m:
            return "article", "text-sky-400", suffix or ".txt"

    return "insert_drive_file", "text-zinc-400", suffix or ".file"
