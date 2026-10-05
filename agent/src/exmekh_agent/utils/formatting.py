"""Pure formatting of byte sizes, token counts, percentages and durations."""

from __future__ import annotations

from exmekh_agent.config.formatting import DEFAULTS as FORMAT_DEFAULTS


def format_file_size(size_bytes: int) -> str:
    base = FORMAT_DEFAULTS["byte_unit_base"]
    units = FORMAT_DEFAULTS["byte_units"]
    if size_bytes < base:
        return FORMAT_DEFAULTS["byte_template"].format(count=size_bytes, unit=units[0])
    index = 0
    scaled_size = size_bytes
    while scaled_size >= base and index < len(units) - 1:
        scaled_size /= base
        index += 1
    return FORMAT_DEFAULTS["scaled_byte_template"].format(
        count=scaled_size, unit=units[index]
    )


def format_compact_tokens(count: int) -> str:
    """Format a token count using compact 'k' notation (e.g. 1.4k, 22.0k, 32.8k)."""
    safe_count = max(0, int(count))
    if safe_count == 0:
        return "0"
    if safe_count < FORMAT_DEFAULTS["compact_token_threshold"]:
        return str(safe_count)
    return FORMAT_DEFAULTS["compact_token_template"].format(
        count=safe_count / FORMAT_DEFAULTS["compact_token_base"]
    )


def format_context_percentage(used_tokens: int, context_length: int) -> str:
    """Format the percentage of context consumed (e.g. '4.2%')."""
    if context_length <= 0:
        return FORMAT_DEFAULTS["percentage_template"].format(count=0.0)
    pct = max(0.0, (max(0, int(used_tokens)) / float(context_length)) * 100.0)
    return FORMAT_DEFAULTS["percentage_template"].format(count=pct)


def format_duration(seconds: float | int | None) -> str:
    """Format a nonnegative duration, keeping complete minutes and hours."""
    total_seconds = max(0, int(seconds or 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        template = FORMAT_DEFAULTS["duration_hours_template"]
    elif minutes > 0:
        template = FORMAT_DEFAULTS["duration_minutes_template"]
    else:
        template = FORMAT_DEFAULTS["duration_seconds_template"]
    return template.format(hours=hours, minutes=minutes, seconds=secs)
