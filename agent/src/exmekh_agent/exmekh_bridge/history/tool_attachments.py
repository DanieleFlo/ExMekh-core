"""Read retained tool image references without decoding their pixel payloads."""

from typing import Any

from exmekh_lib.core.cache.user_message_codec import user_message_image_urls


def tool_attachment_image_urls(content: Any) -> list[str]:
    """Read preview URLs without decoding pixels on each streaming update."""
    if isinstance(content, dict):
        return user_message_image_urls(content)
    if not isinstance(content, list):
        return []
    urls = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            urls.append(part["image_url"]["url"])
    return urls
