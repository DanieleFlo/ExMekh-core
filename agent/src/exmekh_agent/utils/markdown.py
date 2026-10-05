"""Parse YAML front matter without exposing capability file layout."""

from __future__ import annotations

import yaml

from exmekh_agent.config.catalogs import message as app_text


def markdown_parts(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        header, separator, body = text[4:].partition("\n---")
        if separator:
            try:
                metadata = yaml.safe_load(header)
                if metadata is None:
                    metadata = {}
            except yaml.YAMLError as error:
                raise ValueError(
                    app_text(
                        "errors_capabilities.capability_repository.invalid_capability_metadata"
                    )
                ) from error
            if not isinstance(metadata, dict):
                raise ValueError(
                    app_text(
                        "errors_capabilities.capability_repository.capability_metadata_must_be_a_mapping"
                    )
                )
            return metadata, body.strip()
    return {}, text.strip()
