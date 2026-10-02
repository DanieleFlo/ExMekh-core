from pathlib import Path

import pytest
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMX
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider

from fe.config import config
from fe.exmekh_bridge.image_input import user_content
from fe.exmekh_bridge.pasted_input import (
    create_staged_attachment_from_file,
    prepare_user_input,
)
from tests.image_samples import pdf_bytes, png_bytes


@pytest.mark.live
@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("kind", ["images", "pdf"])
def test_live_ds4_reads_multiple_images_and_pdf_pages(
    tmp_path: Path, api: str, kind: str
):
    if kind == "pdf":
        attachments = [
            create_staged_attachment_from_file("colors.pdf", pdf_bytes(), tmp_path)
        ]
    else:
        attachments = [
            create_staged_attachment_from_file("red.png", png_bytes(), tmp_path),
            create_staged_attachment_from_file("blue.png", png_bytes("blue"), tmp_path),
        ]
    prepared = prepare_user_input(
        "Name the dominant color of the first image, then the second. "
        "Reply with only the two English color names in that order.",
        attachments=attachments,
        work_root=tmp_path,
        use_file_references=False,
    )
    provider = LocalProvider(
        model=config.default_model,
        base_url=config.base_url,
        api_key=config.api_key,
        api=api,
        reasoning_effort=ReasoningEffortLMX.OFF,
    )
    provider.client = provider.client.with_options(timeout=90, max_retries=0)
    client = Client(
        provider=provider,
        system_message=SystemMessage("Answer questions about the supplied images."),
    )
    limit = "max_output_tokens" if api == "responses" else "max_tokens"
    with provider.client:
        response = client.call(
            ClientRequest(
                UserMessage(user_content(prepared.prompt, prepared.image_urls)),
                extra={limit: 64, "temperature": 0},
            )
        )
        answer = response.get_message().lower()
        assert "red" in answer and "blue" in answer, answer
        assert answer.index("red") < answer.index("blue"), answer
        response = client.call(
            ClientRequest(
                UserMessage(
                    "Repeat just the two previous color names in the same order."
                ),
                extra={limit: 64, "temperature": 0},
            )
        )
        answer = response.get_message().lower()
        assert "red" in answer and "blue" in answer, answer
        assert answer.index("red") < answer.index("blue"), answer
