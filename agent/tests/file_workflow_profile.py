"""Explicit Custom profile for bounded real file-workflow qualification."""

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.providers.local import LocalProvider
from pydantic import BaseModel

from exmekh_agent.state.agent_design import AgentDesign, DesignLink, DesignNode


class GreedyFileProvider(LocalProvider):
    """Keep real model requests and handlers; fix sampling for this live profile."""

    @staticmethod
    def _greedy_request[Request: ProviderRequest](request: Request) -> Request:
        return request.with_extra({**(request.get_extra() or {}), "temperature": 0})

    def call(self, request: ProviderRequest) -> ProviderResponse:
        return super().call(self._greedy_request(request))

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        return super().call_structured(self._greedy_request(request))


def file_workflow_design() -> AgentDesign:
    root = DesignNode(
        kind="hds",
        name="file-workflow",
        content=(
            "Use file-processing to carry out the user's workspace file operation. "
            "Open it once per request. After it returns the completed result, "
            "answer the user directly and stop. Do not repeat completed operations."
        ),
    )
    source = CAPABILITY_FORMAT.bundled_root / "deep-skill/file-processing.md"
    skill = DesignNode(
        kind="skill",
        name="file-processing",
        source=str(source),
        content=source.read_text(),
    )
    tools = [
        DesignNode(
            kind="tool",
            name=name,
            source=str(CAPABILITY_FORMAT.bundled_root / "tools" / name / "main.py"),
        )
        for name in (
            "inspect-file",
            "read-file",
            "edit-file",
            "new-file",
            "delete-file",
        )
    ]
    return AgentDesign(
        nodes=[root, skill, *tools],
        links=[
            DesignLink(parent=root.id, child=skill.id),
            *(DesignLink(parent=skill.id, child=tool.id) for tool in tools),
        ],
        root_id=root.id,
        skill_connections_explicit=True,
    )
