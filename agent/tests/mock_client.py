from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer
from qs.tokenizers.qwen import QwenTokenizer

from fe.qs_bridge import agent_runner as ar
from fe.qs_bridge.models import ChatTitle, SkillNode, ToolEvent

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Represents a single deterministic step in a consecutive command pipeline."""

    name: str
    prompt_trigger: str | Callable[[str], bool] | None = None
    reasoning_chunks: list[tuple[str, str, int]] = field(
        default_factory=list
    )  # (content, skill_path, depth)
    tool_events: list[ToolEvent] = field(default_factory=list)
    skill_events: list[tuple[SkillNode, str]] = field(
        default_factory=list
    )  # (node, action: "opened"|"updated"|"completed")
    response_tokens: list[str] = field(default_factory=list)
    final_response: str = ""
    delay: float = 0.0
    exception: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, prompt: str) -> bool:
        if self.prompt_trigger is None:
            return True
        if callable(self.prompt_trigger):
            return bool(self.prompt_trigger(prompt))
        if isinstance(self.prompt_trigger, str):
            return self.prompt_trigger.lower() in prompt.lower()
        return False

    def get_full_response(self) -> str:
        if self.final_response:
            return self.final_response
        return "".join(self.response_tokens)


class ScenarioPipeline:
    """Manages a sequence of connected, consecutive command steps with shared state."""

    def __init__(self, name: str, steps: list[PipelineStep] | None = None):
        self.name = name
        self.steps: list[PipelineStep] = list(steps) if steps else []
        self.current_step_index: int = 0
        self.state: dict[str, Any] = {}
        self.history_records: list[dict[str, Any]] = []

    def add_step(self, step: PipelineStep) -> ScenarioPipeline:
        self.steps.append(step)
        return self

    @property
    def is_completed(self) -> bool:
        return self.current_step_index >= len(self.steps)

    def get_current_step(self) -> PipelineStep | None:
        if self.is_completed:
            return None
        return self.steps[self.current_step_index]

    def advance(self) -> PipelineStep | None:
        if not self.is_completed:
            self.current_step_index += 1
        return self.get_current_step()

    def reset(self) -> None:
        self.current_step_index = 0
        self.state.clear()
        self.history_records.clear()

    def find_step_for_prompt(self, prompt: str) -> tuple[int, PipelineStep] | None:
        # First check current step
        curr = self.get_current_step()
        if curr and curr.matches(prompt):
            return self.current_step_index, curr

        # Fallback to search any remaining step
        for idx in range(self.current_step_index, len(self.steps)):
            step = self.steps[idx]
            if step.matches(prompt):
                self.current_step_index = idx
                return idx, step
        return None


def create_default_hierarchical_pipeline() -> ScenarioPipeline:
    """Creates the standard 3-step hierarchical DeepAgent pipeline with linked consecutive commands."""
    pipeline = ScenarioPipeline(name="hierarchical_deep_agent_pipeline")

    # Step 1: Initialization & Diagnostic
    step1 = PipelineStep(
        name="step_1_diagnostic",
        prompt_trigger=lambda p: any(
            k in p.lower() for k in ("inizializza", "diagnostica", "step 1", "fase 1")
        ),
        reasoning_chunks=[
            (
                "[Fase 1 - Root] Inizializzazione pipeline ed esecuzione diagnostica preliminare dell'ambiente.\n",
                "/",
                0,
            ),
            (
                "[Fase 1 - Root] Verifica delle risorse e preparazione del token di contesto 'SYS_DIAG_OK'.",
                "/",
                0,
            ),
        ],
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=0, skill_path="/"),
        ],
        response_tokens=[
            "Diagnostica di sistema completata con successo.\n",
            "Token di inizializzazione generato: `SYS_DIAG_OK`.\n",
            "La pipeline è pronta per l'elaborazione gerarchica dei sotto-livelli.",
        ],
        final_response=(
            "Diagnostica di sistema completata con successo.\n"
            "Token di inizializzazione generato: `SYS_DIAG_OK`.\n"
            "La pipeline è pronta per l'elaborazione gerarchica dei sotto-livelli."
        ),
        metadata={"token": "SYS_DIAG_OK"},
    )

    # Step 2: Nested Sub-Skill & Sub-Tool Execution (linked to Step 1 token)
    sub_skill_node = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="nested-sub-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
            )
        ],
        is_collapsed=False,
    )
    root_skill_node = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="active",
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=1, skill_path="/mock-skill")
        ],
        sub_skills=[sub_skill_node],
        is_collapsed=False,
    )

    step2 = PipelineStep(
        name="step_2_nested_skill",
        prompt_trigger=lambda p: any(
            k in p.lower()
            for k in ("approfondisci", "sotto-skill", "sub-skill", "step 2", "fase 2")
        ),
        reasoning_chunks=[
            (
                "[Fase 2 - Root] Ricevuto comando di approfondimento. Carico contesto 'SYS_DIAG_OK'.\n",
                "/",
                0,
            ),
            (
                "[Fase 2 - Skill mock-skill] Apertura della skill gerarchica 'mock-skill' a depth 1.\n",
                "/mock-skill",
                1,
            ),
            (
                "[Fase 2 - SubSkill nested-sub-skill] Invocazione della sotto-skill annidata 'nested-sub-skill' a depth 2.",
                "/mock-skill/nested-sub-skill",
                2,
            ),
        ],
        skill_events=[
            (root_skill_node, "opened"),
            (sub_skill_node, "opened"),
            (sub_skill_node, "completed"),
        ],
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=1, skill_path="/mock-skill"),
            ToolEvent(
                tool_name="nested-sub-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
            ),
        ],
        response_tokens=[
            "Elaborazione gerarchica completata all'interno di `/mock-skill` e `/mock-skill/nested-sub-skill`.\n",
            "I dati diagnostici derivati dal Passo 1 sono stati trasformati con successo.",
        ],
        final_response=(
            "Elaborazione gerarchica completata all'interno di `/mock-skill` e `/mock-skill/nested-sub-skill`.\n"
            "I dati diagnostici derivati dal Passo 1 sono stati trasformati con successo."
        ),
        metadata={"processed": True},
    )

    # Step 3: Synthesis & Final Consolidation Report
    comp_sub_node = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="completed",
        is_collapsed=True,
    )
    comp_root_node = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        is_collapsed=True,
        sub_skills=[comp_sub_node],
    )

    step3 = PipelineStep(
        name="step_3_synthesis_report",
        prompt_trigger=lambda p: any(
            k in p.lower() for k in ("report", "sintesi", "finale", "step 3", "fase 3")
        ),
        reasoning_chunks=[
            (
                "[Fase 3 - Root Sintesi] Elaborazione del report finale consolidato unificando gli esiti dei passi 1 e 2.\n",
                "/",
                0,
            ),
            (
                "[Fase 3 - Root Sintesi] Tutte le skill gerarchiche sono state chiuse e compattate con successo.",
                "/",
                0,
            ),
        ],
        skill_events=[
            (comp_root_node, "completed"),
        ],
        response_tokens=[
            "### Report Finale della Pipeline Consecutiva\n\n",
            "1. **Passo 1 (Diagnostica)**: Inizializzato con `SYS_DIAG_OK`.\n",
            "2. **Passo 2 (Gerarchia Skills)**: Eseguito `/mock-skill` e `/mock-skill/nested-sub-skill`.\n",
            "3. **Passo 3 (Sintesi)**: Pipeline completata con successo con integrità verificata al 100%.\n",
        ],
        final_response=(
            "### Report Finale della Pipeline Consecutiva\n\n"
            "1. **Passo 1 (Diagnostica)**: Inizializzato con `SYS_DIAG_OK`.\n"
            "2. **Passo 2 (Gerarchia Skills)**: Eseguito `/mock-skill` e `/mock-skill/nested-sub-skill`.\n"
            "3. **Passo 3 (Sintesi)**: Pipeline completata con successo con integrità verificata al 100%.\n"
        ),
        metadata={"done": True},
    )

    pipeline.add_step(step1).add_step(step2).add_step(step3)
    return pipeline


def create_quick_chat_pipeline() -> ScenarioPipeline:
    """Creates a lightweight multi-turn quick chat pipeline without tools/skills."""
    pipeline = ScenarioPipeline(name="quick_chat_pipeline")

    s1 = PipelineStep(
        name="qc_turn_1",
        response_tokens=[
            "Ciao! Sono il client mock in modalità Quick Chat. Come posso assisterti?"
        ],
    )
    s2 = PipelineStep(
        name="qc_turn_2",
        response_tokens=[
            "Ho ricevuto il tuo messaggio di secondo livello nella pipeline rapida."
        ],
    )
    s3 = PipelineStep(
        name="qc_turn_3",
        response_tokens=[
            "Operazione completata con successo nella pipeline Quick Chat."
        ],
    )
    pipeline.add_step(s1).add_step(s2).add_step(s3)
    return pipeline


class MockProvider(Provider):
    """Mock Provider extending qs.interfaces.provider.Provider to serve scenario pipelines."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        pipeline: ScenarioPipeline | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer if tokenizer is not None else QwenTokenizer()
        )
        self.pipeline: ScenarioPipeline = (
            pipeline or create_default_hierarchical_pipeline()
        )
        self.last_request: ProviderRequest | None = None
        self._step_counter = 0

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.last_request = request
        messages = request.get_messages()
        prompt_text = ""
        for m in reversed(messages):
            if m.get_role_name() == "user":
                prompt_text = str(m.get_message() or "")
                break

        match = self.pipeline.find_step_for_prompt(prompt_text)
        if match is not None:
            _, step = match
            self.pipeline.advance()
        else:
            self._step_counter += 1
            step = PipelineStep(
                name=f"auto_step_{self._step_counter}",
                final_response=f"Mock response for: {prompt_text}",
                response_tokens=[f"Mock response for: {prompt_text}"],
            )

        if request.get_streaming() and request.on_event:
            request.on_event.handle(
                ProviderStreamEvent(type=ProviderStreamEventType.START)
            )
            for content, skill_path, depth in step.reasoning_chunks:
                request.on_event.handle(
                    ProviderStreamEvent(
                        type=ProviderStreamEventType.REASONING,
                        content=content,
                    )
                )
            for token in step.response_tokens:
                request.on_event.handle(
                    ProviderStreamEvent(
                        type=ProviderStreamEventType.RESPONSE,
                        content=token,
                    )
                )
            if step.exception:
                raise step.exception
            request.on_event.handle(
                ProviderStreamEvent(type=ProviderStreamEventType.END)
            )
        elif step.exception:
            raise step.exception

        resp_msg = AssistantMessage(step.get_full_response())
        return ProviderResponse(resp_msg)

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        resp_model_cls = request.get_response_model()
        fields = getattr(resp_model_cls, "model_fields", {})
        data: dict[str, Any] = {}
        if "value" in fields:
            data["value"] = 42
        if "title" in fields:
            prompt_text = ""
            for m in reversed(request.get_messages()):
                if m.get_role_name() == "user":
                    prompt_text = str(m.get_message() or "")
                    break
            clean = prompt_text.strip().replace("\n", " ") if prompt_text else ""
            data["title"] = (
                clean[:30] + ("..." if len(clean) > 30 else "")
                if clean
                else "Conversazione Mock"
            )
        try:
            resp_obj = resp_model_cls.model_validate(data)
        except Exception:
            resp_obj = resp_model_cls.model_construct(**data)
        return StructuredProviderResponse(StructuredAssistantMessage(resp_obj))


class MockClient(Client):
    """Mock Client inheriting from qs.core.models.client.Client backed by a MockProvider instance."""

    def __init__(
        self,
        provider: MockProvider | None = None,
        system_message: SystemMessage | None = None,
        tools: Tools | None = None,
        pipeline: ScenarioPipeline | None = None,
    ) -> None:
        self._mock_provider = provider or MockProvider(pipeline=pipeline)
        sys_msg = (
            system_message
            if system_message is not None
            else SystemMessage("You are a mock AI assistant for UI E2E tests.")
        )
        super().__init__(
            provider=self._mock_provider,
            system_message=sys_msg,
            tools=tools,
        )

    @property
    def mock_provider(self) -> MockProvider:
        return self._mock_provider

    @property
    def pipeline(self) -> ScenarioPipeline:
        return self._mock_provider.pipeline

    def set_pipeline(self, pipeline: ScenarioPipeline) -> None:
        self._mock_provider.pipeline = pipeline

    def execute_turn(
        self,
        user_prompt: str,
        on_token: Callable[[str], None],
        on_reasoning: Callable[..., None],
        on_tool_event: Callable[[ToolEvent], None],
        on_skill_event: Callable[[SkillNode, str], None] | None = None,
        delay_between_tokens: float = 0.0,
        on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
        | None = None,
        history: Any | None = None,
    ) -> str:
        """Executes a full turn across the pipeline, driving all UI stream callbacks."""
        prior_qs = ar.convert_to_qs_messages(history, current_user_prompt=user_prompt)
        if prior_qs:
            try:
                self.replace_history_from(1, prior_qs)
            except Exception:
                pass

        # Record provider request with full conversation history
        self._history.add(UserMessage(user_prompt))
        self._mock_provider.last_request = ProviderRequest(messages=self.get_history())

        match = self._mock_provider.pipeline.find_step_for_prompt(user_prompt)
        if match is not None:
            _, step = match
            self._mock_provider.pipeline.advance()
        elif "esempio codice" in user_prompt.lower():
            fenced_resp = (
                "Ecco gli esempi richiesti:\n\n"
                "```python\n"
                "def calcola(x):\n"
                "    return x * 2\n"
                "```\n\n"
                "```text\n"
                "SEZIONE_TESTO_SPECIALE_SOLO_QUESTO\n"
                "```\n\n"
                "Fine della risposta."
            )
            step = PipelineStep(
                name="code_example_step",
                final_response=fenced_resp,
                response_tokens=[fenced_resp],
            )
        else:
            self._mock_provider._step_counter += 1
            step = PipelineStep(
                name=f"fallback_step_{self._mock_provider._step_counter}",
                final_response=f"Risposta mock automatica per: {user_prompt}",
                response_tokens=[f"Risposta mock automatica per: {user_prompt}"],
            )

        effective_delay = (
            delay_between_tokens if delay_between_tokens > 0 else step.delay
        )

        # 1. Stream reasoning chunks
        for content, skill_path, depth in step.reasoning_chunks:
            try:
                on_reasoning(content, skill_path, depth)
            except TypeError:
                on_reasoning(content)
            if effective_delay > 0:
                time.sleep(effective_delay)

        # 2. Emit skill events
        if on_skill_event:
            for node, action in step.skill_events:
                on_skill_event(node, action)

        # 3. Emit tool events
        for tool_ev in step.tool_events:
            on_tool_event(tool_ev)

        # 4. Stream response tokens
        for token in step.response_tokens:
            on_token(token)
            if effective_delay > 0:
                time.sleep(effective_delay)

        if step.exception:
            raise step.exception

        full_text = step.get_full_response()
        self._history.add(AssistantMessage(full_text))

        # 5. Emit raw history if callback is provided
        if on_raw_history:
            raw_hist: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": "You are a friendly, accurate, and professional AI assistant.",
                },
            ]
            for pm in prior_qs:
                raw_hist.append(
                    {"role": pm.get_role_name(), "content": str(pm.get_message() or "")}
                )
            raw_hist.append({"role": "user", "content": user_prompt})

            if step.tool_events:
                tool_calls = [
                    {
                        "id": f"call_{ev.tool_name}_{idx + 1}",
                        "type": "function",
                        "function": {"name": ev.tool_name, "arguments": {}},
                    }
                    for idx, ev in enumerate(step.tool_events)
                ]
                raw_hist.append(
                    {"role": "assistant", "content": None, "tool_calls": tool_calls}
                )
                for idx, ev in enumerate(step.tool_events):
                    raw_hist.append(
                        {
                            "role": "tool",
                            "name": ev.tool_name,
                            "tool_call_id": f"call_{ev.tool_name}_{idx + 1}",
                            "content": f'<Tool name="{ev.tool_name}">Output di {ev.tool_name}</Tool>',
                        }
                    )

            raw_hist.append({"role": "assistant", "content": full_text})

            skill_tree: list[dict[str, Any]] = []
            if step.skill_events:
                for node, _ in step.skill_events:
                    if node.depth == 1:
                        skill_tree.append(
                            {
                                "skill": node.name,
                                "opening_tool_call_id": f"call_skill_{node.name}",
                                "output": f"Completed {node.name}",
                                "history": [
                                    {
                                        "role": "system",
                                        "content": f"Sub-skill runtime for {node.name}",
                                    },
                                    {
                                        "role": "assistant",
                                        "content": f"Completed inner execution for {node.name}",
                                    },
                                ],
                                "children": [
                                    {
                                        "skill": sub.name,
                                        "opening_tool_call_id": f"call_nested_{sub.name}",
                                        "output": f"Completed {sub.name}",
                                        "history": [
                                            {
                                                "role": "system",
                                                "content": f"Nested sub-skill runtime for {sub.name}",
                                            },
                                            {
                                                "role": "assistant",
                                                "content": f"Completed inner nested {sub.name}",
                                            },
                                        ],
                                        "children": [],
                                    }
                                    for sub in node.sub_skills
                                ],
                            }
                        )

            on_raw_history(raw_hist, skill_tree)

        return full_text


# Global mock provider / client registry
_GLOBAL_MOCK_PROVIDER: MockProvider | None = None


def get_global_mock_provider() -> MockProvider:
    global _GLOBAL_MOCK_PROVIDER
    if _GLOBAL_MOCK_PROVIDER is None:
        _GLOBAL_MOCK_PROVIDER = MockProvider()
    return _GLOBAL_MOCK_PROVIDER


def set_global_mock_provider(provider: MockProvider | None) -> None:
    global _GLOBAL_MOCK_PROVIDER
    _GLOBAL_MOCK_PROVIDER = provider


def get_global_mock_client() -> MockClient:
    return MockClient(provider=get_global_mock_provider())


def reset_mock_provider() -> MockProvider:
    global _GLOBAL_MOCK_PROVIDER
    _GLOBAL_MOCK_PROVIDER = MockProvider(
        pipeline=create_default_hierarchical_pipeline()
    )
    return _GLOBAL_MOCK_PROVIDER


_ORIG_LOCAL_PROVIDER = getattr(ar, "LocalProvider", None)
_ORIG_HEALTH = getattr(ar, "check_provider_health", None)
_ORIG_EXECUTE = getattr(ar, "execute_agent_turn_sync", None)
_ORIG_TITLE = getattr(ar, "generate_chat_title_sync", None)


def uninstall_mock_provider() -> None:
    """Restores fe.qs_bridge.agent_runner to original production implementations."""
    if _ORIG_LOCAL_PROVIDER:
        ar.LocalProvider = _ORIG_LOCAL_PROVIDER
    if _ORIG_HEALTH:
        ar.check_provider_health = _ORIG_HEALTH
    if _ORIG_EXECUTE:
        ar.execute_agent_turn_sync = _ORIG_EXECUTE
    if _ORIG_TITLE:
        ar.generate_chat_title_sync = _ORIG_TITLE


def install_mock_provider(mock_provider: MockProvider | None = None) -> MockProvider:
    """Installs the MockProvider into fe.qs_bridge.agent_runner at test runtime."""
    provider = mock_provider or get_global_mock_provider()

    def mock_local_provider(*args: Any, **kwargs: Any) -> MockProvider:
        return provider

    ar.LocalProvider = mock_local_provider

    async def mock_health(*args: Any, **kwargs: Any) -> tuple[bool, list[str]]:
        return True, ["mock-deep-agent", "mock-pipeline-model", "mock-quick-chat"]

    ar.check_provider_health = mock_health

    def mock_execute_turn_sync(
        user_prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        api_type: str,
        reasoning_effort_str: str,
        agent_mode: str,
        capabilities_path_str: str,
        name_hds: str,
        on_token: Callable[[str], None],
        on_reasoning: Callable[..., None],
        on_tool_event: Callable[[ToolEvent], None],
        on_skill_event: Callable[[SkillNode, str], None] | None = None,
        on_raw_history: Callable[..., None] | None = None,
        history: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        client = MockClient(provider=provider)
        return client.execute_turn(
            user_prompt=user_prompt,
            on_token=on_token,
            on_reasoning=on_reasoning,
            on_tool_event=on_tool_event,
            on_skill_event=on_skill_event,
            on_raw_history=on_raw_history,
            history=history,
        )

    ar.execute_agent_turn_sync = mock_execute_turn_sync

    def mock_generate_title_sync(
        user_prompt: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
    ) -> str:
        mock_client = MockClient(provider=provider)
        req = ClientRequest(UserMessage(user_prompt))
        res = mock_client.call_structured(req, ChatTitle, strict=False)
        msg = res.get_message()
        if isinstance(msg, ChatTitle):
            return msg.title
        return getattr(msg, "title", str(msg))

    ar.generate_chat_title_sync = mock_generate_title_sync
    return provider
