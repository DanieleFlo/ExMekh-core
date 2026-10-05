from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.reasoning_effort import (
    ReasoningEffort,
    ReasoningEffortLMH,
    ReasoningEffortLMX,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.agents.cancellation import (
    CancellableProvider,
    raise_if_cancelled,
)
from exmekh_agent.exmekh_bridge.agents.event_handlers import (
    DeepAgentTurnEventHandler,
    QuickChatEventHandler,
    publish_raw_history,
)
from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.agents.sub_agents import (
    ObservedNode,
    SubAgentEvent,
    SubAgentEventRouter,
)
from exmekh_agent.exmekh_bridge.capabilities.agent_design import build_designed_agent
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.history.converter import (
    convert_to_exmekh_messages,
    exmekh_message_to_raw_dict,
    restore_node_history,
    serialize_skill_execution,
)
from exmekh_agent.exmekh_bridge.history.image_input import (
    restore_image_history,
    serialize_user_content,
    user_content,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatSettings,
    ChatTitle,
    ProviderContextUsage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.providers.provider_context import (
    ContextTrackingProvider,
)
from exmekh_agent.exmekh_bridge.providers.provider_health import (
    check_provider_health as check_provider_health,
)
from exmekh_agent.exmekh_bridge.providers.provider_health import (
    extract_model_context_length as extract_model_context_length,
)
from exmekh_agent.exmekh_bridge.resources import load_prompt
from exmekh_agent.exmekh_bridge.tools.tool_approval import (
    DEFAULT_TOOL_APPROVAL_MODE,
    CommandApprovalHook,
    CommandApprovedCallback,
)
from exmekh_agent.state.agent_design import AgentDesign

logger = logging.getLogger(__name__)

# ds4 currently keeps one live agentic checkpoint frontier per server process.
# Serializing complete requests prevents one conversation from restoring another
# conversation's skill checkpoint.  A threading lock also covers worker threads
# used by title generation and by separate NiceGUI clients.
_provider_request_lock = threading.RLock()


def execute_agent_turn_sync(
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
    on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
    | None = None,
    history: Any | None = None,
    agent_context_id: str | None = None,
    on_sub_agent_event: Callable[[SubAgentEvent], None] | None = None,
    sub_agent_history: list[SubAgentConversation] | None = None,
    work_root: str | None = None,
    coding_agent_persistent_history: bool | None = None,
    on_context: Callable[[ProviderContextUsage], None] | None = None,
    on_request_manual_approval: Callable[[str], bool] | None = None,
    on_command_approved: CommandApprovedCallback | None = None,
    tool_approval_mode: str | None = None,
    approved_commands_history: list[str] | None = None,
    tool_approval_raw_history: list[dict[str, Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    steering: TurnSteering | None = None,
    on_steering_added: Callable[[], None] | None = None,
    image_urls: list[str] | None = None,
    agent_profile: str = "generic",
    agent_design: AgentDesign | None = None,
    capabilities_paths: list[str] | None = None,
) -> str:
    """Run one complete provider request without overlapping another request."""
    with _provider_request_lock:
        if cancellation_event is not None:
            raise_if_cancelled(cancellation_event)
        return _execute_agent_turn_sync_unlocked(
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_type=api_type,
            reasoning_effort_str=reasoning_effort_str,
            agent_mode=agent_mode,
            capabilities_path_str=capabilities_path_str,
            name_hds=name_hds,
            on_token=on_token,
            on_reasoning=on_reasoning,
            on_tool_event=on_tool_event,
            on_skill_event=on_skill_event,
            on_raw_history=on_raw_history,
            history=history,
            agent_context_id=agent_context_id,
            on_sub_agent_event=on_sub_agent_event,
            sub_agent_history=sub_agent_history,
            work_root=work_root,
            coding_agent_persistent_history=coding_agent_persistent_history,
            on_context=on_context,
            on_request_manual_approval=on_request_manual_approval,
            on_command_approved=on_command_approved,
            tool_approval_mode=tool_approval_mode,
            approved_commands_history=approved_commands_history,
            tool_approval_raw_history=tool_approval_raw_history,
            cancellation_event=cancellation_event,
            steering=steering,
            on_steering_added=on_steering_added,
            image_urls=image_urls,
            agent_profile=agent_profile,
            agent_design=agent_design,
            capabilities_paths=capabilities_paths,
        )


def _execute_agent_turn_sync_unlocked(
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
    on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
    | None = None,
    history: Any | None = None,
    agent_context_id: str | None = None,
    on_sub_agent_event: Callable[[SubAgentEvent], None] | None = None,
    sub_agent_history: list[SubAgentConversation] | None = None,
    work_root: str | None = None,
    coding_agent_persistent_history: bool | None = None,
    on_context: Callable[[ProviderContextUsage], None] | None = None,
    on_request_manual_approval: Callable[[str], bool] | None = None,
    on_command_approved: CommandApprovedCallback | None = None,
    tool_approval_mode: str | None = None,
    approved_commands_history: list[str] | None = None,
    tool_approval_raw_history: list[dict[str, Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    steering: TurnSteering | None = None,
    on_steering_added: Callable[[], None] | None = None,
    image_urls: list[str] | None = None,
    agent_profile: str = "generic",
    agent_design: AgentDesign | None = None,
    capabilities_paths: list[str] | None = None,
) -> str:
    """Synchronous execution of a turn using exmekh_lib, dispatched in background thread."""
    content = user_content(user_prompt, image_urls or [])
    reasoning_effort: ReasoningEffort | None = None
    if reasoning_effort_str:
        clean_effort = reasoning_effort_str.strip().lower()
        try:
            reasoning_effort = ReasoningEffortLMX(clean_effort)
        except ValueError:
            try:
                reasoning_effort = ReasoningEffortLMH(clean_effort)
            except ValueError:
                reasoning_effort = None

    provider = LocalProvider(
        model=model,
        api="responses" if api_type == "responses" else "chat_completions",
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        api_key=api_key,
    )
    context_provider = None
    if on_context or (steering is not None and agent_mode == "deep_agent"):
        context_provider = ContextTrackingProvider(
            provider,
            on_context=on_context,
            context_id=agent_context_id or name_hds or "main",
            agentic_only=agent_mode == "deep_agent",
            ds4_qwen=config.uses_ds4_qwen_prompt(model, base_url),
            thinking=reasoning_effort_str.strip().lower() != "none",
            api_type=api_type,
        )
        provider = context_provider

    if cancellation_event is not None:
        provider = CancellableProvider(provider, cancellation_event)

    if agent_mode == "deep_agent":
        effective_approval_mode = (
            tool_approval_mode
            if tool_approval_mode is not None
            else DEFAULT_TOOL_APPROVAL_MODE
        )
        approval_hook = CommandApprovalHook(
            mode=effective_approval_mode,
            history=tool_approval_raw_history or approved_commands_history or [],
            request_manual_approval=on_request_manual_approval,
            on_command_approved=on_command_approved,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_type=api_type,
        )
        approval_tool_hooks = {"bash": ToolHooks(before=approval_hook)}
        cap_path = Path(capabilities_path_str or config.capabilities_path)
        cap_paths = (
            [Path(path) for path in capabilities_paths]
            if capabilities_paths
            else [cap_path]
        )
        event_router = SubAgentEventRouter(on_sub_agent_event)
        if agent_design is not None or agent_profile != "generic":
            design = agent_design or CapabilityRepository().default_design(
                agent_profile, [str(path) for path in cap_paths]
            )
            root_node, children = build_designed_agent(
                design, event_router, approval_tool_hooks, agent_context_id
            )
        else:
            default_caps = DefaultCapabilities(
                mocks=True, tool_hooks=approval_tool_hooks
            )
            root_name = name_hds or "main"
            children: list[ObservedNode] = []
            if default_caps.capabilities_path.resolve() in {
                path.resolve() for path in cap_paths
            }:
                children.append(
                    ObservedNode(
                        name_hds="mock-sub-agent",
                        path=f"{root_name}/mock-sub-agent",
                        event_router=event_router,
                        default_capabilities=default_caps,
                    )
                )
            children.append(
                ObservedNode(
                    capabilities_paths=[
                        Path(__file__).resolve().parents[2] / "capabilities"
                    ],
                    default_capabilities=default_caps,
                    name_hds="coding-agent",
                    path=f"{root_name}/coding-agent",
                    event_router=event_router,
                    persistent_history=config.coding_agent_persistent_history
                    if coding_agent_persistent_history is None
                    else coding_agent_persistent_history,
                )
            )
            root_node = Node(
                capabilities_paths=cap_paths,
                children=children,
                name_hds=root_name,
                agent_context_id=agent_context_id,
                default_capabilities=default_caps,
            )

        prior_exmekh = convert_to_exmekh_messages(
            history, current_user_prompt=user_prompt
        )
        content_by_dump = {
            message.message_dump()["content"]: serialize_user_content(message)
            for message in [*prior_exmekh, UserMessage(content)]
            if isinstance(message, UserMessage)
            and not isinstance(message.get_message(), str)
        }

        def publish_multimodal_history(raw_history, skill_tree):
            restored_history, restored_tree = restore_image_history(
                raw_history, skill_tree, content_by_dump
            )
            publish_raw_history(on_raw_history, restored_history, restored_tree)

        event_handler = DeepAgentTurnEventHandler(
            event_router,
            on_token,
            on_reasoning,
            on_tool_event,
            on_skill_event,
            publish_multimodal_history if on_raw_history else None,
            steering,
            on_steering_added,
        )
        if context_provider is not None:
            context_provider.set_request_observer(event_handler.begin_request)

        agent = DeepAgent(
            provider=provider,
            node=root_node,
            work_root=Path(work_root or config.work_root),
            on_stream=event_handler,
        )

        for child in children:
            if not child.persistent_history or not sub_agent_history:
                continue
            previous = next(
                (
                    item
                    for item in reversed(sub_agent_history)
                    if item.path == child.path and item.persistent_history
                ),
                None,
            )
            if previous:
                child.restore_conversation(
                    previous,
                    prior_exmekh=convert_to_exmekh_messages(previous.replay_history()),
                )

        if history:
            restore_node_history(root_node, history, prior_exmekh=prior_exmekh)

        try:
            if steering is not None:
                # Restored messages follow one system message and precede the
                # initial user message. Skill compaction preserves that prefix.
                steering.bind(agent.steer, history_start=len(prior_exmekh) + 2)
            response = agent.run(user_message=content)
        except BaseException:
            event_handler.finish("error")
            raise
        else:
            event_handler.finish()
        finally:
            if steering is not None:
                steering.close()

        if on_raw_history:
            publish_multimodal_history(
                response.history,
                [serialize_skill_execution(s) for s in response.skill_executions],
            )

        return str(response.get_message() or "")

    else:
        # Quick chat using Client
        client = Client(
            provider=provider,
            system_message=SystemMessage(load_prompt("quick_chat")),
        )
        prior_exmekh = convert_to_exmekh_messages(
            history, current_user_prompt=user_prompt
        )
        if prior_exmekh:
            try:
                client.replace_history_from(1, prior_exmekh)
            except Exception as e:  # noqa: BLE001 - legacy history is best-effort
                logger.warning(
                    app_text(
                        "errors_agents.agent_runner.failed_to_seed_history_into_client",
                        value_1=e,
                    )
                )

        event_handler = QuickChatEventHandler(on_token, on_reasoning)

        response = client.call(
            ClientRequest(
                UserMessage(content),
                streaming=True,
                on_event=event_handler,
            )
        )
        final_text = event_handler.finish(str(response.get_message() or ""))

        if on_raw_history:
            publish_raw_history(
                on_raw_history,
                [exmekh_message_to_raw_dict(item) for item in client.get_history()],
                [],
            )

        return final_text


async def run_agent_turn(
    user_prompt: str,
    model: str,
    on_token: Callable[[str], None],
    on_reasoning: Callable[..., None],
    on_tool_event: Callable[[ToolEvent], None],
    agent_mode: str | None = None,
    on_skill_event: Callable[[SkillNode, str], None] | None = None,
    on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
    | None = None,
    history: Any | None = None,
    agent_context_id: str | None = None,
    on_sub_agent_event: Callable[[SubAgentEvent], None] | None = None,
    sub_agent_history: list[SubAgentConversation] | None = None,
    settings: ChatSettings | None = None,
    on_context: Callable[[ProviderContextUsage], None] | None = None,
    on_request_manual_approval: Callable[[str], bool] | None = None,
    on_command_approved: CommandApprovedCallback | None = None,
    steering: TurnSteering | None = None,
    on_steering_added: Callable[[], None] | None = None,
    image_urls: list[str] | None = None,
) -> str:
    """Asynchronously runs the agent turn in a background worker thread."""
    loop = asyncio.get_running_loop()

    mode = agent_mode or config.agent_mode
    settings = settings or ChatSettings()
    cancellation_event = threading.Event()

    def request_manual_approval(callback: Callable[[str], bool], command: str) -> bool:
        raise_if_cancelled(cancellation_event)
        try:
            approved = callback(command)
        except asyncio.CancelledError:
            cancellation_event.set()
            raise
        raise_if_cancelled(cancellation_event)
        return approved

    def _call_sync() -> str:
        return execute_agent_turn_sync(
            user_prompt=user_prompt,
            model=model or config.default_model,
            base_url=config.base_url,
            api_key=config.api_key,
            api_type=config.api_type,
            reasoning_effort_str=config.reasoning_effort,
            agent_mode=mode,
            capabilities_path_str=settings.capabilities_path
            or config.capabilities_path,
            name_hds=settings.name_hds or config.name_hds,
            agent_profile=settings.agent_profile,
            agent_design=settings.agent_design,
            capabilities_paths=settings.capability_paths(config.capabilities_path),
            on_token=on_token,
            on_reasoning=on_reasoning,
            on_tool_event=on_tool_event,
            on_skill_event=on_skill_event,
            on_raw_history=on_raw_history,
            history=history,
            agent_context_id=agent_context_id,
            on_sub_agent_event=on_sub_agent_event,
            sub_agent_history=sub_agent_history,
            work_root=settings.work_root or config.work_root,
            coding_agent_persistent_history=(
                config.coding_agent_persistent_history
                if settings.coding_agent_persistent_history is None
                else settings.coding_agent_persistent_history
            ),
            on_context=on_context,
            on_request_manual_approval=partial(
                request_manual_approval, on_request_manual_approval
            )
            if on_request_manual_approval is not None
            else None,
            on_command_approved=on_command_approved,
            tool_approval_mode=settings.tool_approval_mode,
            approved_commands_history=list(settings.approved_commands_history),
            tool_approval_raw_history=list(settings.tool_approval_raw_history),
            cancellation_event=cancellation_event,
            steering=steering,
            on_steering_added=on_steering_added,
            image_urls=image_urls,
        )

    try:
        return await loop.run_in_executor(None, _call_sync)
    except asyncio.CancelledError:
        cancellation_event.set()
        raise
    finally:
        if steering is not None:
            steering.close()


def generate_chat_title_sync(
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
) -> str:
    """Generate a title without overlapping another local-provider request."""
    with _provider_request_lock:
        return _generate_chat_title_sync_unlocked(
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_type=api_type,
        )


def _generate_chat_title_sync_unlocked(
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
) -> str:
    """Generate a concise chat title from the user's first prompt."""
    if not user_prompt or not user_prompt.strip():
        return app_text("execution.agent_runner.new_conversation")

    clean_prompt = user_prompt.strip().replace("\n", " ")
    fallback_title = clean_prompt[:30] + ("..." if len(clean_prompt) > 30 else "")

    effective_model = model or config.default_model
    effective_api = api_type or config.api_type

    try:
        provider = LocalProvider(
            model=effective_model,
            api="responses" if effective_api == "responses" else "chat_completions",
            base_url=base_url or config.base_url,
            api_key=api_key or config.api_key,
        )

        client = Client(
            provider=provider,
            system_message=SystemMessage(load_prompt("chat_title")),
        )

        request = ClientRequest(
            message=UserMessage(
                load_prompt("chat_title_input").format(user_prompt=user_prompt)
            ),
            extra={
                "temperature": 0.3,
                **(
                    {"max_output_tokens": 256, "reasoning": {"effort": "none"}}
                    if effective_api == "responses"
                    else {"max_tokens": 256, "reasoning_effort": "none"}
                ),
            },
        )

        response = client.call_structured(request, ChatTitle, strict=True)

        title = response.get_message().title.strip().strip("\"'")

        if title:
            return title
    except Exception as e:  # noqa: BLE001 - title generation intentionally falls back
        logger.warning(
            app_text(
                "errors_agents.agent_runner.error_during_structured_title_generation_via_agentic_client",
                value_1=e,
            )
        )

    return fallback_title


async def generate_chat_title(
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
) -> str:
    """Asynchronously generates a chat title using the agentic Client in a worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        generate_chat_title_sync,
        user_prompt,
        model or config.default_model,
        base_url or config.base_url,
        api_key or config.api_key,
        api_type or config.api_type,
    )
