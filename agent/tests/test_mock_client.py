import pytest
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.provider import Provider

import exmekh_agent.exmekh_bridge.agents.agent_runner as ar
from exmekh_agent.exmekh_bridge.models import ToolEvent
from tests.mock_client import (
    MockClient,
    MockProvider,
    PipelineStep,
    ScenarioPipeline,
    create_default_hierarchical_pipeline,
    create_quick_chat_pipeline,
    get_global_mock_provider,
    reset_mock_provider,
    set_global_mock_provider,
    using_mock_provider,
)


def test_mock_provider_and_client_inheritance() -> None:
    provider = MockProvider()
    assert isinstance(provider, Provider)

    client = MockClient(provider=provider)
    assert isinstance(client, Client)
    assert client.mock_provider is provider


def test_mock_client_init_and_history() -> None:
    client = MockClient()
    history = client.get_history()
    assert len(history) == 1
    assert isinstance(history[0], SystemMessage)

    dump = client.dump_history()
    assert len(dump) == 1
    assert dump[0]["role"] == "system"

    copy_h = client.copy_history()
    assert len(copy_h) == 1


def test_mock_client_call_synchronous_pipeline() -> None:
    pipeline = create_default_hierarchical_pipeline()
    provider = MockProvider(pipeline=pipeline)
    client = MockClient(provider=provider)

    # Step 1: Diagnostic command
    req1 = ClientRequest(UserMessage("Inizializza pipeline ed esegui diagnostica"))
    res1 = client.call(req1)
    assert "Diagnostica di sistema completata" in str(res1.get_message())
    assert "SYS_DIAG_OK" in str(res1.get_message())
    assert len(client.get_history()) == 3  # System, User, Assistant

    # Step 2: Linked consecutive command
    req2 = ClientRequest(UserMessage("Approfondisci con la sotto-skill i dati"))
    res2 = client.call(req2)
    assert "Elaborazione gerarchica completata" in str(res2.get_message())
    assert len(client.get_history()) == 5

    # Step 3: Synthesis report
    req3 = ClientRequest(UserMessage("Genera il report finale consolidato"))
    res3 = client.call(req3)
    assert "Report Finale della Pipeline Consecutiva" in str(res3.get_message())
    assert len(client.get_history()) == 7
    assert pipeline.is_completed is True


def test_mock_provider_streaming_events() -> None:
    pipeline = create_default_hierarchical_pipeline()
    provider = MockProvider(pipeline=pipeline)
    client = MockClient(provider=provider)

    events: list[ProviderStreamEvent] = []

    class RecordingEventHandler(ProviderEventHandler):
        def handle(self, event: ProviderStreamEvent) -> None:
            events.append(event)

    req = ClientRequest(
        UserMessage("Inizializza diagnostica"),
        streaming=True,
        on_event=RecordingEventHandler(),
    )
    client.call(req)

    event_types = [ev.type for ev in events]
    assert ProviderStreamEventType.START in event_types
    assert ProviderStreamEventType.REASONING in event_types
    assert ProviderStreamEventType.RESPONSE in event_types
    assert ProviderStreamEventType.END in event_types


def test_mock_client_execute_turn_callbacks() -> None:
    client = MockClient()
    tokens: list[str] = []
    reasoning: list[tuple[str, str, int]] = []
    tools: list[ToolEvent] = []
    skills: list[tuple[str, str]] = []

    res = client.execute_turn(
        user_prompt="Inizializza pipeline di diagnostica",
        on_token=lambda t: tokens.append(t),
        on_reasoning=lambda c, p="/", d=0: reasoning.append((c, p, d)),
        on_tool_event=lambda ev: tools.append(ev),
        on_skill_event=lambda node, act: skills.append((node.path, act)),
    )

    assert "Diagnostica" in res
    assert len(tokens) > 0
    assert len(reasoning) >= 2
    assert reasoning[0][1] == "/"  # Root path
    assert len(tools) == 1
    assert tools[0].tool_name == "mock-tool"


def test_mock_client_consecutive_commands_linked_state() -> None:
    """Tests a custom pipeline where step 2 explicitly consumes state produced by step 1."""
    pipeline = ScenarioPipeline(name="linked_state_pipeline")

    step1 = PipelineStep(
        name="calc_step",
        prompt_trigger="calcola somma 10 e 20",
        response_tokens=["Risultato parziale: 30"],
        final_response="Risultato parziale: 30",
        metadata={"sum": 30},
    )

    step2 = PipelineStep(
        name="mult_step",
        prompt_trigger=lambda p: "moltiplica per 2" in p,
        response_tokens=["Risultato moltiplicato: 60"],
        final_response="Risultato moltiplicato: 60",
    )

    pipeline.add_step(step1).add_step(step2)
    client = MockClient(pipeline=pipeline)

    res1 = client.call(ClientRequest(UserMessage("calcola somma 10 e 20")))
    pipeline.state["sum"] = 30
    assert "30" in str(res1.get_message())

    res2 = client.call(ClientRequest(UserMessage("ora moltiplica per 2 il risultato")))
    assert "60" in str(res2.get_message())


def test_mock_client_simulated_exception() -> None:
    pipeline = ScenarioPipeline(name="error_pipeline")
    pipeline.add_step(
        PipelineStep(
            name="error_step",
            prompt_trigger="provoca errore",
            exception=ConnectionError("Simulazione errore di connessione provider"),
        )
    ).add_step(
        PipelineStep(
            name="recovery_step",
            prompt_trigger="riprova",
            response_tokens=["Operazione recuperata con successo dopo l'errore."],
            final_response="Operazione recuperata con successo dopo l'errore.",
        )
    )
    client = MockClient(pipeline=pipeline)

    # 1. First call fails with ConnectionError
    with pytest.raises(ConnectionError, match="Simulazione errore"):
        client.call(ClientRequest(UserMessage("provoca errore")))

    # 2. Second linked call recovers
    res_recovery = client.call(ClientRequest(UserMessage("riprova l'operazione")))
    assert "recuperata con successo" in str(res_recovery.get_message())

    # 3. Test execute_turn with error on fresh pipeline
    pipeline.reset()
    with pytest.raises(ConnectionError, match="Simulazione errore"):
        client.execute_turn(
            user_prompt="provoca errore",
            on_token=lambda _: None,
            on_reasoning=lambda *args: None,
            on_tool_event=lambda _: None,
        )


def test_mock_client_fallback_auto_step() -> None:
    client = MockClient(pipeline=ScenarioPipeline("empty"))
    res = client.execute_turn(
        user_prompt="Un prompt non mappato",
        on_token=lambda _: None,
        on_reasoning=lambda *args: None,
        on_tool_event=lambda _: None,
    )
    assert "Un prompt non mappato" in res


def test_quick_chat_pipeline() -> None:
    qc_pipe = create_quick_chat_pipeline()
    client = MockClient(pipeline=qc_pipe)

    r1 = client.call(ClientRequest(UserMessage("Ciao")))
    assert "Quick Chat" in str(r1.get_message())
    r2 = client.call(ClientRequest(UserMessage("Secondo comando")))
    assert "secondo livello" in str(r2.get_message())
    r3 = client.call(ClientRequest(UserMessage("Terzo comando")))
    assert "completata con successo" in str(r3.get_message())


def test_global_mock_provider_registry() -> None:
    reset_mock_provider()
    p1 = get_global_mock_provider()
    assert isinstance(p1, MockProvider)

    custom_provider = MockProvider(pipeline=create_quick_chat_pipeline())
    set_global_mock_provider(custom_provider)
    assert get_global_mock_provider() is custom_provider

    reset_mock_provider()
    assert get_global_mock_provider() is not custom_provider


@pytest.mark.asyncio
async def test_agent_runner_installed_mock_provider() -> None:
    """Verifies that install_mock_provider bridges agent_runner without modifying production config."""
    provider = MockProvider()
    with using_mock_provider(provider):
        is_alive, models = await ar.check_provider_health()
        assert is_alive is True
        assert "mock-pipeline-model" in models

        streamed: list[str] = []
        reasoning: list[tuple[str, str, int]] = []
        tools: list[ToolEvent] = []
        res = await ar.run_agent_turn(
            user_prompt="Inizializza pipeline di diagnostica",
            model="mock-pipeline-model",
            on_token=lambda t: streamed.append(t),
            on_reasoning=lambda c, p="/", d=0: reasoning.append((c, p, d)),
            on_tool_event=lambda ev: tools.append(ev),
        )
        assert "Diagnostica di sistema completata" in res
        assert "".join(streamed) != ""
        assert len(tools) == 1
        assert tools[0].tool_name == "mock-tool"


def test_mock_provider_context_restores_outer_provider_after_failure() -> None:
    original = ar.LocalProvider
    first, second = MockProvider(), MockProvider()
    with using_mock_provider(first):
        assert ar.LocalProvider() is first
        with pytest.raises(RuntimeError, match="failed work"):
            with using_mock_provider(second):
                assert ar.LocalProvider() is second
                raise RuntimeError("failed work")
        assert ar.LocalProvider() is first
    assert ar.LocalProvider is original
