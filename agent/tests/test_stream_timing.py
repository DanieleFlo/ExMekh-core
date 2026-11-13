from collections.abc import Iterator

import httpx2
import pytest

from exmekh_agent.exmekh_bridge.models.inference_timing import InferenceTiming
from exmekh_agent.exmekh_bridge.providers.stream_timing import timing_client


class Chunks(httpx2.SyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


def test_timing_comments_are_observed_without_modifying_stream_bytes() -> None:
    chunk1 = b": ds4-timing prefill 50 "
    chunk2 = b"500\r\n\ndata: " + b"x" * 150
    chunk3 = b": ds4-timing tool 70 100\n\n: ds4-timing reasoning 1 0\n\n"
    chunk4 = b": ds4-timing tool 90 1500\n\n: ds4-timing decode 100 2000\n\n"
    stream = Chunks(chunk1, chunk2, b"x" * 150, chunk3, chunk4, b"data: [DONE]\n\n")
    observations = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["X-DS4-Stream-Timing"] == "1"
        return httpx2.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=stream
        )

    with timing_client(
        observations.append, transport=httpx2.MockTransport(reply)
    ) as client:
        with client.stream("POST", "http://core/v1/responses") as response:
            assert b"".join(response.iter_raw()) == b"".join(stream.chunks)
    assert stream.closed
    assert observations == [
        InferenceTiming("prefill", 100.0),
        InferenceTiming("reasoning", None),
        InferenceTiming("tool", 60.0),
        InferenceTiming("decode", 50.0),
        None,
    ]


def test_timing_does_not_overtake_an_earlier_event_in_the_same_chunk() -> None:
    reasoning = b'data: {"reasoning":"choose a tool"}\n\n'
    timing = b": ds4-timing tool 0 0\n\n"
    tool = b'data: {"tool":"lookup"}\n\n'
    observed = []

    def reply(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=Chunks(reasoning + timing[:-1], timing[-1:] + tool),
        )

    with timing_client(
        observed.append, transport=httpx2.MockTransport(reply)
    ) as client:
        with client.stream("POST", "http://core/v1/responses") as response:
            chunks = response.iter_raw()
            assert next(chunks) == reasoning + timing[:-1]
            assert observed == []
            assert next(chunks) == b"\n" + tool
            assert observed == [InferenceTiming("tool", None)]
    assert observed[-1] is None


@pytest.mark.parametrize(
    ("method", "path", "content_type", "encoding", "opted_in"),
    [
        ("GET", "responses", "text/event-stream", "identity", False),
        ("POST", "embeddings", "text/event-stream", "identity", False),
        ("POST", "chat/completions", "application/json", "identity", True),
        ("POST", "chat/completions", "text/event-stream", "gzip", True),
        ("POST", "chat/completions", "text/event-stream", "identity", True),
    ],
)
def test_other_responses_and_legacy_streams_do_not_report_timing(
    method, path, content_type, encoding, opted_in
) -> None:
    observations = []
    data = b"data: {}\n\n"

    def reply(request: httpx2.Request) -> httpx2.Response:
        assert ("X-DS4-Stream-Timing" in request.headers) is opted_in
        return httpx2.Response(
            200,
            headers={"Content-Type": content_type, "Content-Encoding": encoding},
            stream=Chunks(data),
        )

    with timing_client(
        observations.append, transport=httpx2.MockTransport(reply)
    ) as client:
        with client.stream(method, f"http://core/v1/{path}") as response:
            assert b"".join(response.iter_raw()) == data
    assert observations == []
