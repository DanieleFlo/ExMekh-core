"""Read opt-in DS4 timing comments without consuming the model's SSE events."""

import re
from collections.abc import Iterator
from typing import cast

from httpx2 import BaseTransport, Request, Response, SyncByteStream
from openai import DefaultHttpxClient

from exmekh_agent.exmekh_bridge.models.inference_timing import (
    InferencePhase,
    InferenceTiming,
    TimingObserver,
)

_TIMING_HEADER = "X-DS4-Stream-Timing"
_MAX_TIMING_LINE = 128
_TIMING_LINE = re.compile(
    rb": ds4-timing (prefill|decode|reasoning|tool) (\d{1,10}) (\d{1,12})\r?"
)


class _TimingStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, observe: TimingObserver) -> None:
        self._stream = stream
        self._observe = observe
        self._line = b""
        self._discard_line = False
        self._reported = False

    def _read_timing(self, line: bytes) -> InferenceTiming | None:
        if self._discard_line or len(self._line) + len(line) > _MAX_TIMING_LINE:
            return None
        match = _TIMING_LINE.fullmatch(self._line + line)
        if not match:
            return None
        phase, tokens, milliseconds = match.groups()
        count, elapsed = int(tokens), int(milliseconds)
        rate = count * 1000 / elapsed if count and elapsed else None
        return InferenceTiming(cast(InferencePhase, phase.decode()), rate)

    def _consume(self, chunk: bytes) -> Iterator[bytes]:
        forwarded = start = 0
        while (end := chunk.find(b"\n", start)) != -1:
            timing = self._read_timing(chunk[start:end])
            self._line = b""
            self._discard_line = False
            start = end + 1
            if timing is not None:
                # Let the SSE consumer process earlier events before changing phase.
                yield chunk[forwarded:start]
                forwarded = start
                self._reported = True
                self._observe(timing)
        if not self._discard_line:
            if len(self._line) + len(chunk) - start <= _MAX_TIMING_LINE:
                self._line += chunk[start:]
            else:
                self._line = b""
                self._discard_line = True
        if forwarded < len(chunk):
            yield chunk[forwarded:]

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            yield from self._consume(chunk)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            if self._reported:
                self._reported = False
                self._observe(None)


def timing_client(
    observe: TimingObserver, *, transport: BaseTransport | None = None
) -> DefaultHttpxClient:
    """Keep the SDK defaults and observe only streaming inference responses."""

    def request_timing(request: Request) -> None:
        if request.method == "POST" and request.url.path.endswith(
            ("/responses", "/chat/completions")
        ):
            request.headers[_TIMING_HEADER] = "1"

    def observe_timing(response: Response) -> None:
        if (
            _TIMING_HEADER in response.request.headers
            and response.headers.get("content-type", "").startswith("text/event-stream")
            and response.headers.get("content-encoding", "identity") == "identity"
            and isinstance(response.stream, SyncByteStream)
        ):
            response.stream = _TimingStream(response.stream, observe)

    return DefaultHttpxClient(
        transport=transport,
        event_hooks={"request": [request_timing], "response": [observe_timing]},
    )
