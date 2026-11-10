"""Controlled SSE on the real library/tool/frontend path, without model compute."""

import json
import os
import time
from threading import Event
from uuid import uuid4

import httpx2
from exmekh_lib.interfaces.provider import ProviderRequestRejectedError
from nicegui import app

from tests import run_native_media_preview_agent as fixture

original_response = fixture.image_response
advance_progress = Event()


@app.post("/test/advance-image-progress")
def advance_image_progress():
    advance_progress.set()
    return {"advanced": True}


class ProgressStream(httpx2.SyncByteStream):
    def __init__(self, request):
        self.request = request

    def __iter__(self):
        fixture.generating = True
        operation_id = uuid4().hex
        for seq, fraction in ((1, 0.1), (2, 0.42)):
            event = {
                "version": 1,
                "type": "operation.progress",
                "operation_id": operation_id,
                "request_id": operation_id,
                "seq": seq,
                "elapsed_seconds": seq,
                "state": "running",
                "phase": "generation",
                "item_index": 0,
                "item_count": 2,
                "active_items": 2,
                "completed_units": seq,
                "total_units": None,
                "unit": "token",
                "overall_fraction": fraction,
                "overall_kind": "estimated",
            }
            yield (
                "event: operation.progress\ndata: " + json.dumps(event) + "\n\n"
            ).encode()
            time.sleep(0.1)
        advance_progress.wait(30)
        event.update(seq=3, overall_fraction=0.64, elapsed_seconds=3)
        yield (
            "event: operation.progress\ndata: " + json.dumps(event) + "\n\n"
        ).encode()
        fixture.image_release.wait(30)
        result = original_response(self.request).json()
        event.update(
            type="operation.completed",
            seq=4,
            state="completed",
            overall_fraction=1,
            overall_kind="work",
            result=result,
        )
        yield (
            "event: operation.completed\ndata: " + json.dumps(event) + "\n\n"
        ).encode()


def response(request):
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=ProgressStream(request),
    )


class Provider(fixture.PreviewProvider):
    def call(self, request):
        if self._step_counter and os.getenv("EXMEKH_AGENT_TEST_POST_IMAGE_ERROR"):
            raise ProviderRequestRejectedError(
                "Controlled vision rejection after image success"
            )
        return super().call(request)


fixture.image_response = response
fixture.agent_runner.LocalProvider = lambda *args, **kwargs: Provider()

if __name__ == "__main__":
    fixture.main()
