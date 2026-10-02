from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

from exmekh_agent.exmekh_bridge.pasted_input import create_staged_attachment_from_file
from exmekh_agent.state.session_manager import session_manager

upload_router = APIRouter(prefix="/_exmekh")


@upload_router.post("/upload")
async def upload_files(request: Request) -> JSONResponse:
    """Stage files for the conversation that initiated the upload."""
    async with request.form() as form:
        session_id = form.get("session_id")
        session = session_manager.get_session(str(session_id or ""))
        if session is None:
            return JSONResponse(
                {"error": "Upload conversation was not found"}, status_code=404
            )
        storage = session_manager.temp_storage(session)
        files = form.getlist("files") or form.getlist("file")
        loop = asyncio.get_running_loop()
        staged = []
        for upload in files:
            if not isinstance(upload, UploadFile):
                continue
            data = await upload.read()
            try:
                attachment = await loop.run_in_executor(
                    None,
                    partial(
                        create_staged_attachment_from_file,
                        upload.filename or "uploaded_file",
                        data,
                        storage=storage,
                    ),
                )
            except (OSError, ValueError) as error:
                for previous in staged:
                    previous.delete_file()
                return JSONResponse(
                    {"error": f"Upload failed: {error}"}, status_code=400
                )
            if not session_manager.owns_session(session):
                for previous in [*staged, attachment]:
                    previous.delete_file()
                return JSONResponse(
                    {"error": "Upload conversation was deleted"}, status_code=404
                )
            staged.append(attachment)
        return JSONResponse([attachment.upload_payload() for attachment in staged])
