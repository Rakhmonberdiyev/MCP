"""FastAPI web server for the AI Agent Dashboard.

Shares the same Redis (STM) and Mem0 (LTM) instances as the Telegram bot.

Run from the project root:
    uvicorn Web.backend.main:app --reload --port 8000

Or from this directory:
    uvicorn main:app --reload --port 8000
"""

import sys
import os

# Add project root to import path so we can reuse agent, config, memory.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ui
from agent import process_turn, initialize
from memory.session import (
    get_sessions_list,
    get_current_session_id,
    create_session,
    switch_session,
    get_session_history,
)


def _service_unavailable(exc: Exception) -> HTTPException:
    """Convert an infrastructure error (Redis, Qdrant, etc.) to a clean 503."""
    return HTTPException(status_code=503, detail=f"Service unavailable: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await initialize()
    yield


app = FastAPI(title="AI Agent Dashboard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Session endpoints ──────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions(user_id: str):
    try:
        current  = await get_current_session_id(user_id)  # creates session if user has none
        sessions = await get_sessions_list(user_id)        # now includes the just-created session
        return {"sessions": sessions, "current_session_id": current}
    except Exception as exc:
        raise _service_unavailable(exc)


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str, user_id: str):
    try:
        history = await get_session_history(user_id, session_id)
        return {"session_id": session_id, "history": history}
    except Exception as exc:
        raise _service_unavailable(exc)


@app.post("/api/sessions")
async def new_session(user_id: str = Query(...)):
    try:
        sid = await create_session(user_id)
        return {"session_id": sid}
    except Exception as exc:
        raise _service_unavailable(exc)


@app.post("/api/sessions/{session_id}/switch")
async def switch(session_id: str, user_id: str = Query(...)):
    try:
        ok = await switch_session(user_id, session_id)
    except Exception as exc:
        raise _service_unavailable(exc)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ── SSE chat endpoint ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str | None = None
    deepthink: bool = False


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Process a chat turn and stream tokens via Server-Sent Events.

    Event types:
      {"type": "token",  "content": "..."}          — partial response chunk
      {"type": "done",   "response": "...",
                         "metadata": {...}}          — final full response + metadata
      {"type": "error",  "error": "..."}             — pipeline error
    """
    try:
        if req.session_id:
            await switch_session(req.user_id, req.session_id)
    except Exception as exc:
        raise _service_unavailable(exc)

    queue: asyncio.Queue = asyncio.Queue()
    metadata: dict = {}
    tool_collector: dict = {"rag": [], "web": [], "tools": []}

    async def _stream_cb(chunk: str) -> None:
        await queue.put({"type": "token", "content": chunk})

    async def _run() -> None:
        # Set ContextVars inside the task so concurrent requests get isolated log streams
        ui.set_log_queue(queue)
        ui.set_tool_collector(tool_collector)
        try:
            response = await process_turn(
                req.message,
                req.user_id,
                req.deepthink,
                stream_callback=_stream_cb,
                metadata=metadata,
            )
            full_metadata = {
                **metadata,
                "rag_results": tool_collector["rag"],
                "web_results": tool_collector["web"],
                "tool_calls": tool_collector["tools"],
            }
            await queue.put({"type": "done", "response": response, "metadata": full_metadata})
        except asyncio.CancelledError:
            raise  # let the task cancellation propagate cleanly
        except Exception as exc:
            await queue.put({"type": "error", "error": str(exc)})

    async def _event_gen() -> AsyncIterator[str]:
        task = asyncio.create_task(_run())
        try:
            while True:
                item = await queue.get()
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item["type"] in ("done", "error"):
                    break
        except GeneratorExit:
            pass  # client disconnected — fall through to finally
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
