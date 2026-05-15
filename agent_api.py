from __future__ import annotations

import os
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.agent import Agent

load_dotenv()

app = FastAPI(title="Olist Gemma Agent API", version="1.0.0")
agent = Agent()
agent_lock = Lock()
startup_error: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    output_path: str = ""


@app.on_event("startup")
def startup() -> None:
    global startup_error
    if os.getenv("AGENT_API_PRELOAD_GEMMA", "true").lower() in {"1", "true", "yes", "on"}:
        try:
            agent.ensure_gemma()
        except Exception as exc:
            startup_error = str(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "olist-gemma-agent-api"}


@app.get("/status")
def status() -> dict[str, Any]:
    status_result = agent.gemma_runtime_status()
    status_result["preload_error"] = startup_error
    status_result["model_loaded"] = agent.gemma is not None
    return status_result


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    with agent_lock:
        return agent.answer_question(question=request.question, output_path=request.output_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_api:app",
        host=os.getenv("AGENT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_API_PORT", "8000")),
        reload=False,
    )
