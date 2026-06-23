from __future__ import annotations

import logging
import time
from typing import Literal

import requests
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen_service")

app = FastAPI(title="Local Qwen Foundation Model Service")

_OLLAMA_URL = "http://127.0.0.1:11434"
_MODEL = "qwen3.6:latest"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] | str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = 512
    think: bool = False


class ChatResponse(BaseModel):
    model: str
    content: str
    done: bool = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionRequest(ChatRequest):
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


def _ollama_post(path: str, payload: dict, timeout_s: float = 180.0) -> dict:
    url = f"{_OLLAMA_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed at {url}: {exc}",
        ) from exc


def _ollama_get(path: str, timeout_s: float = 5.0) -> dict:
    url = f"{_OLLAMA_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is not reachable at {url}: {exc}",
        ) from exc


def _chat(req: ChatRequest) -> ChatResponse:
    model = req.model or _MODEL
    payload = {
        "model": model,
        "messages": [message.model_dump() for message in req.messages],
        "stream": False,
        "think": req.think,
        "options": {
            "temperature": req.temperature,
        },
    }
    if req.max_tokens is not None:
        payload["options"]["num_predict"] = req.max_tokens

    data = _ollama_post("/api/chat", payload)
    message = data.get("message") or {}
    return ChatResponse(
        model=str(data.get("model", model)),
        content=str(message.get("content") or message.get("thinking") or ""),
        done=bool(data.get("done", True)),
    )


@app.get("/health")
async def health():
    tags = _ollama_get("/api/tags")
    models = [item.get("name") for item in tags.get("models", [])]
    return {
        "ok": _MODEL in models,
        "service": "qwen",
        "model": _MODEL,
        "ollama_url": _OLLAMA_URL,
        "available_models": models,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    return _chat(req)


@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported yet")
    result = _chat(req)
    return ChatCompletionResponse(
        id=f"qwen-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=result.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.content),
            )
        ],
    )


def main(
    model: str = "qwen3.6:latest",
    ollama_url: str = "http://127.0.0.1:11434",
    host: str = "127.0.0.1",
    port: int = 8112,
):
    global _MODEL, _OLLAMA_URL
    _MODEL = model
    _OLLAMA_URL = ollama_url

    logger.info("Starting local Qwen service with model %s via %s", _MODEL, _OLLAMA_URL)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
