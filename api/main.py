import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from src.config import UPLOADED_DOCS_PATH
from src.graph import build_chatbot, build_plain_chatbot
from src.ingestion.ingest import create_vector_store
from src.memory import close_checkpointer, delete_thread, retrieve_all_threads, setup_checkpointer


chatbots = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbots

    checkpointer = await setup_checkpointer()
    chatbots = {
        "crag": build_chatbot(checkpointer),
        "chat": build_plain_chatbot(checkpointer),
    }
    yield
    await close_checkpointer()


app = FastAPI(
    title="CRAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str = "anwes"
    mode: str = "chat"


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    user_id: str
    mode: str
    verdict: str
    reason: str
    route: str
    web_query: str


class UploadResponse(BaseModel):
    filename: str
    message: str


class ThreadMessage(BaseModel):
    role: str
    content: str


class ThreadResponse(BaseModel):
    thread_id: str
    mode: str
    messages: list[ThreadMessage]


class DeleteThreadResponse(BaseModel):
    thread_id: str
    deleted: bool


def build_chat_config(request: ChatRequest):
    return {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }


def get_chatbot(mode: str):
    return chatbots.get(mode, chatbots.get("chat"))


def serialize_messages(messages: list[BaseMessage]):
    serialized = []

    for message in messages:
        role = getattr(message, "type", "assistant")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        serialized.append(
            ThreadMessage(
                role=role,
                content=getattr(message, "content", "") or "",
            )
        )

    return serialized


def format_sse(event: str, data: dict):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def is_generate_event(event: dict):
    metadata = event.get("metadata") or {}
    node = metadata.get("langgraph_node")
    checkpoint_ns = metadata.get("langgraph_checkpoint_ns", "")

    return node == "generate" or ":generate:" in checkpoint_ns or checkpoint_ns.endswith(":generate")


def get_chunk_text(event: dict):
    chunk = (event.get("data") or {}).get("chunk")
    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)

    return ""


def extract_debug_payload(value):
    if not isinstance(value, dict):
        return {}

    debug = {}
    for key in ("answer", "verdict", "reason", "route", "web_query"):
        if key in value:
            debug[key] = value.get(key) or ""

    for nested in value.values():
        if isinstance(nested, dict):
            debug.update(extract_debug_payload(nested))

    return debug


def get_active_node_label(event: dict):
    metadata = event.get("metadata") or {}
    node = metadata.get("langgraph_node")

    labels = {
        "retrieve": "retrieving docs",
        "eval_each_doc": "evaluating chunks",
        "rewrite_query": "rewriting query",
        "web_search": "searching web",
        "refine": "refining context",
        "generate": "generating answer",
        "memory_subgraph": "saving memory",
        "tool_router": "routing",
        "calculator": "calculating",
        "wikipedia": "checking wikipedia",
        "arxiv": "checking arxiv",
    }

    return labels.get(node)


@app.get("/")
def root():
    return {"message": "CRAG Chatbot API is running"}


@app.get("/threads")
async def get_threads():
    return {"threads": await retrieve_all_threads()}


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": "anwes",
        }
    }

    try:
        snapshot = await get_chatbot("chat").aget_state(config)
        values = snapshot.values or {}
        messages = values.get("messages", []) or []
        mode = values.get("route") or "chat"

        return ThreadResponse(
            thread_id=thread_id,
            mode=mode if mode in {"chat", "crag"} else "chat",
            messages=serialize_messages(messages),
        )
    except Exception:
        return ThreadResponse(thread_id=thread_id, mode="chat", messages=[])


@app.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
async def remove_thread(thread_id: str):
    await delete_thread(thread_id)
    return DeleteThreadResponse(thread_id=thread_id, deleted=True)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs(UPLOADED_DOCS_PATH, exist_ok=True)
    filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOADED_DOCS_PATH, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    create_vector_store()

    return UploadResponse(
        filename=filename,
        message=f"{filename} uploaded and indexed.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = build_chat_config(request)
    chatbot = get_chatbot(request.mode)

    result = await chatbot.ainvoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "route": request.mode,
        },
        config=config,
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        thread_id=request.thread_id,
        user_id=request.user_id,
        mode=request.mode,
        verdict=result.get("verdict", ""),
        reason=result.get("reason", ""),
        route=result.get("route", ""),
        web_query=result.get("web_query", ""),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    config = build_chat_config(request)
    chatbot = get_chatbot(request.mode)

    async def event_generator():
        debug = {
            "answer": "",
            "verdict": "",
            "reason": "",
            "route": "",
            "web_query": "",
        }
        last_status = None

        try:
            async for event in chatbot.astream_events(
                {
                    "messages": [HumanMessage(content=request.message)],
                    "route": request.mode,
                },
                config=config,
                version="v2",
            ):
                event_type = event.get("event")
                status = get_active_node_label(event) if request.mode == "crag" else None

                if status and status != last_status and event_type in {
                    "on_chain_start",
                    "on_chain_stream",
                    "on_chat_model_start",
                    "on_chat_model_stream",
                    "on_tool_start",
                    "on_tool_stream",
                }:
                    last_status = status
                    yield format_sse("status", {"status": status})

                if event_type == "on_chat_model_stream" and is_generate_event(event):
                    token = get_chunk_text(event)
                    if token:
                        debug["answer"] += token
                        yield format_sse("token", {"token": token})

                if event_type in {"on_chain_end", "on_tool_end"}:
                    output = (event.get("data") or {}).get("output")
                    debug.update(extract_debug_payload(output))

            yield format_sse(
                "done",
                {
                    "answer": debug.get("answer", ""),
                    "thread_id": request.thread_id,
                    "user_id": request.user_id,
                    "mode": request.mode,
                    "verdict": debug.get("verdict", ""),
                    "reason": debug.get("reason", ""),
                    "route": debug.get("route", ""),
                    "web_query": debug.get("web_query", ""),
                },
            )
        except Exception as exc:
            yield format_sse("error", {"error": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
