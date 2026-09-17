from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag.config import PDF_FOLDER
import os

from rag.rag_engine import (
    RAGEngine
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()

try:

    engine.load_folder(
        PDF_FOLDER
    )

    print(
        f"Documents loaded from: {PDF_FOLDER}"
    )

except Exception as e:

    print(
        f"Startup warning: {e}"
    )

MODEL_NAME = "Local-RAG-Qwen"


class ChatRequest(
    BaseModel
):
    model: str
    messages: list


class FolderRequest(
    BaseModel
):
    folder: str


@app.get("/", response_class=HTMLResponse)
def read_root():

    template_path = os.path.join(
        os.path.dirname(__file__),
        "templates",
        "index.html"
    )

    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@app.get("/v1/models")
def models():

    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model"
            }
        ]
    }


@app.post("/load-folder")
def load_folder(
    req: FolderRequest
):

    engine.load_folder(
        req.folder
    )

    return {
        "status": "success"
    }


@app.get("/status")
def status():

    return {
        "folder":
        engine.current_folder
    }


@app.post("/rebuild")
def rebuild():

    engine.load_folder(
        PDF_FOLDER
    )

    return {
        "status": "rebuilt",
        "folder": PDF_FOLDER
    }


@app.post("/v1/chat/completions")
def chat(
    req: ChatRequest
):

    question = req.messages[-1][
        "content"
    ]

    result = engine.ask(
        question
    )

    return {
        "id": "rag-chat",
        "object":
        "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role":
                    "assistant",
                    "content":
                    result["answer"]
                },
                "finish_reason":
                "stop"
            }
        ]
    }
