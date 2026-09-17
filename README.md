# Local-RAG-Qwen

An offline Retrieval-Augmented Generation (RAG) system utilizing local HuggingFace embeddings and a local Qwen LLM to perform accurate question answering over your PDF documents.

## Architecture

This project implements a **Hybrid Dense-Sparse Retrieval** pipeline with an intelligent routing mechanism:
1. **Dense Retrieval**: Utilizes `langchain-huggingface` with the `BAAI/bge-small-en-v1.5` embeddings model and a local `FAISS` vector database.
2. **Sparse Retrieval**: Uses a `BM25` retrieval index for keyword-based matches.
3. **Hybrid Search Fusion**: Combines candidate documents from both dense and sparse sources using **Reciprocal Rank Fusion (RRF)** to produce a unified, ranked list of relevant context.
4. **Prompt Routing Step**: Uses a lenient classification prompt to check if the question matches the topics/keywords in the retrieved context.
   - If **YES**, it enters **RAG Mode** to answer the question using the context.
   - If **NO**, it routes to **General Knowledge Mode** and answers using the model's pretrained general knowledge (without showing empty source citations).

---

## Features

- **Local & Offline**: All computations (embeddings generation, vector search, and model inference) run locally.
- **Hybrid Retrieval**: Combines semantic meaning (dense) and keyword exact matches (sparse) for maximum accuracy.
- **OpenAI-Compatible API**: Implements standard `/v1/chat/completions` and `/v1/models` endpoints, allowing seamless integration with UIs like **Open WebUI**.
- **Dynamic Re-indexing**: Supports reloading and rebuilding document indexes on the fly.

---

## Tech Stack

- **Frameworks**: FastAPI, Uvicorn, LangChain
- **Embeddings Model**: `BAAI/bge-small-en-v1.5`
- **Vector Database**: FAISS (CPU)
- **Sparse Index**: BM25 (`rank-bm25`)
- **LLM**: Qwen 2.5 3B (run locally via Ollama or custom llama.cpp container)

---

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed and the local LLM running on port `12434`.

### Installation

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/avummdedhiaa24-ship-it/Local-RAG-Qwen.git
   cd Local-RAG-Qwen
   ```

2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the RAG Server

Run the Uvicorn FastAPI server:
```bash
python run.py
```
By default, the server runs on `http://127.0.0.1:8000`.

---

## API Endpoints

- **`GET /status`**: Returns the path of the currently loaded document directory.
- **`POST /load-folder`**: Loads and indexes PDF documents from a specified directory.
  - *Payload*: `{"folder": "/path/to/pdfs"}`
- **`POST /rebuild`**: Re-scans the PDF directory and rebuilds the FAISS/BM25 search indexes.
- **`GET /v1/models`**: OpenAI-compatible endpoint returning the active model ID (`Local-RAG-Qwen`).
- **`POST /v1/chat/completions`**: OpenAI-compatible chat completions endpoint.

---

## Running with Docker Compose (Recommended)

You can run both the local RAG FastAPI backend and the Open WebUI chatbot together using Docker Compose.

1. **Start the stack**:
   ```bash
   docker compose up -d --build
   ```
   *Note: Ensure your local LLM (Ollama/Qwen) is running on port `12434` on the host.*

2. **Access the chatbot**:
   - The **Mockup passenger portal with the AskDisha 2.0 style floating chat widget** is served at: `http://localhost:8000`
   - The standalone **Open WebUI dashboard** is served at: `http://localhost:3000`
   - The widget in the portal dynamically loads the Open WebUI instance inside its iframe.

---

## Alternative: Running services individually

### 1. Run the RAG Server locally
```bash
python run.py
```

### 2. Run Open WebUI in Docker
```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  --name open-webui \
  --restart always \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:latest
```


---

## Integrated Mockup Site & Floating Chat Widget

This project includes a mockup of a public portal (an Indian Railways / IRCTC-inspired theme) served directly at the root of the RAG API (`http://localhost:8000/`).

It features:
- **AskDisha 2.0 Style Chat Widget**: A premium floating chat widget located in the bottom-right corner.
- **Embedded Open WebUI**: Clicking the launcher button expands an overlay container that displays the actual running Open WebUI instance (`http://localhost:3000`) via an iframe. This allows users to access native Open WebUI chats, history, settings, and accounts directly within the mockup portal.
- **Prerequisites**: Ensure both the RAG API server (`python run.py`) and the Open WebUI Docker container are running.
