#VaultRAG

> An offline, privacy-first Retrieval-Augmented Generation system combining Hybrid Dense-Sparse retrieval with a local Qwen LLM — no cloud, no API keys, just your documents and your machine.

---

## Architecture

This project implements a **Hybrid Dense-Sparse Retrieval pipeline** with an intelligent routing mechanism:

| Component | Technology |
|---|---|
| Dense Retrieval | FAISS + `BAAI/bge-small-en-v1.5` embeddings |
| Sparse Retrieval | BM25 keyword index |
| Fusion Strategy | Reciprocal Rank Fusion (RRF) |
| LLM | Qwen 2.5 3B (via Ollama / llama.cpp) |
| API Layer | FastAPI (OpenAI-compatible) |

### Prompt Routing
- **RAG Mode** — question matches retrieved context → answers using your documents
- **General Knowledge Mode** — no match → answers from model's pretrained knowledge

---

## Features

- 🔒 **Fully Local & Offline** — embeddings, vector search, and inference run on your machine
- 🔍 **Hybrid Retrieval** — semantic (dense) + keyword (sparse) for maximum accuracy
- 🔌 **OpenAI-Compatible API** — plug into Open WebUI or any OpenAI-compatible client
- ♻️ **Dynamic Re-indexing** — rebuild document indexes on the fly
- 🚉 **Mockup Portal** — IRCTC-inspired UI with AskDisha 2.0 floating chat widget

---

## 🛠️ Tech Stack

- **Frameworks:** FastAPI, Uvicorn, LangChain
- **Embeddings:** `BAAI/bge-small-en-v1.5` (HuggingFace)
- **Vector DB:** FAISS (CPU)
- **Sparse Index:** BM25 (`rank-bm25`)
- **LLM:** Qwen 2.5 3B (local via Ollama or llama.cpp)

---

##  Getting Started

### Prerequisites
- Python 3.10+
- Local LLM running on port `12434`

### Installation

```bash
git clone https://github.com/yourusername/FusionRAG-Qwen.git
cd FusionRAG-Qwen

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the Server

```bash
python run.py
```

Server starts at `http://127.0.0.1:8000`

---

## 🐳 Docker Compose (Recommended)

Runs the RAG backend + Open WebUI together:

```bash
docker compose up -d --build
```

| Service | URL |
|---|---|
| Mockup Portal + Chat Widget | `http://localhost:8000` |
| Open WebUI Dashboard | `http://localhost:3000` |

> Ensure your local LLM (Ollama/Qwen) is running on port `12434` on the host.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Currently loaded document directory |
| `POST` | `/load-folder` | Load & index PDFs from a folder |
| `POST` | `/rebuild` | Rebuild FAISS + BM25 indexes |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint |

---

## 💬 Mockup Portal & Chat Widget

Served at `http://localhost:8000/` — an IRCTC-inspired public portal featuring:

- **AskDisha 2.0** style floating chat widget (bottom-right)
- Embedded Open WebUI via iframe
- Access chat history, settings, and accounts within the portal

---

## 📄 License

MIT License — free to use, modify, and distribute.