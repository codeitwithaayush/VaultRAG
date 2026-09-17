import os

MODEL_NAME = os.getenv("MODEL_NAME", "docker.io/ai/qwen2.5:3B-Q4_K_M")

QWEN_API_URL = os.getenv("QWEN_API_URL", "http://localhost:12434/v1/chat/completions")

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")

TOP_K = int(os.getenv("TOP_K", "5"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

PDF_FOLDER = os.getenv("PDF_FOLDER", "/Users/avummdedhiaa/Documents/PDFs")

