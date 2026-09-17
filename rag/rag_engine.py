from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS
)

from .vectorstore import (
    build_vectorstore,
    load_vectorstore
)

from .retrieval import retrieve
from .generator import ask_qwen


class RAGEngine:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        try:
            self.db = load_vectorstore(
                self.embeddings
            )
            self._init_bm25()
        except Exception:
            self.db = None
            self.bm25_retriever = None

        self.current_folder = None

    def _init_bm25(self):
        if self.db is not None:
            try:
                docs = list(self.db.docstore._dict.values())
                if docs:
                    from langchain_community.retrievers import BM25Retriever
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    self.bm25_retriever.k = 10
                else:
                    self.bm25_retriever = None
            except Exception as e:
                print(f"Warning: Failed to initialize BM25: {e}")
                self.bm25_retriever = None
        else:
            self.bm25_retriever = None

    def load_folder(
        self,
        folder
    ):

        self.current_folder = folder

        self.db = build_vectorstore(
            folder,
            self.embeddings
        )
        self._init_bm25()

    def ask(
        self,
        question
    ):

        # No documents loaded
        if self.db is None:

            return {
                "answer": ask_qwen(question, system_prompt="You are a helpful AI assistant. Answer the user's question using your general knowledge."),
                "sources": []
            }

        # Retrieve relevant chunks
        results = retrieve(
            self.db,
            self.bm25_retriever,
            question
        )

        # Nothing found
        if len(results) == 0:

            return {
                "answer": ask_qwen(question, system_prompt="You are a helpful AI assistant. Answer the user's question using your general knowledge."),
                "sources": []
            }

        # Build context
        context = []
        sources = []

        total_length = 0

        for doc, score in results:

            file_name = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            page = (
                doc.metadata.get(
                    "page",
                    0
                ) + 1
            )

            text = f"""
FILE: {file_name}
PAGE: {page}

{doc.page_content}
"""

            total_length += len(text)

            if total_length > MAX_CONTEXT_CHARS:
                break

            context.append(text)

            sources.append(
                {
                    "file": file_name,
                    "page": page
                }
            )

        context_text = "\n\n".join(
            context
        )

        # Router step
        router_prompt = f"""Context:
{context_text}

Question:
{question}

Reply YES or NO:"""

        decision = ask_qwen(
            router_prompt,
            system_prompt="You are a routing assistant. If the context contains any relevant information, keywords, or topics related to the question, reply YES. Otherwise, reply NO. Reply ONLY with YES or NO."
        ).strip().upper()

        print("=" * 60)
        print("Question:", question)
        print("Router Decision:", decision)
        print("=" * 60)

        # General Knowledge Mode
        if decision.startswith("NO"):

            print("GENERAL KNOWLEDGE MODE")

            return {
                "answer": ask_qwen(question, system_prompt="You are a helpful AI assistant. Answer the user's question using your general knowledge."),
                "sources": []
            }

            # RAG Mode
        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the document context.

If the answer exists in the context,
use the context.

If the answer is not present,
answer using your own knowledge.

Question:
{question}

Context:
{context_text}

Answer:
"""

        print("\n" + "=" * 80)
        print("QUESTION:")
        print(question)

        print("\nCONTEXT:")
        print(context_text[:3000])

        print("\n" + "=" * 80)

        answer = ask_qwen(
            prompt,
            system_prompt="You are a helpful AI assistant. Answer the user's question using the provided context, or your own knowledge if it is not present in the context."
        )

        source_text = "\n".join(
            [
                f"- {s['file']} (Page {s['page']})"
                for s in sources
            ]
        )

        final_answer = f"""
{answer}

Sources:
{source_text}
"""

        return {
            "answer": final_answer.strip(),
            "sources": sources
        }
