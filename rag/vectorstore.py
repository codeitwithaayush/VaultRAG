import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import VECTORSTORE_PATH


def build_vectorstore(folder_path, embeddings):

    documents = []

    for file in os.listdir(folder_path):

        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(
                folder_path,
                file
            )

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            for doc in docs:

                doc.metadata["source_file"] = file

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    if len(documents) == 0:
        raise Exception(
            f"No PDFs found in folder: {folder_path}"
        )

    chunks = splitter.split_documents(
        documents
    )

    if len(chunks) == 0:
        raise Exception(
            "No chunks created."
        )

    if os.path.exists(VECTORSTORE_PATH):

        shutil.rmtree(VECTORSTORE_PATH)

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local(
        VECTORSTORE_PATH
    )

    return db


def load_vectorstore(embeddings):

    faiss_file = os.path.join(
        VECTORSTORE_PATH,
        "index.faiss"
    )

    pkl_file = os.path.join(
        VECTORSTORE_PATH,
        "index.pkl"
    )

    if (
        not os.path.exists(faiss_file)
        or
        not os.path.exists(pkl_file)
    ):
        return None

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
