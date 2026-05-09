from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_PATH
from src.llm import embeddings


def load_vector_store():
    vector_path = Path(VECTORSTORE_PATH)

    if not (vector_path / "index.faiss").exists() or not (vector_path / "index.pkl").exists():
        raise RuntimeError(
            "Vector store not found. Add PDFs and run: python -m src.ingestion.ingest"
        )

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever():
    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
