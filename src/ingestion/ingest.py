import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCUMENTS_PATH, UPLOADED_DOCS_PATH, VECTORSTORE_PATH
from src.llm import embeddings


def load_documents_from_folder(folder_path):
    docs = []

    if not os.path.exists(folder_path):
        return docs

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            path = os.path.join(folder_path, file)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    return docs


def load_all_documents():
    docs = []
    docs.extend(load_documents_from_folder(DOCUMENTS_PATH))
    docs.extend(load_documents_from_folder(UPLOADED_DOCS_PATH))
    return docs


def create_vector_store():
    docs = load_all_documents()

    if not docs:
        print("No PDF found. Add PDF in documents/ or uploaded_docs/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(docs)

    for doc in chunks:
        doc.page_content = doc.page_content.encode("utf-8", "ignore").decode(
            "utf-8",
            "ignore",
        )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTORSTORE_PATH)

    print("Vector DB created/updated successfully.")


if __name__ == "__main__":
    create_vector_store()
