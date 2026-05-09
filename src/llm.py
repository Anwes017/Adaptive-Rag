import hashlib
import re
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, GEMINI_MODEL


class HashEmbeddings(Embeddings):
    """Small local embeddings so retrieval does not depend on Ollama."""

    def __init__(self, size: int = 1536):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.size
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]

        return vector


if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    raise RuntimeError("GEMINI_API_KEY is missing. Add your real Gemini API key to .env before starting the app.")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

memory_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

embeddings = HashEmbeddings()
