import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
)

VECTORSTORE_PATH = "vectorstore"
DOCUMENTS_PATH = "documents"
UPLOADED_DOCS_PATH = "uploaded_docs"

UPPER_TH = 0.7
LOWER_TH = 0.3
