# Adaptive Chatbot

Adaptive Chatbot is a FastAPI and React application for conversational AI with two modes:

- Normal chat for general questions.
- PDF CRAG for document-grounded answers with retrieval, grading, query rewriting, web search fallback, and streamed status updates.

The backend is built with LangGraph, LangChain, Google Gemini, FAISS, Tavily search, and Postgres-backed checkpointing. The frontend is a Vite React chat UI with PDF upload, streaming responses, conversation history, and debug details for CRAG runs.

## Features

- Streaming chat responses over server-sent events.
- Upload and index PDF files into a local FAISS vector store.
- Adaptive routing for calculator, Wikipedia, arXiv, PDF retrieval, and web search tools.
- Conversation checkpointing by thread ID.
- Long-term memory store with Postgres fallback behavior.
- React UI for normal chat, PDF chat, uploads, thread selection, and thread deletion.

## Project Structure

```text
api/                  FastAPI app and API routes
frontend/             Vite React frontend
src/                  LangGraph chatbot, nodes, tools, memory, and ingestion
documents/            Source documents for indexing
uploaded_docs/        PDFs uploaded through the UI
vectorstore/          Local FAISS index files
requirements.txt      Python dependencies
```

## Requirements

- Python 3.9+
- Node.js and npm
- PostgreSQL
- API keys for Gemini and Tavily

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
TAVILY_API_KEY=your_tavily_api_key
POSTGRES_URI=postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable
```

The frontend can optionally use:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Backend Setup

Install Python dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start PostgreSQL and make sure `POSTGRES_URI` points to a reachable database.

Run the API:

```bash
uvicorn api.main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## Frontend Setup

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the development server:

```bash
npm run dev
```

The UI runs at `http://127.0.0.1:5173`.

## API Endpoints

- `GET /` - health check.
- `GET /threads` - list saved conversation thread IDs.
- `GET /threads/{thread_id}` - load a conversation.
- `DELETE /threads/{thread_id}` - delete a conversation.
- `POST /upload` - upload and index a PDF file.
- `POST /chat` - send a non-streaming chat request.
- `POST /chat/stream` - send a streaming chat request.

## Usage

1. Start PostgreSQL.
2. Start the FastAPI backend.
3. Start the React frontend.
4. Open the frontend in the browser.
5. Use "Normal chat" for general chat or upload a PDF and switch to "PDF CRAG" for document-grounded answers.

## Notes

- Uploaded PDFs are stored in `uploaded_docs/`.
- Vector indexes are stored in `vectorstore/`.
- Conversation state is checkpointed through Postgres.
- The current default `user_id` in the frontend and API is `anwes`.
