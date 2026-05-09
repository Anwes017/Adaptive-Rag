from langgraph.store.postgres import PostgresStore
from langgraph.store.memory import InMemoryStore

from src.config import POSTGRES_URI


_store_context = None
_store = None


def get_postgres_store():
    global _store_context, _store

    if _store is not None:
        return _store

    try:
        _store_context = PostgresStore.from_conn_string(POSTGRES_URI)
        _store = _store_context.__enter__()
        _store.setup()
        return _store
    except Exception as exc:
        print(f"Postgres store unavailable, using in-memory store: {exc.__class__.__name__}")
        _store = InMemoryStore()
        return _store
