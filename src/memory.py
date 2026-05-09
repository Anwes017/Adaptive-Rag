from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import POSTGRES_URI


_checkpointer_context = None
checkpointer = None


async def setup_checkpointer():
    global _checkpointer_context, checkpointer

    if checkpointer is not None:
        return checkpointer

    _checkpointer_context = AsyncPostgresSaver.from_conn_string(POSTGRES_URI)
    checkpointer = await _checkpointer_context.__aenter__()
    await checkpointer.setup()
    return checkpointer


async def close_checkpointer():
    global _checkpointer_context, checkpointer

    if _checkpointer_context is not None:
        await _checkpointer_context.__aexit__(None, None, None)

    _checkpointer_context = None
    checkpointer = None


async def retrieve_all_threads():
    threads = set()

    saver = await setup_checkpointer()
    async for checkpoint in saver.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        threads.add(thread_id)

    return list(threads)


async def delete_thread(thread_id: str):
    saver = await setup_checkpointer()
    await saver.adelete_thread(thread_id)
