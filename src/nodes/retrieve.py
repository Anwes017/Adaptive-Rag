from src.vector.store import get_retriever


def retrieve_node(state):
    try:
        retriever = get_retriever()
    except RuntimeError:
        return {"docs": []}

    return {"docs": retriever.invoke(state["question"])}
