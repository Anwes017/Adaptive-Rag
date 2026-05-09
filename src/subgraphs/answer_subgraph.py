from langgraph.graph import END, START, StateGraph

from src.nodes.generate import generate_node
from src.state import ChatState


def build_answer_subgraph():
    graph = StateGraph(ChatState)

    graph.add_node("generate", generate_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)

    return graph.compile()
