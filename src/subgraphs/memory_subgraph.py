from langgraph.graph import END, START, StateGraph

from src.nodes.extract_question import extract_question_node
from src.nodes.remember import remember_node
from src.state import ChatState


def build_memory_subgraph():
    graph = StateGraph(ChatState)

    graph.add_node("extract_question", extract_question_node)
    graph.add_node("remember", remember_node)

    graph.add_edge(START, "extract_question")
    graph.add_edge("extract_question", "remember")
    graph.add_edge("remember", END)

    return graph.compile()
