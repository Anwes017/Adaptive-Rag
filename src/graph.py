from langgraph.graph import END, START, StateGraph

from src.long_term_memory import get_postgres_store
from src.nodes.arxiv_node import arxiv_node
from src.nodes.calculator_node import calculator_node
from src.nodes.tool_router import route_after_tool_router, tool_router_node
from src.nodes.wikipedia_node import wikipedia_node
from src.state import ChatState
from src.subgraphs.answer_subgraph import build_answer_subgraph
from src.subgraphs.crag_subgraph import build_crag_subgraph
from src.subgraphs.memory_subgraph import build_memory_subgraph


memory_subgraph = build_memory_subgraph()
crag_subgraph = build_crag_subgraph()
answer_subgraph = build_answer_subgraph()


def _build_graph_with_answer_path(checkpointer, include_crag_tools: bool):
    graph = StateGraph(ChatState)

    graph.add_node("memory_subgraph", memory_subgraph)
    graph.add_node("answer_subgraph", answer_subgraph)

    graph.add_edge(START, "memory_subgraph")

    if include_crag_tools:
        graph.add_node("crag_subgraph", crag_subgraph)
        graph.add_node("tool_router", tool_router_node)
        graph.add_node("calculator", calculator_node)
        graph.add_node("wikipedia", wikipedia_node)
        graph.add_node("arxiv", arxiv_node)

        graph.add_edge("memory_subgraph", "tool_router")

        graph.add_conditional_edges(
            "tool_router",
            route_after_tool_router,
            {
                "calculator": "calculator",
                "wikipedia": "wikipedia",
                "arxiv": "arxiv",
                "crag": "crag_subgraph",
            },
        )

        graph.add_edge("calculator", "answer_subgraph")
        graph.add_edge("wikipedia", "answer_subgraph")
        graph.add_edge("arxiv", "answer_subgraph")
        graph.add_edge("crag_subgraph", "answer_subgraph")
    else:
        graph.add_edge("memory_subgraph", "answer_subgraph")

    graph.add_edge("answer_subgraph", END)

    return graph.compile(
        checkpointer=checkpointer,
        store=get_postgres_store(),
    )


def build_chatbot(checkpointer):
    return _build_graph_with_answer_path(checkpointer, include_crag_tools=True)


def build_plain_chatbot(checkpointer):
    return _build_graph_with_answer_path(checkpointer, include_crag_tools=False)
