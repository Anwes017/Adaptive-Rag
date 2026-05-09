from langgraph.graph import END, START, StateGraph

from src.nodes.evaluate import eval_each_doc_node, route_after_eval
from src.nodes.refine import refine_node
from src.nodes.retrieve import retrieve_node
from src.nodes.rewrite_query import rewrite_query_node
from src.nodes.web_search import web_search_node
from src.state import ChatState


def build_crag_subgraph():
    graph = StateGraph(ChatState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("eval_each_doc", eval_each_doc_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("refine", refine_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "eval_each_doc")

    graph.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "refine")
    graph.add_edge("refine", END)

    return graph.compile()
