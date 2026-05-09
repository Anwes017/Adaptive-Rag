from src.tools.arxiv_tool import arxiv_tool


def arxiv_node(state):
    result = arxiv_tool(state["question"])

    return {
        "tool_output": result,
        "verdict": "TOOL",
        "reason": "Arxiv tool was selected.",
    }
