from src.tools.wikipedia_tool import wikipedia_tool


def wikipedia_node(state):
    result = wikipedia_tool(state["question"])

    return {
        "tool_output": result,
        "verdict": "TOOL",
        "reason": "Wikipedia tool was selected.",
    }
