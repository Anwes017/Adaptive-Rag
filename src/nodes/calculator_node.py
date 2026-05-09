from src.tools.calculator_tool import calculator_tool


def calculator_node(state):
    result = calculator_tool(state["question"])

    return {
        "tool_output": result,
        "verdict": "TOOL",
        "reason": "Calculator tool was selected.",
    }
