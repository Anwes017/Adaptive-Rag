def extract_question_node(state):
    latest_message = state["messages"][-1]

    return {
        "question": latest_message.content,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "web_query": "",
        "web_docs": [],
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "route": "",
        "tool_output": "",
        "answer": "",
    }
