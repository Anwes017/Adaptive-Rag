from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from src.llm import llm


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant with long-term memory.

User memory:
{user_memory}

Answer directly and naturally.
Use memory only when helpful.
Do not mention CRAG, verdicts, source labels, or retrieval.
""",
        ),
        (
            "human",
            """
Question:
{question}
""",
        ),
    ]
)


crag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant with long-term memory.

User memory:
{user_memory}

Answer using CRAG context if available.
Use tool output if available.
If both are empty or insufficient, say:
I don't know based on the available information.

At the end, show:
CRAG Verdict: ...
Source Used: Internal Docs / Web Search / Tool / Both

Keep the response concise and useful.
""",
        ),
        (
            "human",
            """
Question:
{question}

CRAG Verdict:
{verdict}

Context:
{context}

Tool Output:
{tool_output}
""",
        ),
    ]
)


def generate_node(state, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")
    ns = ("user", user_id, "details")

    items = store.search(ns)

    user_memory = "\n".join(item.value.get("data", "") for item in items) if items else "(empty)"

    route = state.get("route", "chat")
    prompt = crag_prompt if route == "crag" else chat_prompt
    prompt_input = {
        "question": state["question"],
        "verdict": state.get("verdict", ""),
        "context": state.get("refined_context", ""),
        "tool_output": state.get("tool_output", ""),
        "user_memory": user_memory,
    }

    response = (prompt | llm).invoke(prompt_input)

    return {
        "answer": response.content,
        "messages": [AIMessage(content=response.content)],
        "route": route,
    }
