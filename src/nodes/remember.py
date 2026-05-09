import uuid
from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from src.llm import memory_llm


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")


class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


memory_extractor = memory_llm.with_structured_output(MemoryDecision)


MEMORY_PROMPT = """
You maintain long-term user memory.

CURRENT USER MEMORY:
{user_details_content}

Extract only stable long-term facts:
- name
- preferences
- ongoing projects
- tech stack
- goals

Do not store temporary facts.
Return structured output only.
"""


def remember_node(state, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")
    ns = ("user", user_id, "details")

    items = store.search(ns)

    existing = "\n".join(item.value.get("data", "") for item in items) if items else "(empty)"
    last_text = state["messages"][-1].content

    decision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {
                "role": "user",
                "content": last_text,
            },
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(
                    ns,
                    str(uuid.uuid4()),
                    {"data": mem.text.strip()},
                )

    return {}
