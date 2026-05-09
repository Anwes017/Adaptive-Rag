from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    web_query: str
    web_docs: List[Document]

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    route: str
    tool_output: str

    answer: str