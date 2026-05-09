from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.llm import llm


class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the user question into a short web search query.

Rules:
- 6 to 14 words
- keywords only
- if the question implies recency, add a constraint like last 30 days
- do not answer
- return JSON only with key query
""",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state):
    result = rewrite_chain.invoke({"question": state["question"]})

    return {"web_query": result.query}
