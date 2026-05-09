from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.llm import llm


class RouteDecision(BaseModel):
    route: str


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a tool router.

Available routes:

calculator:
- math
- arithmetic
- equations

wikipedia:
- general knowledge
- history
- definitions

arxiv:
- AI research papers
- ML papers
- LLM research
- transformers papers

crag:
- uploaded PDFs
- document QA
- project docs
- anything else

Return JSON only:
{{"route": "..."}}
""",
        ),
        ("human", "Question: {question}"),
    ]
)

router_chain = router_prompt | llm.with_structured_output(RouteDecision)


def tool_router_node(state):
    result = router_chain.invoke({"question": state["question"]})

    return {"route": result.route}


def route_after_tool_router(state):
    route = state.get("route", "crag")

    if route not in {"calculator", "wikipedia", "arxiv", "crag"}:
        return "crag"

    return route
