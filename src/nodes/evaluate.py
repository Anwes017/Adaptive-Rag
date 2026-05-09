from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.config import LOWER_TH, UPPER_TH
from src.llm import llm


class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval evaluator for CRAG.

You will be given ONE retrieved chunk and a question.
Return a relevance score in [0.0, 1.0].

- 1.0: chunk alone is sufficient to answer fully/mostly.
- 0.0: chunk is irrelevant.

Be conservative with high scores.
Also return a short reason.

Return JSON only.
""",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state):
    question = state["question"]

    scores = []
    good_docs = []

    for doc in state["docs"]:
        result = doc_eval_chain.invoke(
            {
                "question": question,
                "chunk": doc.page_content,
            }
        )

        scores.append(result.score)

        if result.score > LOWER_TH:
            good_docs.append(doc)

    if scores and any(score > UPPER_TH for score in scores):
        return {
            "good_docs": good_docs,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    if scores and all(score < LOWER_TH for score in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}.",
        }

    return {
        "good_docs": good_docs,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }


def route_after_eval(state):
    if state["verdict"] == "CORRECT":
        return "refine"

    return "rewrite_query"
