from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.llm import llm
from src.utils.text_utils import decompose_to_sentences


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict relevance filter.

Return keep=true only if the sentence directly helps answer the question.

Use ONLY the sentence.
Return JSON only.
""",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


def refine_node(state):
    question = state["question"]

    if state["verdict"] == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state["verdict"] == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(doc.page_content for doc in docs_to_use).strip()
    strips = decompose_to_sentences(context)

    kept = []

    for sentence in strips:
        result = filter_chain.invoke(
            {
                "question": question,
                "sentence": sentence,
            }
        )

        if result.keep:
            kept.append(sentence)

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }
