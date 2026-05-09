from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from src.config import TAVILY_API_KEY

tavily = None

if TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key":
    tavily = TavilySearchResults(max_results=5)


def web_search_node(state):
    if tavily is None:
        return {"web_docs": []}

    query = state.get("web_query") or state["question"]

    try:
        results = tavily.invoke({"query": query})
    except Exception:
        return {"web_docs": []}

    web_docs = []

    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "") or result.get("snippet", "")

        text = f"""TITLE: {title}
URL: {url}
CONTENT:
{content}"""

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "url": url,
                },
            )
        )

    return {"web_docs": web_docs}
