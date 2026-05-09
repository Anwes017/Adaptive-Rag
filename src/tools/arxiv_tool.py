import arxiv


def arxiv_tool(query: str):
    try:
        search = arxiv.Search(
            query=query,
            max_results=3,
        )

        papers = []

        for result in search.results():
            text = f"""
TITLE: {result.title}

SUMMARY:
{result.summary}

URL:
{result.entry_id}
"""
            papers.append(text)

        return "\n\n".join(papers)

    except Exception as e:
        return f"Arxiv error: {str(e)}"
