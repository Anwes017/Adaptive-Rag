import wikipedia


def wikipedia_tool(query: str):
    try:
        return wikipedia.summary(
            query,
            sentences=5,
        )
    except Exception as e:
        return f"Wikipedia error: {str(e)}"
