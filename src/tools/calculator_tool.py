import numexpr
import re


def calculator_tool(expression: str):
    try:
        cleaned = re.sub(r"[^0-9+\-*/().% ]", "", expression)
        cleaned = cleaned.strip()

        if not cleaned:
            return "Calculator error: no valid math expression found."

        result = numexpr.evaluate(cleaned)
        return str(result)
    except Exception as e:
        return f"Calculator error: {str(e)}"
