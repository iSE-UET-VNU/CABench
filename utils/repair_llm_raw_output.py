import re

def extract_state_value_from_output(content: str) -> str:
    """
    For openai models, they will always return state number. But for open llm models, the instruction result maybe a
    long text contain target number, so here add a extraction to improve success rate.

    Args:
        content (str): llm's output from `Role._think`
    """
    content = content.strip()  # deal the output cases like " 0", "0\n" and so on.
    pattern = (
        r"(?<!-)[0-9]"  # TODO find the number using a more proper method not just extract from content using pattern
    )
    matches = re.findall(pattern, content, re.DOTALL)
    matches = list(set(matches))
    state = matches[0] if len(matches) > 0 else "-1"
    return state
