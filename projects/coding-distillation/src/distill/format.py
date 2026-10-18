"""Shared chat template for training and evaluation, plus code extraction."""
import re

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Briefly explain your approach, "
    "then give the complete function in a single ```python code block."
)
USER_TEMPLATE = "Complete the following Python function:\n```python\n{prompt}```"

_CLOSED = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_OPEN_TAIL = re.compile(r"```(?:python|py)?[ \t]*\n((?:(?!```).)*)$", re.S)


def _with_newline(s: str) -> str:
    return s if s.endswith("\n") else s + "\n"


def to_messages(prompt, solution=None, explanation=None):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(prompt=_with_newline(prompt))},
    ]
    if solution is not None:
        body = f"{explanation.strip()}\n" if explanation else ""
        msgs.append({
            "role": "assistant",
            "content": f"{body}```python\n{solution.rstrip()}\n```",
        })
    return msgs


def extract_code(text, prompt, entry_point):
    blocks = _CLOSED.findall(text)
    # An odd number of fences means generation stopped inside a block.
    if text.count("```") % 2 and (tail := _OPEN_TAIL.search(text)):
        blocks.append(tail.group(1))
    defining = [b for b in blocks if f"def {entry_point}" in b]
    if defining:
        code = defining[-1]
    elif blocks:
        code = blocks[0]
    else:
        code = text
    code = _with_newline(code)
    # The prompt supplies imports and helper functions; a full definition from
    # the model follows it and overrides the docstring-only stub.
    if f"def {entry_point}" in code:
        return _with_newline(prompt) + "\n" + code
    return _with_newline(prompt) + code
