from distill.format import extract_code, to_messages
from distill.sandbox import run

PROMPT = 'from typing import List\n\n\ndef total(xs: List[int]) -> int:\n    """Sum xs.\n    >>> total([1, 2])\n    3\n    """\n'


def test_to_messages_without_solution():
    msgs = to_messages(PROMPT)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert PROMPT in msgs[1]["content"]
    assert "```python" in msgs[1]["content"]


def test_to_messages_with_solution():
    sol = "def total(xs):\n    return sum(xs)"
    msgs = to_messages(PROMPT, solution=sol, explanation="Use sum.")
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "Use sum.\n```python\ndef total(xs):\n    return sum(xs)\n```"


def test_extract_picks_last_block_with_entry_point():
    text = "Example:\n```python\nprint(total([1]))\n```\nSolution:\n```python\ndef total(xs):\n    return 0\n```\nBetter:\n```python\ndef total(xs):\n    return sum(xs)\n```"
    code = extract_code(text, PROMPT, "total")
    assert code.endswith("return sum(xs)\n")


def test_extract_unclosed_fence():
    # Review Focus 3: generation truncated before closing fence.
    text = "Here:\n```python\ndef total(xs):\n    return sum(xs)\n"
    code = extract_code(text, PROMPT, "total")
    assert run(code + "\nassert total([1, 2]) == 3").passed


def test_extract_keeps_prompt_imports_and_overrides_stub():
    # Review Focus 4: model omits `from typing import List`; prompt supplies it.
    text = "```python\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```"
    code = extract_code(text, PROMPT, "total")
    assert code.startswith(PROMPT)
    assert run(code + "\nassert total([1, 2]) == 3").passed


def test_extract_body_only_completion():
    text = "```python\n    return sum(xs)\n```"
    code = extract_code(text, PROMPT, "total")
    assert run(code + "\nassert total([1, 2]) == 3").passed


def test_extract_no_fence_falls_back_to_raw_text():
    text = "def total(xs):\n    return sum(xs)\n"
    code = extract_code(text, PROMPT, "total")
    assert run(code + "\nassert total([1, 2]) == 3").passed
