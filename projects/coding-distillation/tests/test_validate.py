import json

from distill.validate import split, validate_records

HE = dict(he_prompts=[], he_entry_points={"has_close_elements"})


def rec(**over):
    r = {
        "id": "gen-0001",
        "topic": "math",
        "difficulty": "easy",
        "prompt": 'def double_it(x: int) -> int:\n    """Return 2*x.\n    >>> double_it(2)\n    4\n    """\n',
        "solution": "def double_it(x: int) -> int:\n    return 2 * x\n",
        "explanation": "Multiply by two.",
        "tests": "assert double_it(2) == 4\nassert double_it(0) == 0\nassert double_it(-3) == -6",
        "entry_point": "double_it",
    }
    r.update(over)
    return json.dumps(r)


def reasons(rejected):
    return [r["reason"] for r in rejected]


def test_accepts_good_record_and_skips_blank_lines():
    ok, bad = validate_records([rec(), "", "   "], **HE)
    assert len(ok) == 1 and bad == []


def test_rejects_parse_error():
    _, bad = validate_records(["{not json"], **HE)
    assert reasons(bad) == ["parse_error"]


def test_rejects_missing_field():
    r = json.loads(rec())
    del r["tests"]
    _, bad = validate_records([json.dumps(r)], **HE)
    assert reasons(bad) == ["schema"]


def test_rejects_too_few_asserts():
    _, bad = validate_records([rec(tests="assert double_it(2) == 4")], **HE)
    assert reasons(bad) == ["schema"]


def test_rejects_failing_solution():
    _, bad = validate_records([rec(solution="def double_it(x: int) -> int:\n    return x\n")], **HE)
    assert reasons(bad) == ["tests_failed"]


def test_rejects_contaminated_name():
    r = rec(
        prompt='def has_close_elements(x: int) -> int:\n    """Return 2*x."""\n',
        solution="def has_close_elements(x: int) -> int:\n    return 2 * x\n",
        tests="assert has_close_elements(1) == 2\nassert has_close_elements(0) == 0\nassert has_close_elements(2) == 4",
        entry_point="has_close_elements",
    )
    _, bad = validate_records([r], **HE)
    assert reasons(bad) == ["contamination_name"]


def test_rejects_duplicate_prompt_ignoring_whitespace():
    dup = rec(id="gen-0002", prompt=json.loads(rec())["prompt"].replace("    ", "  "))
    ok, bad = validate_records([rec(), dup], **HE)
    assert len(ok) == 1 and reasons(bad) == ["duplicate"]


def test_split_is_deterministic_and_disjoint():
    records = [{"id": str(i)} for i in range(20)]
    train, val = split(records)
    assert len(val) == 2 and len(train) == 18
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in val})
    assert split(records) == (train, val)


def test_rejects_reused_entry_point_name():
    # Cross-batch near-duplicates: same task reworded under the same name.
    other = rec(id="gen-0002", prompt='def double_it(x: int) -> int:\n    """Twice the input.\n    >>> double_it(3)\n    6\n    """\n')
    ok, bad = validate_records([rec(), other], **HE)
    assert len(ok) == 1 and reasons(bad) == ["duplicate"]
