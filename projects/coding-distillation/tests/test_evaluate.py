import json

from distill.evaluate import build_program, compare, ensure_trailing_newline, load_done, pass_at_1
from distill.sandbox import run

TEST = "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(0) == 0\n"


def test_build_program_pass_and_fail():
    good = "def double_it(x):\n    return 2 * x\n"
    bad = "def double_it(x):\n    return x\n"
    assert run(build_program(good, TEST, "double_it")).passed
    assert not run(build_program(bad, TEST, "double_it")).passed


def test_load_done_missing_file(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == {}


def test_load_done_skips_truncated_last_line(tmp_path):
    # Review Focus 5: interrupted write leaves a partial line.
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"task_id": "HumanEval/0", "passed": True}) + "\n" + '{"task_id": "HumanEval/1", "pas')
    assert list(load_done(path)) == ["HumanEval/0"]


def test_pass_at_1():
    assert pass_at_1([]) == 0.0
    assert pass_at_1([{"passed": True}, {"passed": False}, {"passed": True}, {"passed": False}]) == 0.5


def test_compare_lists_fixed_and_broken():
    def r(tid, passed):
        return {"task_id": tid, "passed": passed, "code": f"# {tid}", "completion": "", "error": ""}

    base = {"A": r("A", False), "B": r("B", True), "C": r("C", True)}
    ft = {"A": r("A", True), "B": r("B", False), "C": r("C", True)}
    md = compare(base, ft)
    assert "| base | 2/3 | 66.7% |" in md
    assert "| fine-tuned | 2/3 | 66.7% |" in md
    assert "Fixed by fine-tuning (1): A" in md
    assert "Broken by fine-tuning (1): B" in md


def test_resume_append_after_truncated_line_is_readable(tmp_path):
    # A re-run record must not be glued onto the torn fragment.
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"task_id": "HumanEval/0", "passed": True}) + "\n" + '{"task_id": "HumanEval/1", "pas')
    ensure_trailing_newline(path)
    with open(path, "a") as f:
        f.write(json.dumps({"task_id": "HumanEval/1", "passed": False}) + "\n")
    assert list(load_done(path)) == ["HumanEval/0", "HumanEval/1"]
