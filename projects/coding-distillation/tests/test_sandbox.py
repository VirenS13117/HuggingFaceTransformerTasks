from distill.sandbox import run


def test_passing_code():
    assert run("assert 1 + 1 == 2").passed


def test_failing_assert_reports_error():
    r = run("assert 1 + 1 == 3")
    assert not r.passed
    assert "AssertionError" in r.error


def test_syntax_error_fails():
    assert not run("def f(:\n  pass").passed


def test_timeout_fails():
    r = run("while True:\n    pass", timeout=1.0)
    assert not r.passed
    assert r.error == "timeout"


def test_early_exit_is_not_a_pass():
    # Review Focus 1: exit(0) before tests must not count as passing.
    assert not run("import sys\nsys.exit(0)\nassert False").passed


def test_input_fails_fast():
    # Review Focus 2: stdin is /dev/null, so input() raises EOFError.
    r = run("x = input()", timeout=5.0)
    assert not r.passed
    assert "EOFError" in r.error
