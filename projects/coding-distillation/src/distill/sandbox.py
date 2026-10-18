"""Run untrusted-ish Python in a subprocess. Not a security boundary."""
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SENTINEL = "__DISTILL_DONE__"


@dataclass(frozen=True)
class Result:
    passed: bool
    error: str


def run(code: str, timeout: float = 5.0) -> Result:
    # The sentinel is printed only if execution reaches the end of the program,
    # so an early exit(0) is not mistaken for passing tests.
    program = f"{code}\n\nprint({SENTINEL!r})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(False, "timeout")
    passed = proc.returncode == 0 and proc.stdout.rstrip().endswith(SENTINEL)
    return Result(passed, "" if passed else proc.stderr[-2000:])
