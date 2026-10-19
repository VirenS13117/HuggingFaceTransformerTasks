"""HumanEval scoring helpers and before/after report."""
import json


def build_program(code, test, entry_point):
    return f"{code}\n\n{test}\n\ncheck({entry_point})\n"


def load_done(path):
    done = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[rec["task_id"]] = rec
    return done


def pass_at_1(records):
    records = list(records)
    return sum(r["passed"] for r in records) / len(records) if records else 0.0


def _row(name, recs):
    n = sum(r["passed"] for r in recs.values())
    return f"| {name} | {n}/{len(recs)} | {100 * pass_at_1(recs.values()):.1f}% |"


def compare(base, ft, n_samples=3):
    common = sorted(base.keys() & ft.keys())
    fixed = [t for t in common if ft[t]["passed"] and not base[t]["passed"]]
    broken = [t for t in common if base[t]["passed"] and not ft[t]["passed"]]
    lines = [
        "# HumanEval results: Qwen2.5-Coder-0.5B-Instruct, base vs. distilled LoRA",
        "",
        "Greedy decoding, max 512 new tokens, pass@1.",
        "",
        "| model | passed | pass@1 |",
        "|---|---|---|",
        _row("base", base),
        _row("fine-tuned", ft),
        "",
        f"Fixed by fine-tuning ({len(fixed)}): {', '.join(fixed) or 'none'}",
        "",
        f"Broken by fine-tuning ({len(broken)}): {', '.join(broken) or 'none'}",
        "",
    ]
    for t in fixed[:n_samples]:
        lines += [
            f"## {t}",
            "",
            "**Base (failed):**",
            "```python", base[t]["code"].rstrip(), "```",
            "",
            "**Fine-tuned (passed):**",
            "```python", ft[t]["code"].rstrip(), "```",
            "",
        ]
    return "\n".join(lines)


def ensure_trailing_newline(path):
    """Terminate a torn last line so appended records start on their own line."""
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with open(path, "a") as f:
            f.write("\n")
