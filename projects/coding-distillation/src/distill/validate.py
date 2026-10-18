"""Filter raw teacher records: schema, contamination, dedupe, execution."""
import json
import random
import re

from distill.contamination import check
from distill.sandbox import run

REQUIRED = ("id", "topic", "difficulty", "prompt", "solution", "explanation", "tests", "entry_point")


def _schema_error(r):
    missing = [k for k in REQUIRED if not isinstance(r.get(k), str) or not r[k].strip()]
    if missing:
        return f"missing/empty: {missing}"
    if r["tests"].count("assert") < 3:
        return "fewer than 3 asserts"
    defn = f"def {r['entry_point']}"
    if defn not in r["prompt"] or defn not in r["solution"]:
        return f"{defn} not in prompt and solution"
    return None


def _normalize(prompt):
    return re.sub(r"\s+", " ", prompt).strip().lower()


def validate_records(lines, he_prompts, he_entry_points):
    accepted, rejected, seen = [], [], set()

    def reject(r, reason, detail=""):
        rejected.append({"id": r.get("id") if isinstance(r, dict) else None, "reason": reason, "detail": detail})

    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            reject(None, "parse_error", str(e))
            continue
        if not isinstance(r, dict):
            reject(None, "parse_error", "not an object")
            continue
        if err := _schema_error(r):
            reject(r, "schema", err)
            continue
        if reason := check(r["prompt"], r["entry_point"], he_prompts, he_entry_points):
            reject(r, reason)
            continue
        key = _normalize(r["prompt"])
        if key in seen:
            reject(r, "duplicate")
            continue
        result = run(r["solution"] + "\n\n" + r["tests"])
        if not result.passed:
            reject(r, "tests_failed", result.error)
            continue
        seen.add(key)
        accepted.append(r)
    return accepted, rejected


def split(records, val_frac=0.1, seed=42):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]
