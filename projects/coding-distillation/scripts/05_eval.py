"""HumanEval pass@1 for the base model or base + LoRA adapter; --compare writes results.md."""
import argparse
import json
from pathlib import Path

from distill.contamination import load_humaneval
from distill.evaluate import build_program, compare, load_done, pass_at_1
from distill.format import extract_code, to_messages
from distill.modeling import generate, load_model, pick_device
from distill.sandbox import run

OUT = Path("outputs")


def evaluate(adapter, out_path, limit, device):
    problems = load_humaneval()
    if limit:
        problems = problems[:limit]
    done = load_done(out_path)
    todo = [p for p in problems if p["task_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {out_path}")
    if todo:
        model, tokenizer = load_model(pick_device(device), adapter)
    with open(out_path, "a") as f:
        for i, p in enumerate(todo, 1):
            completion = generate(model, tokenizer, to_messages(p["prompt"]))
            code = extract_code(completion, p["prompt"], p["entry_point"])
            result = run(build_program(code, p["test"], p["entry_point"]))
            rec = {"task_id": p["task_id"], "completion": completion, "code": code,
                   "passed": result.passed, "error": result.error}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[p["task_id"]] = rec
            print(f"[{i}/{len(todo)}] {p['task_id']}: {'PASS' if result.passed else 'FAIL'}")
    scored = [done[p["task_id"]] for p in problems]
    print(f"pass@1 = {pass_at_1(scored):.3f} ({sum(r['passed'] for r in scored)}/{len(scored)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="path to LoRA adapter; omit for base model")
    ap.add_argument("--out", help="output jsonl (default outputs/eval_base.jsonl or eval_ft.jsonl)")
    ap.add_argument("--limit", type=int, help="only the first N problems (smoke test)")
    ap.add_argument("--device", help="mps or cpu (default: auto)")
    ap.add_argument("--compare", action="store_true", help="write results.md from eval_base/eval_ft")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.compare:
        md = compare(load_done(OUT / "eval_base.jsonl"), load_done(OUT / "eval_ft.jsonl"))
        Path("results.md").write_text(md)
        print(md)
        return
    default = OUT / ("eval_ft.jsonl" if args.adapter else "eval_base.jsonl")
    evaluate(args.adapter, Path(args.out) if args.out else default, args.limit, args.device)


if __name__ == "__main__":
    main()
