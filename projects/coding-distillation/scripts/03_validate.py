"""raw/*.jsonl -> data/train.jsonl, data/val.jsonl, data/rejected.jsonl"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from distill.contamination import load_humaneval
from distill.format import to_messages
from distill.validate import split, validate_records

DATA = Path("data")


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="data/raw/batch_*.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print stats only, write nothing")
    args = ap.parse_args()

    files = sorted(glob.glob(args.raw_glob))
    lines = [line for path in files for line in Path(path).read_text().splitlines()]
    he = load_humaneval()
    accepted, rejected = validate_records(
        lines, [p["prompt"] for p in he], {p["entry_point"] for p in he}
    )

    print(f"files: {len(files)}  accepted: {len(accepted)}  rejected: {len(rejected)}")
    for reason, n in Counter(r["reason"] for r in rejected).most_common():
        print(f"  {reason}: {n}")
    for r in rejected[:10]:
        print(f"  - {r['id']}: {r['reason']} {r['detail'][:200]}")
    if args.dry_run:
        return

    train, val = split(accepted)
    to_row = lambda r: {"messages": to_messages(r["prompt"], r["solution"], r["explanation"])}
    write_jsonl(DATA / "train.jsonl", map(to_row, train))
    write_jsonl(DATA / "val.jsonl", map(to_row, val))
    write_jsonl(DATA / "rejected.jsonl", rejected)
    print(f"train: {len(train)}  val: {len(val)}")


if __name__ == "__main__":
    main()
