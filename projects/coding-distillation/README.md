# Coding distillation: Opus 5.5 → Qwen2.5-Coder-0.5B-Instruct

Personal learning project (see `../../coding-distillation-design.md`). Teacher
data was written by Opus 5.5 in a Claude Code session; it and the resulting
adapter are not for publication.

## Setup

```bash
uv sync --python 3.12
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run pytest                      # unit tests
uv run python scripts/01_setup.py  # download model + HumanEval, smoke generate
```

## Pipeline

| Step | Command | Output |
|---|---|---|
| 02 generate | in-session Opus subagents (see plan Task 7) | `data/raw/batch_*.jsonl` |
| 03 validate | `uv run python scripts/03_validate.py` | `data/train.jsonl`, `val.jsonl`, `rejected.jsonl` |
| baseline | `uv run python scripts/05_eval.py` | `outputs/eval_base.jsonl` |
| 04 train | `uv run python scripts/04_train.py` | `outputs/lora-adapter/` |
| fine-tuned eval | `uv run python scripts/05_eval.py --adapter outputs/lora-adapter` | `outputs/eval_ft.jsonl` |
| report | `uv run python scripts/05_eval.py --compare` | `results.md` |

Smoke flags: `05_eval.py --limit 5`, `04_train.py --max-steps 5`. Eval resumes
from its output file if interrupted. `--device cpu` forces CPU.

## Results

| model | passed | pass@1 |
|---|---|---|
| base | 86/164 | 52.4% |
| fine-tuned | 61/164 | 37.2% |

Greedy decoding, max 512 new tokens. 14 problems fixed by fine-tuning, 39
broken; full lists and samples in `results.md`.

400 teacher records → 397 after validation (3 reworded duplicates) → 357 train
/ 40 val. LoRA (r=16, all linear layers), 3 epochs, 69 steps, 6.4 min on MPS.
`eval_loss` 1.040 → 0.890 → 0.872 → 0.873: imitation of the teacher improved
through epoch 2, then flattened while train loss kept falling (0.91 → 0.67).

**Distillation regressed HumanEval by 15 points.** The student learned the
teacher's *style* very well — answers shrank from ~1,800 to ~330 characters,
and truncation failures (SyntaxError at the 512-token cap) fell from 12 to 2,
which is where most of the 14 fixes came from. But wrong answers rose from 48
to 82. The teacher's explanations are one confident sentence stating the key
insight; the student reproduces that *form* without the capability behind it,
so it now opens with a confabulated "insight" and then faithfully implements
it ("a product is one if any number is zero", "ties are broken by
lexicographical order" — neither is in the spec). The base model's long,
step-by-step explanations were acting as chain-of-thought before the code;
training it to be terse removed that thinking room. Lower `eval_loss` on
teacher data did not mean higher correctness on a different benchmark.

Things to try next (not done here, out of scope): teacher explanations that
reason step by step *before* the code instead of a one-line summary; 1 epoch
or a lower learning rate; more and more varied data.
