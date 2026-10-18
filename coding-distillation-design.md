# Coding Distillation: Opus 5.5 → Qwen2.5-Coder-0.5B — Design

**Date:** 2026-10-18
**Status:** Approved design, pending spec review

## Goal

A small, local, educational project that distills Python coding behavior from
Opus 5.5 (teacher) into `Qwen/Qwen2.5-Coder-0.5B-Instruct` (student) via LoRA
supervised fine-tuning, then measures the effect on HumanEval pass@1
(base vs. fine-tuned).

**Success criteria**
- End-to-end pipeline runs locally on an M4 Max (MPS) with one command per step.
- `results.md` reports base and fine-tuned HumanEval pass@1 from the same harness
  and decoding settings, plus a few qualitative examples.
- Every training example has been executed and passes its own tests.
- No training example is a near-duplicate of a HumanEval problem.

**Non-goals:** publishing weights or data, multi-benchmark evaluation, logit-level
(KD-loss) distillation, hyperparameter sweeps, non-Python languages.

**Constraint:** Teacher outputs are for personal learning; per Anthropic's terms,
the resulting model and dataset are not to be published or used as a competing
product.

## Decisions

| Topic | Decision |
|---|---|
| Teacher data source | Generated inside the Claude Code session (subagents), no API key |
| Dataset size | ~400 raw examples → expect ~300–380 after validation |
| Student | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Fine-tuning | LoRA SFT with `trl` `SFTTrainer` + `peft`, on MPS |
| Benchmark | HumanEval (`openai/openai_humaneval`, 164 problems), pass@1, greedy |
| Location | `projects/coding-distillation/`, standalone `uv` project |

## Layout

```
projects/coding-distillation/
  pyproject.toml           # uv project: torch, transformers, datasets, peft, trl, accelerate
  README.md                # how to run each step, what to expect
  src/distill/
    sandbox.py             # run code + tests in a subprocess with timeout
    format.py              # build chat messages; extract ```python blocks
    contamination.py       # similarity check vs HumanEval prompts
  scripts/
    01_setup.py            # download student model + HumanEval into HF cache
    03_validate.py         # raw/*.jsonl -> data/train.jsonl, data/val.jsonl, data/rejected.jsonl
    04_train.py            # LoRA SFT -> outputs/lora-adapter/
    05_eval.py             # HumanEval pass@1 for base and/or base+adapter
  data/
    raw/batch_*.jsonl      # teacher output written in-session (step 02)
    train.jsonl, val.jsonl, rejected.jsonl
  outputs/                 # adapters, eval generations (gitignored)
  results.md               # before/after table + sample comparisons
  tests/                   # pytest for sandbox, format, contamination
```

Step 02 (generation) is not a script: it is performed in-session by subagents
writing `data/raw/batch_NN.jsonl` files.

## Data schema

Raw teacher record (one JSON per line):

```json
{
  "id": "gen-0001",
  "topic": "strings",
  "difficulty": "easy|medium|hard",
  "prompt": "def f(x: list[int]) -> int:\n    \"\"\"Docstring with spec and >>> examples\"\"\"\n",
  "solution": "def f(x: list[int]) -> int:\n    ...full implementation...\n",
  "explanation": "1–3 sentence explanation of the approach",
  "tests": "assert f([1,2]) == 3\nassert ...",
  "entry_point": "f"
}
```

- `prompt` mirrors HumanEval style: imports (if any), signature with type hints,
  docstring containing the spec and doctest-style examples.
- `tests` are plain asserts (≥3, including edge cases), hidden from training.
- Topic diversity: subagents are each assigned distinct topic buckets (strings,
  lists/arrays, math/number theory, dicts/counting, sorting/searching, recursion,
  simple DP, parsing, geometry, simulation) to avoid duplicates.
- Subagents are instructed not to reproduce known HumanEval problems.

Training record (`train.jsonl` / `val.jsonl`), chat format for `SFTTrainer`:

```json
{"messages": [
  {"role": "system", "content": "<fixed system prompt>"},
  {"role": "user", "content": "Complete the following Python function:\n```python\n<prompt>```"},
  {"role": "assistant", "content": "<explanation>\n```python\n<solution>```"}
]}
```

The **same user-message template and system prompt** are used at eval time, so
the fine-tune teaches a consistent output structure: brief explanation followed by
exactly one fenced Python block containing the complete function.

## Components

**`sandbox.py`** — `run(code: str, timeout: float = 5.0) -> Result(passed, error)`.
Writes code to a temp dir, runs `python -I` in a subprocess with timeout, cwd set
to the temp dir. Used by both validation and evaluation. Not a security boundary;
acceptable because inputs are our own generated code and the model's HumanEval
completions run locally.

**`format.py`** — `to_messages(prompt, solution=None, explanation=None)` and
`extract_code(text, prompt, entry_point) -> str`. Extraction takes the last
```python block containing `def <entry_point>`; if none, falls back to the
first block, then to the raw text; if the extracted code lacks the function
definition, prepends `prompt` (handles body-only completions).

**`contamination.py`** — for each generated prompt, compute max similarity vs.
all 164 HumanEval prompts using (a) word 5-gram Jaccard on the docstring and
(b) entry-point name equality. Reject if Jaccard ≥ 0.3 or names match.

**`03_validate.py`** — for each raw record: schema check → run
`solution + tests` in sandbox → contamination check → exact/near dedupe within
the dataset (normalized prompt hash). Passing records shuffled (seed 42) and
split 90/10 into train/val; failures go to `rejected.jsonl` with a reason.
Prints counts per stage.

**`04_train.py`** — loads student in bf16 (fallback fp32 if MPS bf16 is
problematic), LoRA r=16, alpha=32, dropout 0.05 on all linear projection
layers; `SFTTrainer` with completion-only loss (assistant tokens only),
lr 2e-4, 3 epochs, effective batch 16, max length 1024, eval on val each epoch,
seed 42. Saves adapter + training log to `outputs/lora-adapter/`.

**`05_eval.py`** — `--adapter PATH` optional. For each HumanEval problem: build
messages with the shared template, greedy decode (max 512 new tokens), extract
code, run `code + test + check(entry_point)` in sandbox. Writes per-problem
generations to `outputs/eval_{base|ft}.jsonl` and prints pass@1. A
`--compare` mode reads both files and writes `results.md` (score table, problems
fixed/broken by fine-tuning, 2–3 side-by-side samples).

## Data flow

```
01_setup ──► HF cache (model, HumanEval)
02 (in-session subagents) ──► data/raw/batch_*.jsonl
03_validate ──► data/train.jsonl, val.jsonl, rejected.jsonl
05_eval (base) ──► outputs/eval_base.jsonl          # baseline before training
04_train ──► outputs/lora-adapter/
05_eval --adapter ──► outputs/eval_ft.jsonl
05_eval --compare ──► results.md
```

## Error handling

- Sandbox timeouts and exceptions are recorded as failures, never crash the run.
- Malformed raw JSON lines are logged to `rejected.jsonl` with reason `parse_error`.
- If MPS errors on an op, scripts honor `PYTORCH_ENABLE_MPS_FALLBACK=1`
  (set in README instructions); `--device cpu` flag available on train/eval.
- Eval is resumable: skips task_ids already present in the output jsonl.

## Testing

- `pytest` unit tests: sandbox (pass, fail, timeout), `extract_code` (multiple
  blocks, missing block, body-only completion), contamination (known HumanEval
  prompt rejected, unrelated prompt accepted), `to_messages` round-trip.
- Smoke runs: `05_eval.py --limit 5` and `04_train.py --max-steps 5` before full runs.

## Expected outcome

Baseline ≈ 30% pass@1 (our harness may differ from reported numbers). A few
hundred examples may yield a modest gain, no change, or a small regression; the
report states the result as measured.
