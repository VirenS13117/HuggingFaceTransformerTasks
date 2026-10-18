# Learning Log — Coding Distillation Project

A running record, written after each task: what was built, a short quiz, and
the answers. Read a task's quiz first and try it before looking at the answers.

---

## Task 1 — Project scaffold + subprocess sandbox

Commit `6d9e0fe`. Tests: 6/6.

### Summary
- Branch `coding-distillation`; project at `projects/coding-distillation/`.
- `uv` env on Python 3.12: torch 2.14, transformers 5.20, trl 1.16, peft 0.21,
  datasets 5.1; MPS (Apple GPU) available.
- `src/distill/sandbox.py` — `run(code, timeout=5.0) -> Result(passed, error)`.
  Writes code to a temp file, runs `python -I` in a child process. Used by both
  data validation and HumanEval scoring. Three safety rails:
  1. **Timeout** — infinite loops are killed after 5 s → fail.
  2. **stdin = /dev/null** — `input()` fails instantly instead of hanging.
  3. **Sentinel** — appends `print("__DISTILL_DONE__")`; pass requires exit
     code 0 *and* the sentinel printed (proof the end was reached).

### Quiz
1. Why run code in a separate subprocess instead of `exec()` inside the eval script?
2. If the model's code calls `sys.exit(0)` before the tests run, what would a
   naive "exit code 0 = pass" rule report, and how does the sentinel prevent it?
3. Without `stdin=DEVNULL`, what happens to a solution calling `input()`? Why
   does that matter more for eval than for teacher-data validation?
4. The sandbox is "not a security boundary." What could model code still do,
   and why is that acceptable here?

### Answers
**1.** *Your answer:* keep the processes independent and avoid changing the
system. *Feedback:* "independent" is right, but a subprocess does **not**
protect your system — the child runs as you, with your files and network. What
it protects is **the evaluator itself**:

| Problem | `exec()` in-process | Subprocess |
|---|---|---|
| Infinite loop | Evaluator hangs; can't reliably interrupt | `timeout=5` kills the child |
| `sys.exit()` / crash | Kills the **whole eval run** | Only the child dies → recorded fail |
| Leftover state | Problem 37 redefines `sorted`, breaks problem 38 | Fresh interpreter every time |
| Memory leaks / segfaults | Accumulate or crash the evaluator | Vanish with the child |

`-I` (isolated mode) also ignores `PYTHON*` env vars and user site-packages, and
stops imports from the current directory.

**2.** The program would exit 0 before `check(candidate)` ran, so a naive rule
counts an untested solution as a pass — a model could get high pass@1 by
emitting `exit()`. The sentinel is printed *after* the tests, so it only
appears if execution got through them.

**3.** The child inherits the parent's stdin; `input()` blocks, burning the
full timeout per problem (or hanging interactively). With `DEVNULL`, `input()`
raises `EOFError` immediately. It matters more for eval because the student's
output is uncontrolled — a weak model may write `n = int(input())` in
competitive-programming style — whereas teacher data is written under explicit
"no I/O" rules and validated.

**4.** The child runs as your user: it could delete or read files, make network
calls, or spawn processes. The temp directory only changes where relative paths
point. It's acceptable because the code comes from Opus and from a 0.5B model
solving 164 known puzzles — not adversarial input. Untrusted code would need a
real sandbox (container, no network, read-only filesystem, resource limits).

---

## Task 2 — Chat template + code extraction

Commit `50220bd`. Tests: 13/13 (cumulative).

### Summary
`src/distill/format.py`:
- **`to_messages(prompt, solution=None, explanation=None)`** — one fixed system
  prompt and user template ("Complete the following Python function: …"), used
  for **both training and evaluation**.
  - No solution → `[system, user]`, for eval (the model writes the reply).
  - With solution → adds an assistant turn: explanation, then **exactly one**
    ```` ```python ```` block. This is the output structure being taught.
- **`extract_code(text, prompt, entry_point)`** — turns a messy reply into a
  runnable program:
  - picks the **last** block containing `def <entry_point>` (skips usage
    examples and earlier drafts);
  - handles **truncated** replies (odd number of ``` fences → use the open block);
  - **always prepends the HumanEval prompt**, which supplies imports and helper
    functions; the model's full `def` follows and overrides the stub;
  - body-only replies become prompt + body (still a valid function);
  - no fences at all → falls back to the raw text.

### Quiz
1. Why must training and evaluation use the *exact same* system prompt and user
   template? What could go wrong if eval used a different wording?
2. A reply contains a usage example `print(total([1]))`, a first attempt, and a
   corrected version. Why pick the *last* block containing `def total`?
3. HumanEval's `decode_cyclic` prompt defines helper `encode_cyclic` above the
   target. The model returns only `def decode_cyclic(...)`. What error occurs if
   we run just the model's block, how does prepending the prompt fix it, and why
   is defining `decode_cyclic` twice safe?
4. Why could sloppy extraction make a 0.5B model's score wrong in either
   direction? Which is more dangerous for a before/after comparison?

### Answers
**1.** Fine-tuning teaches a *conditional* behavior: "given this context,
respond in this format." A different wording at eval is a distribution shift,
and the fine-tuned model might not trigger its learned structure. It also
confounds the comparison: base and fine-tuned must see identical inputs so any
score difference comes from the weights alone, not the prompt.

**2.** Small models often show an example call first or iterate (draft, then
"actually, fix…"). The last *defining* block is the final answer. The first
block might be a call with no definition (→ `NameError`) or a buggy draft.
Requiring `def total` filters out usage snippets.

**3.** The canonical `decode_cyclic` calls `encode_cyclic`, so running only the
model's block raises `NameError: name 'encode_cyclic' is not defined`.
Prepending the prompt defines the helper and any imports first. Defining a
function twice is safe because `def` is just an assignment at runtime: the
prompt's docstring-only stub binds the name first (a valid body that returns
`None`), then the model's definition rebinds it before `check()` runs.

**4.** *Too low:* correct solutions marked failing because of extraction bugs
(wrong block, missing imports). *Too high:* rarer, e.g. counting code that
never really ran — the sentinel guards against that. The most dangerous case is
**asymmetric** bias: if the fine-tuned model formats more cleanly and a strict
extractor only handles clean formatting, part of the "gain" is really just
extraction-friendliness. A lenient, robust extractor measures coding ability
rather than formatting quirks. Some formatting gain is legitimately part of the
"structural" fine-tune, but we want to see it for what it is.

---

## Task 3 — HumanEval contamination check

Commit: see `git log` ("HumanEval contamination check"). Tests: 17/17 (cumulative).

### Summary
`src/distill/contamination.py`:
- `load_humaneval()` — loads the 164 HumanEval problems from the HF Hub
  (`openai/openai_humaneval`, split `test`).
- `ngrams(text, n=5)` — lowercases, keeps alphanumeric tokens, returns the set
  of consecutive 5-word tuples.
- `jaccard(a, b)` — |A ∩ B| / |A ∪ B| (0.0 when both are empty).
- `check(prompt, entry_point, he_prompts, he_entry_points, threshold=0.3)` —
  returns `"contamination_name"` if the function name equals any HumanEval
  entry point, `"contamination_ngram"` if 5-gram Jaccard ≥ 0.3 against any
  HumanEval prompt, otherwise `None`.
- A renamed copy of `has_close_elements` scores Jaccard **0.49**, so it is
  caught; an unrelated run-length-encoding prompt scores ~0.

### Quiz
1. What is **benchmark contamination**, and why would it make our before/after
   HumanEval comparison meaningless even if training "worked"?
2. Why use **5-grams of words** rather than single words (1-grams) for the
   similarity check? What goes wrong with 1-grams?
3. Why a **set-based** Jaccard rather than counting raw shared words?
4. Name a kind of contamination this check would **miss**.

### Answers
**1.** Contamination means benchmark problems (or near-copies) leak into
training data. The model can then score well by *recalling* answers rather than
solving new problems, so the "gain" measures memorization, not improved
skill. Our teacher (Opus) has certainly seen HumanEval, so it might reproduce
those problems when asked for "coding problems" — that's why subagents are
told to avoid them *and* we check mechanically.

**2.** With 1-grams, almost every Python docstring shares words like `the`,
`list`, `return`, `int`, `given`, `numbers`, so unrelated problems look
similar → many false rejections. A 5-word sequence like "are any two numbers
closer" is specific; sharing many of them strongly suggests copied text.
Longer n is stricter (fewer false positives) but misses light paraphrases;
5 is a common compromise (similar to what n-gram decontamination reports for
GPT-3/4 use, which typically use 8–13 *token* n-grams over much longer docs).

**3.** Sets make the score length-normalized and bounded in [0, 1]: a long
prompt can't look "similar" just by having many words, and repeated phrases
don't count multiple times. That lets one threshold (0.3) work across prompts
of very different lengths.

**4.** Semantic paraphrases: the same task described in completely different
words (e.g. "return True if some pair differs by less than t" vs. HumanEval's
wording) shares few 5-grams and passes. Also problems whose *solution* matches
a HumanEval solution but whose prompt is new. Catching those would need
embedding similarity or manual review — out of scope here; the subagent
instructions are the first line of defense.

---

## Task 5 — Teacher-data validation

Commit: "validate teacher data by execution, contamination, dedupe". Tests: 25/25 (cumulative).
(Task 4 — model loading — is being finished in parallel while the model downloads.)

### Summary
`src/distill/validate.py` + `scripts/03_validate.py`. Every raw teacher record
goes through a **filter pipeline**; the first failing stage rejects it with a
reason:

```
line → parse_error? → schema? → contamination_name / _ngram? → duplicate? → tests_failed? → accepted
```

- **schema** — all 8 fields are non-empty strings, ≥3 `assert`s, and
  `def <entry_point>` appears in both prompt and solution.
- **contamination** — Task 3's check against all 164 HumanEval prompts.
- **duplicate** — prompts compared after collapsing whitespace + lowercasing.
- **tests_failed** — `solution + tests` run in the Task 1 sandbox.
- `split()` shuffles with a fixed seed (42) and holds out 10% as validation.
- The script writes `data/train.jsonl` / `val.jsonl` as chat `{"messages": [...]}`
  rows (Task 2's `to_messages`), plus `rejected.jsonl` with reasons. `--dry-run`
  only prints stats — the teacher subagents use it to self-check their batches.

### Quiz
1. Why put the **cheap checks first** (parse, schema, contamination, dedupe)
   and the sandbox execution **last**?
2. Why is it important that every training solution **actually passes tests**?
   What does training on a subtly wrong solution teach the student?
3. Why hold out a **validation split** if we're going to evaluate on HumanEval
   anyway? What does `eval_loss` tell us that HumanEval doesn't?
4. Why is the shuffle **seeded**?

### Answers
**1.** Running code costs a process launch (~30–50 ms) and up to a 5 s timeout;
the others are microseconds. Rejecting early saves that work and gives the most
specific reason (a contaminated record is reported as contaminated, not "also
passes tests"). The dedupe key is only added to `seen` *after* a record is
accepted, so a failing first copy doesn't block a later correct copy.

**2.** In SFT the model imitates its targets token by token — it has no idea
which ones are wrong. A buggy solution teaches the bug as "correct style";
distillation copies the teacher's mistakes along with its strengths.
Execution-filtering (keeping only verified outputs) is a standard trick:
it's the core of "rejection sampling" / self-instruct pipelines for code.

**3.** HumanEval measures final capability, but only after training ends and
only on a different distribution. `eval_loss` on held-out *teacher* data,
computed each epoch, tells us whether the model is learning generalizable
patterns (both losses drop) or **memorizing** the training set (train loss keeps
dropping, val loss rises → overfitting). With ~350 examples and 3 epochs,
that's a real risk worth watching.

**4.** Reproducibility: rerunning `03_validate.py` on the same raw data gives
the identical train/val split, so a change in results comes from what you
changed, not from a different random split.

---
