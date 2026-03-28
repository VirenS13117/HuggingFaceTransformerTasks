# Chapter 3: Tokenizers — Summary & Explanation

## Environment

| Package | Version |
|---------|---------|
| PyTorch | 2.11.0 |
| Transformers | 5.3.0 |
| NumPy | 2.4.3 |
| Device | MPS (Apple GPU) |

---

## What is a Tokenizer?

A tokenizer converts raw text into numerical representations (token IDs) that a model can process. Different models use different tokenizers — here we used **BERT's `bert-base-uncased`** tokenizer, which uses **WordPiece** tokenization.

---

## Experiments & Results

### 1. Single Sentence Encoding

```python
tokenizer("Hello, I am single sentence")
```

**Output:**
| Field | Value | Meaning |
|-------|-------|---------|
| `input_ids` | `[101, 7592, 1010, 1045, 2572, 2309, 6251, 102]` | Token IDs — 101 = `[CLS]`, 102 = `[SEP]` |
| `token_type_ids` | `[0, 0, 0, 0, 0, 0, 0, 0]` | All 0s — single sentence, so everything belongs to segment A |
| `attention_mask` | `[1, 1, 1, 1, 1, 1, 1, 1]` | All 1s — every token should be attended to (no padding) |

**Key Takeaway:** BERT automatically wraps input with special tokens:
- `[CLS]` (101) at the start — used for classification tasks
- `[SEP]` (102) at the end — marks sentence boundary

Decoding confirmed this: `"[CLS] hello, i am single sentence [SEP]"` — note that `bert-base-uncased` lowercases everything.

---

### 2. Sentence Pair Encoding

```python
tokenizer("Hello, how are you?", "I am fine, thank you")
```

**Output:**
| Field | Value |
|-------|-------|
| `input_ids` | `[101, 7592, 1010, 2129, 2024, 2017, 1029, 102, 1045, 2572, 2986, 1010, 4067, 2017, 102]` |
| `token_type_ids` | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]` |
| `attention_mask` | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` |

**Key Takeaway:** When encoding two sentences:
- Both are joined with `[SEP]` between them: `[CLS] sent1 [SEP] sent2 [SEP]`
- **`token_type_ids`** now matters — `0` for the first sentence, `1` for the second
- This is how BERT distinguishes between sentence pairs (used in tasks like NLI, QA)

---

### 3. Returning PyTorch Tensors

```python
tokenizer("Hello, how are you?", "I am fine, Thank you", padding=True, return_tensors="pt")
```

**Key Takeaway:** Adding `return_tensors="pt"` returns PyTorch tensors instead of plain lists — required for feeding directly into a model. The `padding=True` argument pads shorter sequences in a batch to match the longest (not visible here with a single pair).

---

### 4. Truncation of Long Sentences

```python
tokenizer("This is a very very very ... long sentence.", truncation=True)
```

**Result:** 57 token IDs — the sentence was NOT truncated because it was still under BERT's 512-token limit.

**Key Takeaway:** `truncation=True` tells the tokenizer to cut sequences that exceed the model's maximum length (512 for BERT). Here the sentence had ~50 "very" repetitions but still fit. If it exceeded 512 tokens, it would have been clipped.

---

### 5. Batch Encoding with Max Length

```python
tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True, truncation=True, max_length=5, return_tensors="pt"
)
```

**Output:**
```
input_ids: [[ 101, 2129, 2024, 2017,  102],
            [ 101, 1045, 1005, 1049,  102]]
```

**Key Takeaway:** This is the most practical usage pattern:
- **Batch encoding** — pass a list of strings to encode multiple sentences at once
- **`max_length=5`** — forces truncation to exactly 5 tokens (including `[CLS]` and `[SEP]`), so only 3 actual word tokens fit
- **`padding=True`** — shorter sequences get padded to match the longest in the batch
- **`return_tensors="pt"`** — returns a batch-ready PyTorch tensor
- Notice `"I'm fine, thank you!"` was truncated to just `["I", "'", "m"]` (IDs: 1045, 1005, 1049)

---

## Concepts Cheat Sheet

| Concept | What It Does |
|---------|-------------|
| **`input_ids`** | Numerical token IDs that the model reads |
| **`token_type_ids`** | Segment IDs — distinguishes sentence A (0) from sentence B (1) |
| **`attention_mask`** | 1 = real token, 0 = padding token (model should ignore) |
| **`[CLS]`** (101) | Special classification token prepended to every input |
| **`[SEP]`** (102) | Separator token — marks end of sentence or boundary between pairs |
| **`padding`** | Adds `[PAD]` tokens so all sequences in a batch have equal length |
| **`truncation`** | Clips sequences that exceed `max_length` or the model's limit |
| **`return_tensors`** | `"pt"` for PyTorch, `"tf"` for TensorFlow, `"np"` for NumPy |

---

## Error Encountered

Cell 10 referenced `encoded_sequences` which was not defined — this was a leftover variable name. The fix would be to use `encoded_input["input_ids"]` instead:

```python
model_inputs = torch.tensor(encoded_input["input_ids"])
```

---

## Next Steps

The remaining notebook sections (Text Generation, Sentiment Analysis, Embeddings, Fill-Mask, QA) are ready to run and explore in future sessions.
