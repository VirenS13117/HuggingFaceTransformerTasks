"""Keep generated training problems from overlapping with HumanEval."""
import re


def load_humaneval():
    from datasets import load_dataset

    return list(load_dataset("openai/openai_humaneval", split="test"))


def ngrams(text, n=5):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def check(prompt, entry_point, he_prompts, he_entry_points, threshold=0.3):
    if entry_point in he_entry_points:
        return "contamination_name"
    grams = ngrams(prompt)
    if any(jaccard(grams, ngrams(p)) >= threshold for p in he_prompts):
        return "contamination_ngram"
    return None
