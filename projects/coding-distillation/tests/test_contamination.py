from distill.contamination import check, jaccard, ngrams

HE_PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''
HE = dict(he_prompts=[HE_PROMPT], he_entry_points={"has_close_elements"})


def test_jaccard_basics():
    assert jaccard(set(), set()) == 0.0
    a = ngrams("one two three four five six")
    assert jaccard(a, a) == 1.0


def test_rejects_renamed_copy():
    copy = HE_PROMPT.replace("has_close_elements", "any_pair_within")
    assert check(copy, "any_pair_within", **HE) == "contamination_ngram"


def test_rejects_same_entry_point_name():
    prompt = 'def has_close_elements(s: str) -> bool:\n    """Return True if s is a palindrome."""\n'
    assert check(prompt, "has_close_elements", **HE) == "contamination_name"


def test_accepts_unrelated():
    prompt = 'def run_length_encode(s: str) -> str:\n    """Compress runs of repeated characters.\n    >>> run_length_encode("aaab")\n    \'a3b1\'\n    """\n'
    assert check(prompt, "run_length_encode", **HE) is None
