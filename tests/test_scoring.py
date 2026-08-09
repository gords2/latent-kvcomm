"""D10: KVComm's code metric (fraction with F1 > 0.5) differs from the paper's
stated 'F1 score'. Both are implemented; Task 15 decides which reproduces."""
from lkv.scoring import (
    extract_gsm8k_answer,
    extract_python_block,
    f1_match,
    f1_score,
    normalize,
    run_python,
    score_item,
)


def test_boxed_answer_wins_over_trailing_numbers():
    assert extract_gsm8k_answer("blah \\boxed{42} then 99 more") == "42"


def test_falls_back_to_last_number():
    assert extract_gsm8k_answer("first 7 then 12") == "12"


def test_returns_none_when_no_number():
    assert extract_gsm8k_answer("no digits here") is None


def test_comma_formatted_gold_is_normalized():
    """Upstream's regex cannot span '1,000'; ours must."""
    assert normalize(extract_gsm8k_answer("\\boxed{1,000}")) == "1000"


def test_last_python_block_is_taken():
    text = "```python\nx=1\n```\ntext\n```python\ny=2\n```"
    assert extract_python_block(text) == "y=2"


def test_run_python_success_and_failure():
    ok, err = run_python("assert 1 == 1")
    assert ok is True and err is None
    ok, err = run_python("assert 1 == 2")
    assert ok is False and "AssertionError" in err


def test_run_python_times_out():
    ok, err = run_python("while True: pass", timeout=2)
    assert ok is False and "Timeout" in err


def test_f1_is_continuous_and_match_is_boolean():
    assert f1_score("the red car", "the red car") == 1.0
    partial = f1_score("the red car", "the red truck")
    assert 0.0 < partial < 1.0
    assert f1_match("the red car", "the red car") is True
    assert f1_match("the red car", "completely different words") is False


def test_f1_match_threshold_is_strictly_greater():
    """utils/f1.py:65 uses `> threshold`, not `>=`."""
    assert f1_match("a b", "a b", threshold=1.0) is False


def test_score_item_gsm8k():
    correct, score = score_item("gsm8k", "so \\boxed{18}", {"gold": "18"})
    assert correct is True and score == 1.0
    correct, score = score_item("gsm8k", "so \\boxed{17}", {"gold": "18"})
    assert correct is False and score == 0.0


def test_score_item_mbppplus_executes_gold_tests():
    item = {"gold": "assert add(1, 2) == 3"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert score_item("mbppplus", good, item)[0] is True
    assert score_item("mbppplus", bad, item)[0] is False
    assert score_item("mbppplus", "no code block", item)[0] is False


def test_score_item_hotpotqa_returns_both_metrics():
    correct, score = score_item("hotpotqa", "the red car", {"gold": "red car"})
    assert isinstance(correct, bool) and 0.0 <= score <= 1.0


def test_f1_score_lemmatizes_both_sides():
    """Regenerated brief finding #2: stock-kvcomm/utils/f1.py:12-13 lemmatizes
    both reference and candidate before comparing (metric_tools.py:188-210). A
    plural/verb-inflection-only mismatch has zero overlap in raw tokens but
    should be a perfect match once both sides are lemmatized to their lemma."""
    reference, candidate = "the dog runs", "dogs running"
    raw_ref = set(reference.lower().split()) - {"the"}
    raw_cand = set(candidate.lower().split())
    assert not (raw_ref & raw_cand), "test setup: raw tokens must not overlap"
    assert f1_score(reference, candidate) == 1.0


def test_f1_score_is_symmetric():
    """utils/f1.py:18-27 divides precision/recall by the reference and
    candidate set sizes respectively (transposed vs. the usual convention),
    but F1 is the harmonic mean of the two, which is symmetric in P and R --
    so f1_score(a, b) == f1_score(b, a) regardless. Do not 'fix' the
    transposition; it is harmless and preserved on purpose."""
    a, b = "the red car sped away", "a red truck stopped"
    assert f1_score(a, b) == f1_score(b, a)
    assert 0.0 < f1_score(a, b) < 1.0
