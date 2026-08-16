"""The tipsheets loader, pinned against a synthetic file using the REAL keys.

Same shape as tests/test_countries.py and for the same reason: the corpus is
deliberately absent from this repo, so a loader that reads keys the file does
not have would pass every test and raise KeyError on the first GPU run. That is
exactly D17, which cost a run. The fixture below therefore uses the key names
read off the actual tipsheets.jsonl -- prompt_A, prompt_B, options, label,
attitudes -- not a plausible guess at them.
"""
import json

import pytest

from lkv.data import load_task, tipsheets

# One real row, trimmed, plus a second to prove ordering is preserved.
_ROWS = [
    {
        "prompt_A": "Atlas LLC is under pressure; EPS -17%. Sable LLC shows "
                    "clear momentum. Trace LLC looks balanced.",
        "prompt_B": "You must invest in exactly one company from Atlas LLC, "
                    "Sable LLC, Trace LLC. Which do you choose? You must only "
                    "answer with the company name.\nThe company name is:",
        "options": ["Atlas LLC", "Sable LLC", "Trace LLC"],
        "label": "Sable LLC",
        "attitudes": {"Atlas LLC": "negative", "Sable LLC": "positive",
                      "Trace LLC": "neutral"},
    },
    {
        "prompt_A": "Apex LLC is gaining traction. Borealis LLC is flat.",
        "prompt_B": "You must invest in exactly one company from Apex LLC, "
                    "Borealis LLC. Which do you choose?\nThe company name is:",
        "options": ["Apex LLC", "Borealis LLC"],
        "label": "Apex LLC",
        "attitudes": {"Apex LLC": "positive", "Borealis LLC": "neutral"},
    },
]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    path = tmp_path / "tipsheets.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    monkeypatch.setattr(tipsheets, "_PATH", path)
    return path


def test_loader_maps_the_real_columns(corpus):
    items = tipsheets.load()
    assert len(items) == 2
    first = items[0]
    assert first["context"] == _ROWS[0]["prompt_A"]
    assert first["gold"] == "Sable LLC", "gold comes from `label`, upstream's `answer`"
    assert first["options"] == ["Atlas LLC", "Sable LLC", "Trace LLC"]
    assert first["attitudes"]["Sable LLC"] == "positive"
    assert items[1]["gold"] == "Apex LLC", "row order must be preserved"


def test_question_is_prompt_b_verbatim_with_no_suffix(corpus):
    """Countries appends a sentence to prompt_B; tipsheets must NOT.

    Upstream's TipsheetsEvaluator only renames label -> answer, and prompt_B
    already ends in its own answer prefix ("The company name is:") -- which is
    why eval.py's think-model branch skips adding one for this task. Appending
    anything here would push that prefix away from the generation point, and
    the whole budget is 10 tokens.
    """
    item = tipsheets.load()[0]
    assert item["question"] == _ROWS[0]["prompt_B"]
    assert item["question"].endswith("The company name is:")


def test_instruction_is_the_shared_qa_string(corpus):
    from lkv.prompts import QA_INSTRUCTION
    assert tipsheets.load()[0]["instruction"] == QA_INSTRUCTION


def test_task_is_registered_and_reachable_through_load_task(corpus):
    """Registration in _LOADERS is the step that was missing -- the loader
    existing is not enough, `--task tipsheets` has to resolve to it."""
    items = load_task("tipsheets", max_samples=1, seed=42)
    assert len(items) == 1
    assert items[0]["gold"] in ("Sable LLC", "Apex LLC")


def test_missing_corpus_raises_instead_of_reporting_zero(tmp_path, monkeypatch):
    """A silent [] would surface as accuracy 0.0000 -- indistinguishable from a
    real measurement, which is the failure mode this project keeps hitting."""
    monkeypatch.setattr(tipsheets, "_PATH", tmp_path / "absent.jsonl")
    with pytest.raises(FileNotFoundError):
        tipsheets.load()
