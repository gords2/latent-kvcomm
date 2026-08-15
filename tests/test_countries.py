"""The countries loader, pinned to the REAL jsonl schema.

This exists because the loader shipped reading "context"/"question"/"answer" --
keys that do not occur in the file at all. It raised KeyError on the very first
real load, on the GPU box, after the environment was built. Nothing caught it
earlier: the data file is deliberately absent from the repo (countries.jsonl is
bundled in KVComm, not on the Hub), so anything touching this module got
FileNotFoundError first and never reached the mapping.

The fixture below therefore writes a synthetic file using the ACTUAL key names,
so the mapping is verified without the corpus being present.
"""
import json

import pytest

from lkv.data import countries

# Two records in the real schema, values synthetic. The second row deliberately
# differs from the first so a loader returning a constant, or reading the wrong
# column, cannot pass.
_ROWS = [
    {"id": "countries_0000", "person": "Uma", "landmark": "Mahaffie House",
     "country": "United States", "prompt_A": "Uma is at the Mahaffie House.",
     "prompt_B": "Which country is Uma located in?", "y": "United States"},
    {"id": "countries_0001", "person": "Bo", "landmark": "Sagrada Familia",
     "country": "Spain", "prompt_A": "Bo is at the Sagrada Familia.",
     "prompt_B": "Which country is Bo located in?", "y": "Spain"},
]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    path = tmp_path / "countries.jsonl"
    _write(path, _ROWS)
    monkeypatch.setattr(countries, "_PATH", path)
    return path


def test_load_maps_the_real_schema_onto_harness_fields(corpus):
    items = countries.load()
    assert len(items) == 2

    first = items[0]
    assert sorted(first) == ["context", "gold", "instruction", "question"]
    assert first["context"] == "Uma is at the Mahaffie House.", (
        "context must come from prompt_A -- it is the sender's information, and "
        "prompt_A is the context column across every upstream task")
    assert first["gold"] == "United States"
    assert first["instruction"] == countries.INSTRUCTION

    # Not a constant: the second row must carry its own values.
    assert items[1]["gold"] == "Spain"
    assert items[1]["context"] == "Bo is at the Sagrada Familia."


def test_question_appends_the_upstream_suffix_without_a_space(corpus):
    """Upstream concatenates with no separator:
    `x['prompt_B'] + "You are required to..."` (dataloader/countries.py:21).

    The missing space is upstream's, and it lands in the receiver prompt that
    produced their published 0.51/0.62/0.62. Silently 'fixing' it would change
    the tokenisation of every item in the task and make a reproduction attempt
    unfalsifiable -- a mismatch could no longer separate our bug from theirs.
    """
    q = countries.load()[0]["question"]
    assert q == ("Which country is Uma located in?"
                 "You are required to extrapolate the country from the context.")
    assert "in?You are required" in q, "the absent space is deliberate"


def test_missing_corpus_raises_rather_than_returning_empty(tmp_path, monkeypatch):
    """Returning [] would report accuracy 0.0 as though it were a measurement."""
    monkeypatch.setattr(countries, "_PATH", tmp_path / "absent.jsonl")
    with pytest.raises(FileNotFoundError, match="countries"):
        countries.load()


def test_y_column_is_not_used(corpus):
    """Upstream renames `country` -> `answer` and never reads `y`. The two agree
    on all 200 real items, so this pins a fidelity choice rather than a
    behaviour: if a future edit switches to `y`, that should be noticed.
    """
    rows = [dict(r) for r in _ROWS]
    rows[0]["y"] = "SENTINEL-NOT-USED"
    _write(corpus, rows)
    assert countries.load()[0]["gold"] == "United States"
