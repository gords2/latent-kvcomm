"""Regenerated Task 7 review finding #1: HotpotQA's sender context must be the
supporting-fact sentences only (stock-kvcomm/dataloader/hotpotqa.py:4-14), not
the whole ten-paragraph distractor set. A synthetic row stands in for a real
dataset row so this is testable without network -- construct_support itself
takes a plain nested-dict row and needs no download."""
from lkv.data.hotpotqa import construct_support


def _row(supporting_titles, supporting_sent_ids):
    return {
        "context": {
            "title": ["Paragraph A", "Paragraph B"],
            "sentences": [
                ["A0. ", "A1. ", "A2. "],
                ["B0. ", "B1. "],
            ],
        },
        "supporting_facts": {
            "title": supporting_titles,
            "sent_id": supporting_sent_ids,
        },
    }


def test_construct_support_pulls_only_named_sentences_in_order():
    row = _row(["Paragraph B", "Paragraph A"], [1, 0])
    result = construct_support(row)
    assert result == ["B1. ", "A0. "]
    # The other sentences in both paragraphs are distractors and must not leak in.
    assert "A1. " not in result
    assert "A2. " not in result
    assert "B0. " not in result


def test_construct_support_drops_out_of_range_sent_id():
    """Upstream prints a warning and drops the entry rather than raising
    (dataloader/hotpotqa.py:9-14) -- a missing sentence is a dataset artifact,
    not something that should fail the run."""
    row = _row(["Paragraph B", "Paragraph A", "Paragraph A"], [1, 0, 5])
    result = construct_support(row)
    assert result == ["B1. ", "A0. "]
