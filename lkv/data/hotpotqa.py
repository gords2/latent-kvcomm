"""HotpotQA, matching KVComm's own context construction.

CRITICAL: the sender's context is ONLY the supporting-fact sentences, not the
whole distractor paragraph set. Upstream builds it with `construct_support`
(stock-kvcomm/dataloader/hotpotqa.py:4-14): for each (title, sent_id) in
`supporting_facts`, pull that one sentence out of the matching context
paragraph, then join with newlines (`:34-35`, `prompt_A`). Handing the model all
ten paragraphs instead would make the context roughly an order of magnitude
larger and mostly distractors, which would depress every KVComm score in a way
that looks like the method failing rather than the harness being wrong.
"""
from datasets import load_dataset

INSTRUCTION = "Answer the question using the given context."


def construct_support(row) -> list[str]:
    """The supporting-fact sentences only, in supporting_facts order."""
    facts = row["supporting_facts"]
    ctx = row["context"]
    out = []
    for title, sent_id in zip(facts["title"], facts["sent_id"]):
        for ctx_title, sentences in zip(ctx["title"], ctx["sentences"]):
            if ctx_title == title:
                if 0 <= sent_id < len(sentences):
                    out.append(sentences[sent_id])
                # Upstream prints a warning and drops it; a missing sentence is
                # a dataset artifact, not something to fail the run over.
    return out


def load(split: str = "validation"):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    items = []
    for row in ds:
        items.append({
            "instruction": INSTRUCTION,
            "context": "\n".join(construct_support(row)),
            "question": row["question"],
            "gold": row["answer"],
        })
    return items
