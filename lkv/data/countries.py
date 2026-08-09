"""KVComm bundles countries.jsonl (200 items) in-repo rather than on the Hub.

Place a copy at lkv/data/files/countries.jsonl with one object per line:
    {"context": "...", "question": "...", "answer": "..."}
Fail loudly if it is missing -- silently returning [] would report accuracy 0.0
as if it were a measurement.
"""
import json
from pathlib import Path

INSTRUCTION = "Answer the question using the given context."
_PATH = Path(__file__).parent / "files" / "countries.jsonl"


def load(split: str = "test"):
    if not _PATH.exists():
        raise FileNotFoundError(
            f"{_PATH} missing. Copy it from the KVComm repo "
            "(dataloader/data/countries.jsonl)."
        )
    items = []
    with _PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            items.append({
                "instruction": INSTRUCTION,
                "context": row["context"],
                "question": row["question"],
                "gold": row["answer"],
            })
    return items
