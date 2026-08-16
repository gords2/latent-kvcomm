"""Tipsheets: pick one of three companies from a sentiment-laden briefing.

Bundled in the KVComm repo (dataloader/data/tipsheets.jsonl, 500 rows), not on
the Hub -- same as countries.jsonl. Fail loudly if it is missing; returning []
would report accuracy 0.0 as though it were a measurement.

THE REAL SCHEMA (read off the actual file, 2026-08-16 -- not guessed, which is
how D17 happened to the countries loader):

    {"prompt_A": "Atlas LLC is under pressure amid softer trends; EPS -17% ...",
     "prompt_B": "You must invest in exactly one company from Atlas LLC, Sable
                  LLC, Trace LLC. Which do you choose? You must only answer
                  with the company name.\\nThe company name is:",
     "options": ["Atlas LLC", "Sable LLC", "Trace LLC"],
     "label": "Sable LLC",
     "attitudes": {"Atlas LLC": "negative", "Sable LLC": "positive",
                   "Trace LLC": "neutral"}}

Two differences from countries.py, both deliberate:

  * NO suffix is appended to prompt_B. Upstream's CountriesEvaluator maps a
    sentence onto its question; TipsheetsEvaluator does not -- it only renames
    `label` -> `answer`. prompt_B already ends in its own answer prefix,
    "The company name is:", which is exactly why upstream's think-model branch
    skips adding one for this task: "there is already an answer prefix in the
    data" (eval.py:56-58).
  * `max_tokens` is 10, not 5 (TipsheetsEvaluator). Already in config's
    _MAX_NEW_TOKENS.

`options` and `attitudes` are carried through untouched. Nothing reads them
today; they are what a per-attitude breakdown would need (does the receiver
inherit the sender's sentiment through the cache?), and dropping them here
would mean re-reading the corpus later to get them back.
"""
import json
from pathlib import Path

from lkv.prompts import QA_INSTRUCTION

# Upstream's, verbatim. Tipsheets takes eval.py's QA branch like every other
# non-math, non-code, non-summarise task -- see D18.
INSTRUCTION = QA_INSTRUCTION

_PATH = Path(__file__).parent / "files" / "tipsheets.jsonl"


def load(split: str = "test"):
    if not _PATH.exists():
        raise FileNotFoundError(
            f"{_PATH} missing. Copy it from the KVComm repo "
            "(dataloader/data/tipsheets.jsonl)."
        )
    items = []
    with _PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            items.append({
                "instruction": INSTRUCTION,
                "context": row["prompt_A"],
                "question": row["prompt_B"],
                "gold": row["label"],
                "options": row["options"],
                "attitudes": row["attitudes"],
            })
    return items
