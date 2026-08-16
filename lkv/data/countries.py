"""KVComm bundles countries.jsonl (200 items) in-repo rather than on the Hub.

Copy it from the KVComm repo (dataloader/data/countries.jsonl). Fail loudly if
it is missing -- silently returning [] would report accuracy 0.0 as if it were
a measurement.

THE REAL SCHEMA (verified against the actual file, 2026-08-15):

    {"id": "countries_0000", "person": "Uma", "landmark": "Mahaffie House",
     "country": "United States", "prompt_A": "Uma is at the Mahaffie House.",
     "prompt_B": "Which country is Uma located in?", "y": "United States"}

There is no "context", "question" or "answer" key. This loader previously
guessed those names and raised KeyError on the first real load -- invisible
until the file existed, because it is deliberately absent from the repo.

The mapping mirrors upstream's CountriesEvaluator
(stock-kvcomm/dataloader/countries.py:15-22), which drops `id`/`person`,
renames `country` -> `answer`, and appends a sentence to `prompt_B`.
`prompt_A`/`prompt_B` are context/question across every upstream task -- the
same convention hotpotqa.py follows.

`y` is ignored because upstream uses `country`; the two agree on all 200 items
(checked), so this is a fidelity choice, not a behavioural one.
"""
import json
from pathlib import Path

from lkv.prompts import QA_INSTRUCTION

# Upstream's, verbatim -- NOT a paraphrase. See its definition for why the
# exact wording decides whether this task scores at all (D18).
INSTRUCTION = QA_INSTRUCTION

# Appended to prompt_B verbatim and WITHOUT a separating space, because upstream
# does exactly that: `x['prompt_B'] + "You are required to..."` (countries.py:21).
# The missing space is theirs; reproducing their numbers means reproducing it.
_QUESTION_SUFFIX = "You are required to extrapolate the country from the context."

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
                "context": row["prompt_A"],
                "question": row["prompt_B"] + _QUESTION_SUFFIX,
                "gold": row["country"],
            })
    return items
