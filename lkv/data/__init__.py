"""Task loading. max_samples slices deterministically after a seeded shuffle so
that two runs with the same seed compare item-for-item."""
from __future__ import annotations

import random

from lkv.data import countries, gsm8k, hotpotqa, mbppplus

_LOADERS = {
    "gsm8k": gsm8k.load,
    "mbppplus": mbppplus.load,
    "hotpotqa": hotpotqa.load,
    "countries": countries.load,
}

# Paper Table 4 sample counts, used when max_samples is -1.
PAPER_N = {"countries": 200, "tipsheets": 500, "hotpotqa": 500}


def load_task(name: str, max_samples: int = -1, seed: int = 42) -> list[dict]:
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; have {sorted(_LOADERS)}") from None
    items = loader()
    target = max_samples if max_samples > 0 else PAPER_N.get(name, -1)
    if target > 0 and target < len(items):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(items)), target))
        items = [items[i] for i in idx]
    return items
