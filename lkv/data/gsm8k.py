from datasets import load_dataset

from lkv.scoring import extract_gold_gsm8k, normalize


def load(split: str = "test"):
    ds = load_dataset("gsm8k", "main", split=split)
    return [
        {"question": row["question"].strip(),
         "gold": normalize(extract_gold_gsm8k(row["answer"]))}
        for row in ds
    ]
