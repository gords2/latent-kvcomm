from datasets import load_dataset

INSTRUCTION = "Answer the question using the given context."


def load(split: str = "validation"):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    items = []
    for row in ds:
        paras = ["".join(s) for s in row["context"]["sentences"]]
        items.append({
            "instruction": INSTRUCTION,
            "context": "\n".join(paras),
            "question": row["question"],
            "gold": row["answer"],
        })
    return items
