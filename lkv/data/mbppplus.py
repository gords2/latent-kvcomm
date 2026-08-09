from datasets import load_dataset

_TEMPLATE = """Please provide a self-contained Python script that solves the following problem in a markdown code block:
```python
YOUR_PYTHON_CODE
```:
{prompt}
Your answer will be tested on test cases like:
{t0}
{t1}
{t2}
"""


def load(split: str = "test"):
    ds = load_dataset("evalplus/mbppplus", split=split)
    items = []
    for row in ds:
        tests = list(row["test_list"])[:3]
        while len(tests) < 3:
            tests.append("")
        items.append({
            "question": _TEMPLATE.format(prompt=row["prompt"], t0=tests[0],
                                         t1=tests[1], t2=tests[2]),
            "gold": str(row["test"]),
        })
    return items
