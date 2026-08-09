"""Answer extraction, sandboxed execution, and both F1 variants.

score_item returns (correct, score): `correct` is the binary outcome that goes
into per_item and McNemar; `score` is the continuous value so a mean-F1 metric
can be computed from the same run. For GSM8K and MBPP+ they coincide.
"""
from __future__ import annotations

import re
import traceback
from multiprocessing import Manager, Process

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_PY_BLOCK = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_gsm8k_answer(text: str) -> str | None:
    boxes = _BOXED.findall(text)
    if boxes:
        m = _NUMBER.search(boxes[-1])
        return m.group(0) if m else boxes[-1].strip()
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


def extract_gold_gsm8k(solution: str) -> str | None:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", solution)
    return m.group(1) if m else None


def normalize(text: str | None) -> str | None:
    if text is None:
        return None
    out = text.strip().lower().replace(",", "")
    return out.rstrip(".") or None


def extract_python_block(text: str) -> str | None:
    blocks = _PY_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else None


def _worker(ns, code):
    try:
        exec(code, {})
        ns["ok"], ns["error"] = True, None
    except BaseException:
        ns["ok"], ns["error"] = False, traceback.format_exc()


def run_python(code: str, timeout: int = 10):
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=_worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False, f"TimeoutError: exceeded {timeout} seconds"
        return bool(ns.get("ok", False)), ns.get("error", "process died")


_PUNCT = re.compile(r"[^a-z0-9 ]")
_ARTICLES = {"a", "an", "the"}


def _tokens(text: str) -> list[str]:
    cleaned = _PUNCT.sub(" ", text.lower())
    return [t for t in cleaned.split() if t and t not in _ARTICLES]


def f1_score(reference: str, candidate: str) -> float:
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return float(ref == cand)
    common = set(ref) & set(cand)
    if not common:
        return 0.0
    precision = len(common) / len(set(cand))
    recall = len(common) / len(set(ref))
    return 2 * precision * recall / (precision + recall)


def f1_match(reference: str, candidate: str, threshold: float = 0.5) -> bool:
    """KVComm's implemented metric: utils/f1.py:65 uses strict `>`."""
    return f1_score(reference, candidate) > threshold


def score_item(task: str, prediction: str, item: dict) -> tuple[bool, float]:
    gold = item.get("gold", "")
    if task == "gsm8k":
        pred = normalize(extract_gsm8k_answer(prediction))
        ok = bool(pred) and bool(normalize(gold)) and pred == normalize(gold)
        return ok, float(ok)
    if task == "mbppplus":
        code = extract_python_block(prediction)
        if code is None:
            return False, 0.0
        ok, _ = run_python(code + "\n" + gold)
        return ok, float(ok)
    if task in ("hotpotqa", "countries", "tipsheets"):
        return f1_match(gold, prediction), f1_score(gold, prediction)
    raise ValueError(f"no scorer for task {task!r}")
