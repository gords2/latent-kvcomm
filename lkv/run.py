"""Entry point. One JSON per run, always including per_item."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import torch

from lkv.config import RunConfig, parse_args
from lkv.data import load_task
from lkv.latentmas import run_latentmas
from lkv.modeling import load_model
from lkv.scoring import score_item


def build_record(cfg: RunConfig, items, predictions, per_item, hops, timings) -> dict:
    n = len(items)
    if not (len(predictions) == len(per_item) == n):
        raise ValueError(
            f"length mismatch: items={n} predictions={len(predictions)} "
            f"per_item={len(per_item)}"
        )
    correct = sum(1 for p in per_item if p["correct"])
    return {
        **{k: v for k, v in asdict(cfg).items() if k != "extra"},
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "mean_score": (sum(p["score"] for p in per_item) / n) if n else 0.0,
        "per_item": per_item,
        "hop_records": hops,
        "total_sec": timings["total_sec"],
        "sec_per_item": (timings["total_sec"] / n) if n else 0.0,
    }


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def main(argv=None) -> int:
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    items = load_task(cfg.task, cfg.max_samples, cfg.seed)
    model, tokenizer = load_model(cfg.model_name, cfg.device)

    predictions: list[str] = [""] * len(items)
    per_item: list[dict] = [None] * len(items)
    hops: list[dict] = []
    started = time.time()

    for offset, batch in _batches(items, cfg.generate_bs):
        if cfg.system in ("latentmas", "single"):
            # --system single is "the judger alone": same config, n_agents=1.
            # dataclasses.replace copies every other field (including extra)
            # from the existing instance, so nothing can be dropped or
            # duplicated the way a hand-rolled asdict/dict-merge could.
            local_cfg = cfg if cfg.system == "latentmas" else replace(cfg, n_agents=1)
            texts, batch_hops = run_latentmas(model, tokenizer, local_cfg, batch)
        else:
            raise NotImplementedError(f"system {cfg.system!r} arrives in a later task")
        if offset == 0:
            hops = batch_hops
        for j, text in enumerate(texts):
            i = offset + j
            ok, score = score_item(cfg.task, text, batch[j])
            predictions[i] = text
            per_item[i] = {"i": i, "correct": bool(ok), "score": float(score)}

    record = build_record(cfg, items, predictions, per_item, hops,
                          {"total_sec": time.time() - started})
    payload = json.dumps(record, indent=2)
    if cfg.out_json:
        with open(cfg.out_json, "w") as fh:
            fh.write(payload)
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("per_item", "hop_records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
