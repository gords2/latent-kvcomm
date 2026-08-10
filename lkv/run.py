"""Entry point. One JSON per run, always including per_item."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import torch

from lkv.config import RunConfig, parse_args
from lkv.data import load_task
from lkv.kvcomm.importance import select_layers
from lkv.kvcomm.pair import calibrate, run_baseline, run_pair, run_skyline
from lkv.latentmas import run_latentmas
from lkv.modeling import load_model
from lkv.scoring import score_item


def build_record(cfg: RunConfig, items, predictions, per_item, hops, timings,
                 layer_retention: float | None = None,
                 kept_layers: list[int] | None = None) -> dict:
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
        # Named for what it is. Hop metadata is captured from the FIRST batch
        # only -- cache lengths differ per batch because prompts differ in
        # length, so this is one representative relay, not a whole-run summary.
        # Anything that needs a run-level figure must aggregate across batches
        # rather than reading a single entry out of here.
        "hop_records_first_batch": hops,
        "total_sec": timings["total_sec"],
        "sec_per_item": (timings["total_sec"] / n) if n else 0.0,
        # None for every system except kvcomm, where calibrate()+select_layers()
        # ran once before the batch loop -- see main().
        "layer_retention": layer_retention,
        "kept_layers": kept_layers,
    }


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def main(argv=None) -> int:
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    items = load_task(cfg.task, cfg.max_samples, cfg.seed)
    model, tokenizer = load_model(cfg.model_name, cfg.device)
    # Same-model is the default case (RunConfig.model_b defaults to
    # model_name), so reuse the object rather than loading it twice --
    # loading it again would double GPU memory for the common case.
    model_b = model if cfg.model_b == cfg.model_name else load_model(cfg.model_b, cfg.device)[0]

    # kvcomm calibrates ONCE, at full retention, before any batch runs -- not
    # per batch. The resulting layer selection is then reused for every batch.
    layer_retention: float | None = None
    kept_layers: list[int] | None = None
    if cfg.system == "kvcomm":
        totals = calibrate(model, tokenizer, items, cfg)
        kept_layers = select_layers(totals, cfg.top_layers, cfg.alpha, cfg.mu,
                                    cfg.sigma, cfg.random_selection, cfg.seed)
        layer_retention = len(kept_layers) / model.config.num_hidden_layers

    predictions: list[str] = [""] * len(items)
    per_item: list[dict] = [None] * len(items)
    hops: list[dict] = []
    started = time.time()

    for offset, batch in _batches(items, cfg.generate_bs):
        batch_hops: list[dict] = []
        if cfg.system in ("latentmas", "single"):
            # --system single is "the judger alone": same config, n_agents=1.
            # dataclasses.replace copies every other field (including extra)
            # from the existing instance, so nothing can be dropped or
            # duplicated the way a hand-rolled asdict/dict-merge could.
            local_cfg = cfg if cfg.system == "latentmas" else replace(cfg, n_agents=1)
            texts, batch_hops = run_latentmas(model, tokenizer, local_cfg, batch)
        elif cfg.system == "kvcomm":
            texts = run_pair(model, model_b, tokenizer, cfg, batch, kept_layers)
        elif cfg.system == "skyline":
            texts = run_skyline(model, tokenizer, cfg, batch)
        elif cfg.system == "baseline":
            texts = run_baseline(model, tokenizer, cfg, batch)
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
                          {"total_sec": time.time() - started},
                          layer_retention=layer_retention, kept_layers=kept_layers)
    payload = json.dumps(record, indent=2)
    if cfg.out_json:
        with open(cfg.out_json, "w") as fh:
            fh.write(payload)
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("per_item", "hop_records_first_batch")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
