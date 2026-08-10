"""Entry point. One JSON per run, always including per_item."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import torch

from lkv.config import RunConfig, parse_args
from lkv.data import load_task
from lkv.instrument import PeakMemory, Timer, attention_positions, cache_bytes
from lkv.integrate import make_trim_hook
from lkv.kvcomm.importance import select_layers
from lkv.kvcomm.pair import calibrate, run_baseline, run_pair, run_skyline
from lkv.latentmas import run_latentmas
from lkv.modeling import load_model
from lkv.relay_calib import calibrate_relay
from lkv.scoring import score_item


def _measuring_hook(base_hook, stats_out: dict):
    """Wrap a (possibly-None) trim hook so its LAST call each batch also
    measures the cache it hands back.

    run_latentmas returns only (final_texts, hop_records) -- Task 20 and the
    existing tests depend on that 2-tuple, so this is how the per-batch cache
    is reached WITHOUT widening the signature: trim_hook already receives the
    cache at every non-final hop, and the judger role that follows never
    touches it again, so the last call this wrapper sees in a batch is
    exactly the cache the judger generates against -- the batch's "final hop."

    Building this unconditionally (whenever the run is latentmas, regardless
    of whether a real trim hook exists) is what makes the reference condition
    measurable too: at top_layers==1.0 there is no trim hook to hang this off
    of, yet that untrimmed run is the baseline every other run's ratio is
    divided by, so it still needs its own cache_bytes/attention_positions.
    """
    def hook(cache, hop_index, inherited_len):
        out = base_hook(cache, hop_index, inherited_len) if base_hook is not None else cache
        stats_out["bytes"] = cache_bytes(out)
        stats_out["positions"] = attention_positions(out)
        return out
    return hook


def build_record(cfg: RunConfig, items, predictions, per_item, hops, timings,
                 layer_retention: float | None = None,
                 kept_layers: list[int] | None = None,
                 hook_records: list[dict] | None = None,
                 peak_mem_bytes: int = 0,
                 cache_bytes_at_final_hop_mean: float = 0.0,
                 attention_positions_at_final_hop_mean: float = 0.0,
                 cache_stat_batches: int = 0,
                 ragged_path: str = "n/a") -> dict:
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
        # None unless calibrate()+select_layers() ran before the batch loop --
        # always for kvcomm, and for latentmas only when top_layers<1.0 built a
        # trim hook. For kvcomm this is the static len(kept_layers)/n_layers
        # ratio; for latentmas it is the MEAN of the trim hook's own per-hop
        # `retention` (see hook_records below) -- a different quantity, since
        # scope and every_hop both change how much of the cache a layer
        # selection actually keeps. See main().
        "layer_retention": layer_retention,
        "kept_layers": kept_layers,
        # Task 17's evidence trail: one entry per trim_hook call from the
        # FIRST batch (see hop_records_first_batch's docstring above for why
        # only the first batch -- cache lengths differ per batch since
        # prompts differ in length, so this is one representative relay).
        # Empty for every system/config that never built a trim hook.
        "hook_records": hook_records if hook_records is not None else [],
        # Wall clock and peak allocator usage for the whole batch loop --
        # PeakMemory is a no-op (0) off CUDA, so this is always present but
        # only ever nonzero on the GPU box (see lkv/instrument.py).
        "peak_mem_bytes": peak_mem_bytes,
        # MEAN cache_bytes/attention_positions at each batch's final relay hop
        # (see _measuring_hook above), NOT a single batch's figure -- cache
        # lengths differ between batches because prompt lengths differ, so
        # one batch's number would silently be "whichever batch ran first"
        # rather than a run-level figure. KVComm's paper reports no measured
        # bytes at all, so these are new information, not a re-derivation.
        # Zero/empty for any run with no latentmas relay cache to measure
        # (system != "latentmas", e.g. kvcomm/skyline/baseline/single).
        "cache_bytes_at_final_hop_mean": cache_bytes_at_final_hop_mean,
        "attention_positions_at_final_hop_mean": attention_positions_at_final_hop_mean,
        # How many batches the two means above were averaged over -- lets a
        # reader (and Task 20) tell "averaged over 40 batches" apart from
        # "averaged over 1", which the mean alone cannot.
        "cache_stat_batches": cache_stat_batches,
        # Travels with the data rather than living only in docs/FINDINGS.md:
        # every top_layers<1.0 relay run measures non-causal attention at
        # every boundary after the first trim (D8, the LatentMAS-trim-hook
        # section) because run_latentmas hands the trimmed, ragged cache to
        # the stock forward path, not forward_with_ragged_cache. A reader
        # holding only this JSON file has no way to know that unless it is
        # recorded here. "n/a" for any run that never trimmed at all.
        "ragged_path": ragged_path,
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
    # latentmas does the same calibrate()+select_layers() step when
    # top_layers<1.0 asks for a trim hook at its agent boundaries (Task 17) --
    # kept_layers is the layer SELECTION either way; layer_retention's meaning
    # differs per system and is finalised below the batch loop (see the field's
    # docstring in build_record).
    layer_retention: float | None = None
    kept_layers: list[int] | None = None
    if cfg.system == "kvcomm":
        totals = calibrate(model, tokenizer, items, cfg)
        kept_layers = select_layers(totals, cfg.top_layers, cfg.alpha, cfg.mu,
                                    cfg.sigma, cfg.random_selection, cfg.seed)
        layer_retention = len(kept_layers) / model.config.num_hidden_layers
    elif cfg.system == "latentmas" and cfg.top_layers < 1.0:
        # calib_source="pair" reuses Task 14's KVComm-style calibration -- cheap,
        # and the honest baseline for "we just dropped KVComm in", but it needs
        # instruction/context fields gsm8k and mbppplus items don't have.
        # calib_source="relay" (the default) measures importance where it is
        # actually applied: the next role's prefill over the accumulated relay
        # cache (lkv/relay_calib.py) -- see that module's docstring for why a
        # KVComm-pair ranking would otherwise come from a different distribution.
        if cfg.calib_source == "relay":
            totals = calibrate_relay(model, tokenizer, cfg, items)
        else:
            totals = calibrate(model, tokenizer, items, cfg)
        kept_layers = select_layers(totals, cfg.top_layers, cfg.alpha, cfg.mu,
                                    cfg.sigma, cfg.random_selection, cfg.seed)

    predictions: list[str] = [""] * len(items)
    per_item: list[dict] = [None] * len(items)
    hops: list[dict] = []
    hook_records: list[dict] = []
    # Accumulated across batches, not read off any single one -- see
    # _measuring_hook and the cache_bytes_at_final_hop_mean docstring below.
    cache_byte_total = 0
    position_total = 0
    cache_stat_batches = 0
    # layer_retention (latentmas branch) is likewise a mean over EVERY batch's
    # hook, not just the first -- see the field's docstring in build_record
    # and the retention_sum/retention_count accumulation below. hook_records
    # itself stays first-batch-only (see its own docstring); this is the
    # separate run-level aggregate hop_records_first_batch's sibling field
    # `hop_records` was already fixed to have.
    retention_sum = 0.0
    retention_count = 0

    # Built ONCE, before the batch loop, and reused for every batch -- not
    # rebuilt per batch. LatentProjector lazily builds its [D, D] realign
    # matrix (and/or the cheaper target-norm-only path) on first use and
    # caches it; a fresh projector per batch would rebuild it every batch,
    # which is exactly the transient this shared instance avoids (see
    # lkv/realign.py and the projector= parameter on run_latentmas).
    projector = None
    if cfg.system in ("latentmas", "single"):
        from lkv.realign import LatentProjector
        projector = LatentProjector(model, cfg.realign, cfg.renorm_latent, cfg.device)

    with Timer() as timer, PeakMemory(cfg.device) as mem:
        for offset, batch in _batches(items, cfg.generate_bs):
            batch_hops: list[dict] = []
            if cfg.system in ("latentmas", "single"):
                # --system single is "the judger alone": same config, n_agents=1.
                # dataclasses.replace copies every other field (including extra)
                # from the existing instance, so nothing can be dropped or
                # duplicated the way a hand-rolled asdict/dict-merge could.
                local_cfg = cfg if cfg.system == "latentmas" else replace(cfg, n_agents=1)
                # A FRESH hook per batch: run_latentmas always starts each batch's
                # relay from cache=None, so every_hop=False's "trim only the first
                # boundary" must reset per batch too -- a hook shared across
                # batches would only ever trim in the very first batch and
                # silently pass every later batch through untouched. The same
                # goes for batch_cache_stats, the dict _measuring_hook writes
                # this batch's final-hop figures into.
                base_hook = (make_trim_hook(kept_layers, cfg.trim_scope,
                                            every_hop=cfg.trim_every_hop)
                            if cfg.system == "latentmas" and kept_layers is not None
                            else None)
                batch_cache_stats: dict = {}
                # Built for EVERY latentmas batch, even when base_hook is None
                # (top_layers==1.0, no trimming) -- that untrimmed run is the
                # reference every other run's ratio divides by, and it would
                # go unmeasured otherwise. "single" never calls hooks at all
                # (no non-final agent, hence nothing to measure), so leave it
                # out rather than pass a hook that can never fire.
                trim_hook = (_measuring_hook(base_hook, batch_cache_stats)
                            if cfg.system == "latentmas" else None)
                texts, batch_hops = run_latentmas(model, tokenizer, local_cfg, batch,
                                                  trim_hook=trim_hook, projector=projector)
                if base_hook is not None:
                    if offset == 0:
                        # First-batch-only inspection copy -- see
                        # hook_records's docstring in build_record. The
                        # run-level figure is retention_sum/retention_count
                        # below, accumulated from EVERY batch's hook, not
                        # just this one.
                        hook_records = base_hook.records
                    retention_sum += sum(r["retention"] for r in base_hook.records)
                    retention_count += len(base_hook.records)
                if batch_cache_stats:
                    cache_byte_total += batch_cache_stats["bytes"]
                    position_total += batch_cache_stats["positions"]
                    cache_stat_batches += 1
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

    if retention_count:
        # latentmas's own figure: the mean of what the trim hook actually kept
        # at each boundary, ACROSS EVERY BATCH -- not the static
        # len(kept_layers)/n_layers ratio (scope "all" vs "delta" and
        # every_hop both change how much of the cache a given layer selection
        # ends up retaining), and not just the first batch's hook_records
        # either (cache lengths, and therefore retention, can differ batch to
        # batch since prompt lengths differ -- the same reason
        # cache_bytes_at_final_hop_mean above is accumulated rather than read
        # off one batch).
        layer_retention = retention_sum / retention_count
    elif cfg.system == "latentmas" and layer_retention is None:
        # No trim hook ever ran for this latentmas run (top_layers==1.0, or a
        # degenerate n_agents==1 with no boundary to trim at) -- that is full
        # retention, a known fact about the run, not an unknown. None stays
        # reserved for systems with no relay/trim concept at all (single,
        # skyline, baseline).
        layer_retention = 1.0

    # See build_record's ragged_path docstring: honest only when a real
    # layer selection (kept_layers) was actually used to trim a cache below
    # top_layers==1.0 -- kvcomm always computes kept_layers even at
    # top_layers==1.0 (a no-op selection), and single/skyline/baseline never
    # compute one at all, so gating on top_layers alone would mislabel both.
    ragged_path = (
        "stock-noncausal" if (kept_layers is not None and cfg.top_layers < 1.0)
        else "n/a"
    )

    record = build_record(
        cfg, items, predictions, per_item, hops, {"total_sec": timer.elapsed},
        layer_retention=layer_retention, kept_layers=kept_layers,
        hook_records=hook_records, peak_mem_bytes=mem.peak_bytes,
        cache_bytes_at_final_hop_mean=(
            cache_byte_total / cache_stat_batches if cache_stat_batches else 0.0),
        attention_positions_at_final_hop_mean=(
            position_total / cache_stat_batches if cache_stat_batches else 0.0),
        cache_stat_batches=cache_stat_batches,
        ragged_path=ragged_path,
    )
    payload = json.dumps(record, indent=2)
    if cfg.out_json:
        with open(cfg.out_json, "w") as fh:
            fh.write(payload)
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("per_item", "hop_records_first_batch")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
