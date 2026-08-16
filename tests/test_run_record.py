import json

import pytest
import torch
from transformers import DynamicCache

from lkv.config import RunConfig
from lkv.run import build_record, main


def _cfg():
    return RunConfig(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                     latent_steps=40, greedy=True, device="cpu")


def test_record_has_per_item_and_both_metrics():
    per_item = [{"i": 0, "correct": True, "score": 1.0},
                {"i": 1, "correct": False, "score": 0.25}]
    rec = build_record(_cfg(), items=[{}, {}], predictions=["a", "b"],
                       per_item=per_item, hops=[], timings={"total_sec": 2.0})
    assert rec["n"] == 2
    assert rec["accuracy"] == 0.5
    assert rec["mean_score"] == 0.625
    assert rec["per_item"] == per_item
    assert rec["sec_per_item"] == 1.0
    assert rec["latent_steps"] == 40 and rec["greedy"] is True
    assert rec["ragged_path"] == "n/a", "default must be n/a, not silently absent"


def test_ragged_path_records_stock_noncausal_when_the_caller_says_it_trimmed():
    """build_record itself is a dumb reporter for this field -- main() decides
    what to pass. This just pins that build_record actually writes the value
    through rather than always defaulting to 'n/a'."""
    per_item = [{"i": 0, "correct": True, "score": 1.0}]
    rec = build_record(_cfg(), items=[{}], predictions=["a"], per_item=per_item,
                       hops=[], timings={"total_sec": 1.0}, ragged_path="stock-noncausal")
    assert rec["ragged_path"] == "stock-noncausal"


def test_record_refuses_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        build_record(_cfg(), items=[{}, {}], predictions=["a"],
                     per_item=[{"i": 0, "correct": True, "score": 1.0}],
                     hops=[], timings={"total_sec": 1.0})


def test_record_excludes_extra_dict():
    """cfg.extra is a scratch field for ad-hoc knobs; it must not leak into
    the on-disk record, which is meant to be a flat, diffable summary."""
    cfg = _cfg()
    cfg.extra = {"scratch": "should not appear"}
    per_item = [{"i": 0, "correct": True, "score": 1.0}]
    rec = build_record(cfg, items=[{}], predictions=["a"], per_item=per_item,
                       hops=[], timings={"total_sec": 1.0})
    assert "extra" not in rec


def test_main_keeps_per_item_aligned_across_batch_boundaries(monkeypatch, tmp_path):
    """`i = offset + j` in main()'s loop body is the only line that maps a
    per-batch position back to a run-level index; the brief itself names it
    as the line where an error would silently misalign every paired
    comparison in the project.

    Checking `per_item[i]["i"] == i` alone would NOT catch a broken `i` --
    that field is written as `"i": i` at the very slot indexed by that same
    `i`, so it is self-consistent by construction no matter what `i` is.
    Instead each item is given its own designed correctness outcome (True
    for even gold values, False for odd) computed only from the item's own
    content (batch[j]["gold"]), never from a batch position or offset. If
    the storage index is wrong, an item's designed outcome lands in the
    wrong slot -- or a slot is left as the initial `None` and crashes -- and
    this test catches it either way.

    generate_bs=2 over 5 items yields 3 batches (sizes 2, 2, 1), so the loop
    crosses two batch boundaries.
    """
    n = 5
    items = [{"gold": str(k)} for k in range(n)]

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        return object(), object()

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None,
                           **kwargs):
        texts = []
        for item in batch:
            k = int(item["gold"])
            # Correct iff k is even; wrong on purpose otherwise, so the
            # per-item outcome is a real assertion, not a tautology.
            texts.append(f"\\boxed{{{k}}}" if k % 2 == 0 else "\\boxed{999}")
        return texts, []

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.run_latentmas", fake_run_latentmas)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--generate_bs", "2", "--device", "cpu",
            "--greedy", "--out_json", str(out)]

    assert main(argv) == 0

    per_item = json.loads(out.read_text())["per_item"]
    assert len(per_item) == n
    for k in range(n):
        assert per_item[k]["i"] == k
        assert per_item[k]["correct"] == (k % 2 == 0), (
            f"item {k}'s own designed outcome did not land at position {k} -- "
            "batch offset arithmetic misaligned per_item"
        )


def test_per_item_stores_the_prediction_and_the_gold(monkeypatch, tmp_path):
    """Task 0. Six Countries conditions scored exactly 0.0000 including
    skyline -- which is handed the full context AND the question, so zero is
    impossible unless the harness is at fault. That could not be diagnosed,
    because a record stores only whether each item was correct, never what the
    model emitted: `predictions` reaches build_record and is used solely for a
    length cross-check. A run that scores zero has to be able to say what it
    saw, or the next zero costs another GPU session and another blind guess.
    """
    items = [{"gold": "0"}, {"gold": "1"}]

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        return object(), object()

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None,
                           **kwargs):
        return [f"the answer is \\boxed{{{item['gold']}}}" for item in batch], []

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.run_latentmas", fake_run_latentmas)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--generate_bs", "1", "--device", "cpu", "--greedy",
            "--out_json", str(out)]

    assert main(argv) == 0
    per_item = json.loads(out.read_text())["per_item"]

    for k, entry in enumerate(per_item):
        assert entry["prediction"] == f"the answer is \\boxed{{{k}}}", (
            "per_item must carry the text the model actually produced"
        )
        assert entry["gold"] == str(k), "per_item must carry the gold it was scored against"
        assert entry["prediction_chars"] == len(entry["prediction"])


def test_a_long_prediction_is_elided_from_the_middle_keeping_both_ends(monkeypatch, tmp_path):
    """Storing generations verbatim would put megabytes into a tracked runs/
    record -- MBPP+ alone is 378 items at max_new_tokens=4096. Storing only a
    head would be worse than useless for GSM8K and MBPP+, whose answers are
    extracted from the END of the generation (`numbers[-1]`, the LAST ```python
    block); storing only a tail loses the preamble that Task 0's leading
    hypothesis is about. So the middle is dropped and both ends kept, and the
    UNELIDED length is recorded separately -- that number is what distinguishes
    "the model emitted nothing" from "the model ran into max_new_tokens".
    """
    body = "x" * 5000
    items = [{"gold": "0"}]

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        return object(), object()

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None,
                           **kwargs):
        return ["HEAD" + body + "\\boxed{0}" for _ in batch], []

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.run_latentmas", fake_run_latentmas)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--generate_bs", "1", "--device", "cpu", "--greedy",
            "--out_json", str(out)]

    assert main(argv) == 0
    entry = json.loads(out.read_text())["per_item"][0]

    full = "HEAD" + body + "\\boxed{0}"
    assert entry["prediction_chars"] == len(full), (
        "the recorded length must be the TRUE length, not the stored length"
    )
    assert len(entry["prediction"]) < len(full), "a 5000-char generation must not be stored whole"
    assert entry["prediction"].startswith("HEAD"), "the head must survive"
    assert entry["prediction"].endswith("\\boxed{0}"), (
        "the tail must survive -- gsm8k/mbppplus answers are extracted from the end"
    )
    assert "elided" in entry["prediction"], "an elision must be visibly marked, not silent"
    # The item is still scored on the FULL text, never on the stored excerpt.
    assert entry["correct"] is True


def test_projector_is_built_once_and_shared_across_batches(monkeypatch, tmp_path):
    """Finding 2(a) regression.

    run_latentmas used to be called without projector=, so
    lkv/latentmas.py built a FRESH LatentProjector every batch -- and a fresh
    projector's _ensure() rebuilds the realign matrix (or the target-norm-only
    path) on its own first use, once per batch rather than once for the whole
    run. At Qwen3-8B the full build is a multi-GB transient (two fp32
    embedding copies plus a 4096x4096 solve); paying it every batch inflates
    peak_mem_bytes and sec_per_item identically across every latentmas
    condition, swamping the differential those metrics exist to show. main()
    must build the projector ONCE, before the batch loop, and pass that SAME
    object into every run_latentmas call.
    """
    items = [{"gold": "0"}, {"gold": "0"}, {"gold": "0"}]  # generate_bs=1 -> 3 batches
    seen_projectors = []

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        return object(), object()

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None,
                           **kwargs):
        seen_projectors.append(projector)
        return ["\\boxed{0}" for _ in batch], []

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.run_latentmas", fake_run_latentmas)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--generate_bs", "1", "--device", "cpu", "--greedy",
            "--out_json", str(out)]

    assert main(argv) == 0
    assert len(seen_projectors) == 3
    assert all(p is not None for p in seen_projectors), "run_latentmas must receive a real projector"
    assert seen_projectors[0] is seen_projectors[1] is seen_projectors[2], (
        "main() must build ONE projector before the batch loop and reuse it "
        "across every batch, not build a fresh one per batch"
    )


def test_main_kvcomm_calibrates_once_and_reuses_model_b(monkeypatch, tmp_path):
    """main()'s NEW kvcomm wiring (Task 14), not exercised by tests/test_pair.py
    since those call calibrate/run_pair directly rather than through main().

    Three things that a bug here would get wrong silently (each would still
    produce SOME JSON, just a scientifically wrong one -- exactly the kind of
    mistake Task 15/16 would only discover after a real GPU run):
    1. calibrate() must run once total, not once per batch (5 items,
       generate_bs=2 -> 3 batches).
    2. model_b must be REUSED (not reloaded) when cfg.model_b == cfg.model_name,
       the default -- reloading it would double GPU memory in the common case.
    3. kept_layers/layer_retention in the record must be exactly what
       select_layers computes from calibrate's own totals, not a placeholder.
    """
    import types

    items = [{"gold": str(k)} for k in range(5)]
    calibrate_calls = []
    load_model_calls = []
    totals = {i: [1.0] for i in range(4)}

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        load_model_calls.append(model_name)
        model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=4))
        return model, object()

    def fake_calibrate(model, tokenizer, items_, cfg):
        calibrate_calls.append(len(items_))
        return totals

    def fake_run_pair(model_a, model_b, tokenizer, cfg, batch, layers_list):
        assert model_a is model_b, "model_b was reloaded instead of reused"
        assert layers_list == kept_layers_expected
        return ["\\boxed{4}" for _ in batch]

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.calibrate", fake_calibrate)
    monkeypatch.setattr("lkv.run.run_pair", fake_run_pair)

    from lkv.kvcomm.importance import select_layers
    kept_layers_expected = select_layers(totals, top_layers=0.5, alpha=1.0,
                                         mu=0.5, sigma=10.0)

    out = tmp_path / "out.json"
    argv = ["--system", "kvcomm", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--generate_bs", "2", "--device", "cpu", "--greedy",
            "--top_layers", "0.5", "--alpha", "1.0", "--out_json", str(out)]

    assert main(argv) == 0
    assert calibrate_calls == [5], "calibrate must run once, over all items, not per batch"
    assert load_model_calls == ["Qwen/Qwen3-8B"], "model_b must be reused, not reloaded"

    record = json.loads(out.read_text())
    assert record["kept_layers"] == kept_layers_expected
    assert record["layer_retention"] == len(kept_layers_expected) / 4


def _mk_cache(n_pos):
    """A 1-layer DynamicCache with n_pos positions -- cheap and deterministic,
    so cache_bytes/attention_positions on it are exact arithmetic rather than
    something read off a real model."""
    cache = DynamicCache()
    cache.update(torch.zeros(1, 1, n_pos, 1), torch.zeros(1, 1, n_pos, 1), 0)
    return cache


def test_main_aggregates_cache_stats_across_batches_not_just_the_first(monkeypatch, tmp_path):
    """Task 19's central correctness point, exercised end-to-end through main().

    Two things a bug here would get wrong SILENTLY (both would still produce
    valid JSON, just a scientifically wrong number):
    1. Only the LAST trim_hook call in a batch is that batch's "final hop" --
       an earlier, non-final call (2 positions, below) must not leak into the
       figure. This is exactly the run_latentmas contract Task 19 leans on to
       reach the cache without widening its return signature.
    2. The mean must be built across BOTH batches' final-hop values (5 and 9
       positions, deliberately different -- real batches differ because
       prompt lengths differ), not "whichever batch happened to run first".
       Task 9 already caught this exact mistake once for hop_records
       (renamed hop_records_first_batch as a result); this is its analogue
       for the new _mean fields.

    Also covers the reference condition: top_layers defaults to 1.0, so
    kept_layers stays None and main() never builds a real trim hook -- yet
    the measuring wrapper must still run, since an untrimmed run is the
    baseline every other run's ratio divides by.
    """
    items = [{"gold": "0"}, {"gold": "4"}]

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        return object(), object()

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None,
                           **kwargs):
        final_n_pos = 5 + int(batch[0]["gold"])  # 5 for batch 1, 9 for batch 2
        if trim_hook is not None:
            trim_hook(_mk_cache(2), 0, 0)             # earlier hop -- must not count
            trim_hook(_mk_cache(final_n_pos), 1, 2)    # this batch's final hop
        return ["\\boxed{0}" for _ in batch], []

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.run_latentmas", fake_run_latentmas)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--generate_bs", "1", "--device", "cpu",
            "--greedy", "--out_json", str(out)]

    assert main(argv) == 0
    record = json.loads(out.read_text())

    assert record["kept_layers"] is None, "top_layers=1.0 must never select layers"
    assert record["cache_stat_batches"] == 2
    assert record["attention_positions_at_final_hop_mean"] == (5 + 9) / 2
    # 1 layer, K+V, float32 (4 bytes/elem), numel == n_pos -> 8 bytes/position.
    assert record["cache_bytes_at_final_hop_mean"] == 8 * (5 + 9) / 2
    assert record["peak_mem_bytes"] == 0, "PeakMemory must be a safe no-op on cpu"
    # Finding 4: a latentmas run that built no trim hook (top_layers==1.0, the
    # reference every other condition's ratio divides by) is full retention,
    # a known fact, not an unknown -- the analyzer's `d.get("layer_retention",
    # 1.0)` would TypeError on a literal None here, since .get's default only
    # covers a MISSING key and this key is always present.
    assert record["layer_retention"] == 1.0
    assert record["ragged_path"] == "n/a", "no kept_layers means nothing was ever trimmed"


def test_layer_retention_is_the_all_batch_mean_not_the_first_batch(monkeypatch, tmp_path):
    """Finding 1 regression.

    `hook_records` is (deliberately, per its own docstring) captured from the
    FIRST batch only -- cache lengths differ per batch since prompt lengths
    differ. `layer_retention` used to be computed as
    `sum(r["retention"] for r in hook_records) / len(hook_records)`, i.e. the
    mean of that SAME first-batch-only list wearing a run-level name. This is
    the sibling fix to hop_records -> hop_records_first_batch (Task 9): here
    the run-level figure must be an accumulated sum/count across EVERY
    batch's own hook, not a read of the first batch's inspection copy.

    Two batches see genuinely different retention by construction: the hook
    trims a cache to `sink` positions per unselected layer, so varying the
    untrimmed length the hook is handed (10 vs 40) changes what fraction
    `sink` represents, without changing which layers are selected.
    """
    import types

    from lkv.kvcomm.importance import select_layers

    n_layers = 4
    totals = {i: [1.0] for i in range(n_layers)}
    kept_layers_expected = select_layers(totals, top_layers=0.5, alpha=1.0,
                                         mu=0.5, sigma=10.0)
    kept = set(kept_layers_expected) | {0}  # layer 0 is always retained too

    def _expected_retention(full_len):
        lengths = [full_len if i in kept else 1 for i in range(n_layers)]
        return sum(lengths) / (max(lengths) * n_layers)

    ret_batch1, ret_batch2 = _expected_retention(10), _expected_retention(40)
    assert ret_batch1 != ret_batch2, "test setup must give genuinely different per-batch retentions"

    items = [{"gold": "0"}, {"gold": "0"}]  # generate_bs=1 -> 2 batches
    full_lens = [10, 40]

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=n_layers))
        return model, object()

    def fake_calibrate_relay(model, tokenizer, cfg, items_):
        return totals

    def _mk_relay_cache(full_len):
        cache = DynamicCache()
        for i in range(n_layers):
            cache.update(torch.zeros(1, 1, full_len, 1), torch.zeros(1, 1, full_len, 1), i)
        return cache

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None,
                           **kwargs):
        full_len = full_lens.pop(0)
        if trim_hook is not None:
            trim_hook(_mk_relay_cache(full_len), 0, 0)
        return ["\\boxed{0}" for _ in batch], []

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)
    monkeypatch.setattr("lkv.run.calibrate_relay", fake_calibrate_relay)
    monkeypatch.setattr("lkv.run.run_latentmas", fake_run_latentmas)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--generate_bs", "1", "--device", "cpu", "--greedy",
            "--top_layers", "0.5", "--alpha", "1.0", "--calib_size", "1",
            "--out_json", str(out)]

    assert main(argv) == 0
    record = json.loads(out.read_text())

    assert record["kept_layers"] == kept_layers_expected
    all_batch_mean = (ret_batch1 + ret_batch2) / 2
    assert record["layer_retention"] == pytest.approx(all_batch_mean)
    assert record["layer_retention"] != pytest.approx(ret_batch1), (
        "layer_retention must not silently equal only the first batch's retention"
    )
    assert record["ragged_path"] == "stock-noncausal"
