import json

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

    def fake_run_latentmas(model, tokenizer, cfg, batch, trim_hook=None, projector=None):
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
