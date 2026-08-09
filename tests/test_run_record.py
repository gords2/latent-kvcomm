from lkv.config import RunConfig
from lkv.run import build_record


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
