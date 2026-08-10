import json

import torch

from lkv.config import RunConfig
from lkv.integrate import make_trim_hook, trim_delta
from lkv.kvcomm.trim import retention
from lkv.latentmas import past_length, run_latentmas
from tests.test_latentmas_loop import _StubTok


def _cache(model, n_pos):
    with torch.no_grad():
        return model(torch.randint(1, 200, (1, n_pos)), use_cache=True).past_key_values


def test_trim_delta_keeps_the_inherited_prefix(tiny_qwen3):
    cache = _cache(tiny_qwen3, 20)
    out = trim_delta(cache, layers_list=[2], inherited_len=12, sink=1)
    lengths = [int(k.shape[-2]) for k in out.key_cache]
    # layer 0 always full, layer 2 selected -> 20; layers 1 and 3 -> 12 + 1
    assert lengths == [20, 13, 20, 13], lengths


def test_trim_delta_at_first_hop_equals_scope_all(tiny_qwen3):
    from lkv.kvcomm.trim import prepare_key_cache
    cache = _cache(tiny_qwen3, 10)
    delta = trim_delta(cache, [2], inherited_len=0, sink=1)
    allsc = prepare_key_cache(cache, [2], sink=1)
    assert [int(k.shape[-2]) for k in delta.key_cache] == \
           [int(k.shape[-2]) for k in allsc.key_cache]


def test_scope_all_is_more_aggressive_than_delta(tiny_qwen3):
    cache = _cache(tiny_qwen3, 20)
    h_all = make_trim_hook([2], scope="all")
    h_delta = make_trim_hook([2], scope="delta")
    r_all = retention(h_all(cache, hop_index=1, inherited_len=12))
    r_delta = retention(h_delta(cache, hop_index=1, inherited_len=12))
    assert r_all < r_delta


def test_hook_records_one_entry_per_call(tiny_qwen3):
    hook = make_trim_hook([2], scope="all")
    cache = _cache(tiny_qwen3, 12)
    hook(cache, hop_index=0, inherited_len=0)
    hook(cache, hop_index=1, inherited_len=6)
    assert [r["hop"] for r in hook.records] == [0, 1]
    assert all(0.0 < r["retention"] <= 1.0 for r in hook.records)
    assert all(r["kept_layers"] == [2] for r in hook.records)


def test_every_hop_false_trims_only_the_first_boundary(tiny_qwen3):
    hook = make_trim_hook([2], scope="all", every_hop=False)
    cache = _cache(tiny_qwen3, 12)
    first = hook(cache, hop_index=0, inherited_len=0)
    second = hook(cache, hop_index=1, inherited_len=6)
    assert retention(first) < 1.0
    assert retention(second) == 1.0, "every_hop=False must pass hop 1 through"


def test_top_layers_one_is_a_no_op(tiny_qwen3):
    """Guard against silently trimming at full retention."""
    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    hook = make_trim_hook(all_layers, scope="all")
    cache = _cache(tiny_qwen3, 12)
    assert retention(hook(cache, hop_index=0, inherited_len=0)) == 1.0


def test_relay_with_a_hook_shortens_the_cache(tiny_qwen3):
    cfg = RunConfig(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                    latent_steps=4, n_agents=4, greedy=True, max_new_tokens=4,
                    device="cpu", generate_bs=1, top_layers=0.5,
                    trim_scope="all")
    items = [{"question": "2+2?", "gold": "4"}]
    hook = make_trim_hook([2], scope="all", every_hop=True)
    _, hops = run_latentmas(tiny_qwen3, _StubTok(), cfg, items, trim_hook=hook)
    trimmed_hops = [h for h in hops if "positions_after_trim" in h]
    assert trimmed_hops, hops
    assert len(hook.records) == 3


def test_every_hop_true_keeps_the_running_mask_consistent_with_the_cache(tiny_qwen3):
    """Beyond the brief: the correctness property that makes trimming safe to
    thread into the relay at all.

    run_latentmas re-cuts its running attention mask to
    `run_mask[:, :past_length(cache)]` immediately after every trim_hook call
    (lkv/latentmas.py). That re-cut is silently correct ONLY because both
    scopes keep layer 0 (and every selected layer) at full length -- so
    past_length(cache), the MAX over all layers, is provably unchanged by a
    trim, and the "re-cut" is a no-op. If a hook ever collapsed layer 0 too
    (a regression in trim_delta/prepare_key_cache, or a future recalibrate
    that forgets the rule), this re-cut would silently truncate run_mask
    below the length the untouched layers still hold, desynchronising
    attention_mask from cache_position on the very next hop -- exactly the
    D12-class bug this codebase has hit before, just one layer deeper.

    Runs the full four-agent relay with every_hop=True -- the compounding
    case, where all three boundaries trim, so a one-off desync would surface
    at the second or third boundary rather than being masked by only ever
    trimming once -- and checks at EVERY trim_hook call that past_length is
    unchanged by the trim, then cross-checks that against run_latentmas's own
    independently-recorded `positions_after_trim` bookkeeping.
    """
    before_after: list[tuple[int, int]] = []
    inner = make_trim_hook([2], scope="all", every_hop=True)

    def spy(cache, hop_index, inherited_len):
        pre = past_length(cache)
        out = inner(cache, hop_index, inherited_len)
        before_after.append((pre, past_length(out)))
        return out

    cfg = RunConfig(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                    latent_steps=3, n_agents=4, greedy=True, max_new_tokens=4,
                    device="cpu", generate_bs=1, top_layers=0.5,
                    trim_scope="all", trim_every_hop=True)
    items = [{"question": "2+2?", "gold": "4"}]
    _, hops = run_latentmas(tiny_qwen3, _StubTok(), cfg, items, trim_hook=spy)

    assert len(before_after) == 3, "expected one trim_hook call per boundary"
    for hop_index, (pre, post) in enumerate(before_after):
        assert pre == post, (
            f"hop {hop_index}: past_length changed from {pre} to {post} across "
            "the trim -- run_latentmas's run_mask re-cut would desynchronise "
            "attention_mask from cache_position on the next hop"
        )

    # Cross-check against run_latentmas's OWN bookkeeping, not just the spy's:
    # positions_after_trim is past_length read by the loop itself right after
    # the hook returns, so it must agree with what the spy independently saw.
    latent_hops = [h for h in hops if "positions_after_trim" in h]
    assert [h["positions_after_trim"] for h in latent_hops] == \
           [post for _, post in before_after]


def test_main_wires_latentmas_trim_hook_and_surfaces_hook_records(monkeypatch, tmp_path,
                                                                   tiny_qwen3):
    """main()'s Task 17 wiring: system=latentmas + top_layers<1.0 must calibrate,
    select layers, build a trim hook, thread it into run_latentmas, and surface
    the hook's own records (plus a mean retention) in the on-disk run record --
    none of the tests above exercise this, since they call run_latentmas
    directly and bypass main() entirely.

    calibrate() (lkv/kvcomm/pair.py) needs "instruction"/"context"/"question"
    fields -- the KVComm sender/receiver split -- so these items carry all
    three even though run_latentmas itself only ever reads "question" (see
    lkv/latentmas.py's _render). Real gsm8k/mbppplus items have no
    instruction/context at all, so --system latentmas --task gsm8k
    --top_layers <1.0 would raise KeyError inside calibrate() for that reason
    -- a pre-existing coupling in calibrate(), not something this test papers
    over; see the task report's concerns section.
    """
    from lkv.run import main

    items = [{"instruction": "Answer using the context.",
             "context": "Paris is the capital of France.",
             "question": "2+2?", "gold": "4"}]

    def fake_load_task(task, max_samples, seed):
        return items

    def fake_load_model(model_name, device):
        return tiny_qwen3, _StubTok()

    monkeypatch.setattr("lkv.run.load_task", fake_load_task)
    monkeypatch.setattr("lkv.run.load_model", fake_load_model)

    out = tmp_path / "out.json"
    argv = ["--system", "latentmas", "--task", "gsm8k", "--model_name", "Qwen/Qwen3-8B",
            "--latent_steps", "1", "--n_agents", "4", "--generate_bs", "1",
            "--device", "cpu", "--greedy", "--top_layers", "0.5", "--alpha", "1.0",
            "--calib_size", "1", "--trim_scope", "all", "--out_json", str(out)]

    assert main(argv) == 0
    record = json.loads(out.read_text())

    n_layers = tiny_qwen3.config.num_hidden_layers
    assert record["kept_layers"], "top_layers=0.5 must select a non-empty, strict subset"
    assert len(record["kept_layers"]) < n_layers

    hook_records = record["hook_records"]
    assert len(hook_records) == 3, "4 agents -> 3 boundaries, one hook call each"
    assert [r["hop"] for r in hook_records] == [0, 1, 2]

    mean_retention = sum(r["retention"] for r in hook_records) / len(hook_records)
    assert record["layer_retention"] == mean_retention
