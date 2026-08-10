import torch

from lkv.config import RunConfig
from lkv.relay_calib import calibrate_relay
from tests.test_latentmas_loop import _StubTok


def _cfg(**kw):
    base = dict(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                latent_steps=4, n_agents=4, greedy=True, max_new_tokens=4,
                device="cpu", generate_bs=1, calib_size=1, alpha=1.0,
                mu=0.5, sigma=10.0)
    base.update(kw)
    return RunConfig(**base)


_ITEMS = [{"question": "2+2?", "gold": "4"}, {"question": "3+3?", "gold": "6"}]


def test_returns_one_list_per_layer(tiny_qwen3):
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    assert sorted(totals) == [0, 1, 2, 3]
    assert all(len(v) == 1 for v in totals.values())
    assert all(isinstance(v[0], float) for v in totals.values())


def test_honours_calib_size(tiny_qwen3):
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(calib_size=2), _ITEMS)
    assert all(len(v) == 2 for v in totals.values())


def test_importance_is_finite_and_nonnegative(tiny_qwen3):
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    for values in totals.values():
        for v in values:
            assert v >= 0.0 and v == v and abs(v) < 1e6


def test_tracer_is_removed_afterwards(tiny_qwen3):
    calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    assert not hasattr(tiny_qwen3.model.layers[0].self_attn, "attn_inputs")


def test_later_hop_index_sees_a_longer_context(tiny_qwen3):
    """Calibrating at hop 2 accumulates two agents' output, so the sliced
    context is wider than at hop 0."""
    h0 = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS, hop_index=0)
    h2 = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS, hop_index=2)
    assert sorted(h0) == sorted(h2)
    assert h0 != h2, "hop index had no effect on calibration"


def test_ranking_is_usable_by_select_layers(tiny_qwen3):
    from lkv.kvcomm.importance import select_layers
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    layers = select_layers(totals, 0.5, alpha=1.0, mu=0.5, sigma=10.0)
    assert len(layers) == 2 and all(0 <= i < 4 for i in layers)


def test_pair_and_relay_calibration_disagree(tiny_qwen3):
    """The comparison this whole task exists to make (module docstring, and the
    task-18 brief's framing): calib_source="pair" reuses Task 14's KVComm-style
    sender/receiver calibration, calib_source="relay" measures importance at the
    actual relay boundary. On this fixture they produce DIFFERENT rankings --
    confirming the distribution-shift concern is real here, not merely
    theoretical. Had they agreed, that would equally have been worth recording
    (it would mean KVComm's ranking transfers across the shift); it is not what
    was measured, so this pins the disagreement instead of asserting an
    invariant that must hold in general.
    """
    from lkv.kvcomm.importance import layer_ranking
    from lkv.kvcomm.pair import calibrate

    pair_items = [{"instruction": "Answer using the context.",
                  "context": "Paris is the capital of France.",
                  "question": "2+2?", "gold": "4"}]

    pair_totals = calibrate(tiny_qwen3, _StubTok(), pair_items, _cfg())
    relay_totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS[:1])

    pair_ranking = layer_ranking(pair_totals, alpha=1.0, mu=0.5, sigma=10.0)
    relay_ranking = layer_ranking(relay_totals, alpha=1.0, mu=0.5, sigma=10.0)
    assert pair_ranking != relay_ranking, (
        f"pair={pair_ranking} relay={relay_ranking}: expected the KVComm-pair "
        "ranking to differ from the relay-native one on this fixture"
    )
