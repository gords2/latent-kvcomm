"""Bit-exactness against upstream. The reference block below is copied verbatim
from stock-kvcomm/layer_importance.py -- do not tidy it."""
import copy

import numpy as np
import torch
from transformers import LlamaForCausalLM

from lkv.kvcomm.importance import layer_importance, layer_ranking, select_layers


# ---- verbatim upstream reference ----
def _ref_normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32)
    mn, mx = v.min(), v.max()
    if mx - mn < eps:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn + eps)


def _ref_gaussian_prior(L, mu, sigma=10.0):
    x = np.arange(L, dtype=np.float32)
    return np.exp(-0.5 * ((x - mu) / float(sigma)) ** 2)


def _ref_calc(B_attn_weights: dict):
    out = {}
    for i in B_attn_weights:
        aw = B_attn_weights[i]
        attn_matrix = aw[..., 1:-aw.shape[-2]]
        importance = attn_matrix.sum(dim=-1).mean()
        out[i] = 0 if torch.isnan(importance) else importance.item()
    return out


def _ref_ranking(totals: dict, alpha, mu_frac, sigma):
    importance = []
    n = len(totals)
    for i in range(n):
        importance.append(np.mean(totals[i]))
    importance = _ref_normalize(importance)
    mu = mu_frac * (n - 1)
    g = _ref_normalize(_ref_gaussian_prior(n, mu=mu, sigma=sigma))
    importance = alpha * importance + (1.0 - alpha) * g
    return np.argsort(importance)[::-1]
# ---- end reference ----


def _fake_attn(n_layers=6, batch=2, heads=4, q_len=3, ctx=10, seed=0):
    torch.manual_seed(seed)
    out = {}
    for i in range(n_layers):
        logits = torch.randn(batch, heads, q_len, ctx + q_len)
        out[i] = torch.softmax(logits, dim=-1)
    return out


def test_layer_importance_matches_upstream_exactly():
    attn = _fake_attn()
    ours = layer_importance(attn, q_len=3)
    theirs = _ref_calc(attn)
    for i in theirs:
        assert abs(ours[i] - theirs[i]) < 1e-12, (i, ours[i], theirs[i])


def test_importance_excludes_position_zero_and_receiver_block():
    """Sliced range is 1 .. ctx-1, so ctx-1 columns out of ctx+q."""
    attn = _fake_attn(n_layers=1, batch=1, heads=1, q_len=3, ctx=10)
    manual = attn[0][..., 1:-3].sum(dim=-1).mean().item()
    assert abs(layer_importance(attn, q_len=3)[0] - manual) < 1e-12


def test_nan_importance_becomes_zero():
    attn = {0: torch.full((1, 1, 2, 6), float("nan"))}
    assert layer_importance(attn, q_len=2)[0] == 0


def test_ranking_matches_upstream_for_several_alphas():
    totals = {i: [float(v)] for i, v in enumerate([0.1, 0.9, 0.4, 0.7, 0.2, 0.55])}
    for alpha in (1.0, 0.8, 0.5, 0.0):
        ours = layer_ranking(totals, alpha=alpha, mu=0.5, sigma=10.0)
        theirs = list(_ref_ranking(totals, alpha, 0.5, 10.0))
        assert ours == theirs, (alpha, ours, theirs)


def test_alpha_one_ignores_the_gaussian():
    totals = {i: [float(v)] for i, v in enumerate([0.1, 0.9, 0.4, 0.7])}
    a = layer_ranking(totals, alpha=1.0, mu=0.5, sigma=10.0)
    b = layer_ranking(totals, alpha=1.0, mu=0.1, sigma=1.0)
    assert a == b, "mu/sigma changed the ranking at alpha=1.0"


def test_alpha_zero_is_pure_gaussian_peaking_mid_depth():
    totals = {i: [0.0] for i in range(11)}   # flat importance
    r = layer_ranking(totals, alpha=0.0, mu=0.5, sigma=2.0)
    assert r[0] == 5, r


def test_select_layers_count_is_floored():
    totals = {i: [float(i)] for i in range(32)}
    assert len(select_layers(totals, 0.3, 1.0, 0.5, 10.0)) == 9    # int(9.6)
    assert len(select_layers(totals, 0.5, 1.0, 0.5, 10.0)) == 16
    assert len(select_layers(totals, 0.7, 1.0, 0.5, 10.0)) == 22   # int(22.4)


def test_select_layers_takes_the_highest_scoring():
    totals = {i: [float(i)] for i in range(10)}
    assert sorted(select_layers(totals, 0.3, 1.0, 0.5, 10.0)) == [7, 8, 9]


def test_random_selection_is_seeded_and_same_size():
    totals = {i: [float(i)] for i in range(32)}
    a = select_layers(totals, 0.5, 1.0, 0.5, 10.0, random_selection=True, seed=7)
    b = select_layers(totals, 0.5, 1.0, 0.5, 10.0, random_selection=True, seed=7)
    c = select_layers(totals, 0.5, 1.0, 0.5, 10.0, random_selection=True, seed=8)
    assert a == b and len(a) == 16 and a != c


def test_end_to_end_ranking_from_a_traced_forward(tiny_llama):
    """The tracer must produce stashes that importance can consume."""
    from lkv.kvcomm.importance import attention_from_stash
    from lkv.kvcomm.tracer import install_tracer, remove_tracer

    saved = install_tracer(tiny_llama)
    try:
        with torch.no_grad():
            tiny_llama(torch.randint(1, 200, (1, 8)), use_cache=True)
        attn = {}
        for i, layer in enumerate(tiny_llama.model.layers):
            attn[i] = attention_from_stash(layer.self_attn,
                                           layer.self_attn.attn_inputs, q_len_receiver=8)
        assert set(attn) == {0, 1, 2, 3}
        for w in attn.values():
            assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4)
    finally:
        remove_tracer(tiny_llama, saved)


def test_attention_from_stash_causal_mask_matches_real_forward_with_cached_prefix(tiny_llama):
    """Not in the brief -- added on self-review.

    The brief's own end-to-end test only checks that rows sum to 1, which a
    *wrong* offset also satisfies (softmax over any contiguous causal block
    still sums to 1). This exercises attention_from_stash with a genuine
    non-zero cached prefix -- context prefilled first, receiver prefilled
    second against that cache -- so kv_len - q_len > 0, and compares against
    an eager-mode twin's real output_attentions=True weights for the same
    tokens computed in a single full-sequence forward. A tril(0) mask instead
    of tril(offset) still produces rows that sum to 1 but is wrong on 73% of
    entries (verified by hand while writing this test); this is the check
    that would catch that class of bug.
    """
    from lkv.kvcomm.importance import attention_from_stash
    from lkv.kvcomm.tracer import install_tracer, remove_tracer

    context_len, receiver_len = 5, 8
    torch.manual_seed(3)
    context_ids = torch.randint(1, 200, (1, context_len))
    receiver_ids = torch.randint(1, 200, (1, receiver_len))

    eager_config = copy.deepcopy(tiny_llama.config)
    eager_config._attn_implementation = "eager"
    twin = LlamaForCausalLM(eager_config).to(torch.float32).eval()
    twin.load_state_dict(tiny_llama.state_dict())

    full_ids = torch.cat([context_ids, receiver_ids], dim=-1)
    with torch.no_grad():
        real_attn = twin(full_ids, use_cache=True, output_attentions=True).attentions

    saved = install_tracer(tiny_llama)
    try:
        with torch.no_grad():
            prefix = tiny_llama(context_ids, use_cache=True)
            tiny_llama(receiver_ids, use_cache=True, past_key_values=prefix.past_key_values)
        for i, layer in enumerate(tiny_llama.model.layers):
            stash = layer.self_attn.attn_inputs
            assert stash["key"].shape[-2] == context_len + receiver_len
            assert stash["query"].shape[-2] == receiver_len
            recomputed = attention_from_stash(layer.self_attn, stash,
                                              q_len_receiver=receiver_len)
            expected = real_attn[i][:, :, context_len:, :]
            torch.testing.assert_close(
                recomputed, expected, atol=1e-5, rtol=1e-4,
                msg=lambda m, i=i: f"layer {i}: {m}",
            )
    finally:
        remove_tracer(tiny_llama, saved)
