import copy
from importlib import import_module

import torch
from transformers import LlamaForCausalLM, Qwen3ForCausalLM

from lkv.kvcomm.tracer import install_tracer, remove_tracer
from lkv.kvcomm.trim import layer_map, prepare_key_cache, retention


def _cache(model, n_pos=12):
    ids = torch.randint(1, 200, (1, n_pos))
    with torch.no_grad():
        return model(ids, use_cache=True).past_key_values


def test_selected_layers_keep_everything_others_keep_one(tiny_qwen3):
    cache = _cache(tiny_qwen3, 12)
    out = prepare_key_cache(cache, layers_list=[2])
    lengths = [int(k.shape[-2]) for k in out.key_cache]
    # layer 0 is always retained in full; layer 2 is selected; 1 and 3 collapse
    assert lengths == [12, 1, 12, 1], lengths


def test_layer_zero_is_retained_even_when_not_selected(tiny_qwen3):
    out = prepare_key_cache(_cache(tiny_qwen3, 9), layers_list=[])
    assert int(out.key_cache[0].shape[-2]) == 9
    assert all(int(k.shape[-2]) == 1 for k in out.key_cache[1:])


def test_full_selection_is_a_faithful_copy(tiny_qwen3):
    cache = _cache(tiny_qwen3, 7)
    out = prepare_key_cache(cache, layers_list=[0, 1, 2, 3])
    for a, b in zip(cache.key_cache, out.key_cache):
        assert torch.equal(a, b)
    assert retention(out) == 1.0


def test_retention_is_positions_kept_over_positions_total(tiny_qwen3):
    out = prepare_key_cache(_cache(tiny_qwen3, 10), layers_list=[1])
    # layers 0 and 1 keep 10 each, layers 2 and 3 keep 1 each -> 22 / 40
    assert abs(retention(out) - 22 / 40) < 1e-9


def test_sink_size_is_configurable(tiny_qwen3):
    out = prepare_key_cache(_cache(tiny_qwen3, 10), layers_list=[], sink=3)
    assert [int(k.shape[-2]) for k in out.key_cache] == [10, 3, 3, 3]


def test_layer_map_is_the_upstream_formula():
    assert layer_map(4, 4) == [0, 1, 2, 3]
    # L_A=4 -> L_B=2: round((l+0.5)*2/4 - 0.5)
    assert layer_map(4, 2) == [0, 0, 1, 1]
    assert layer_map(2, 4) == [0, 2]


def test_tracer_preserves_the_attention_implementation(tiny_llama):
    before = tiny_llama.config._attn_implementation
    saved = install_tracer(tiny_llama)
    try:
        assert tiny_llama.config._attn_implementation == before == "sdpa"
        ids = torch.randint(1, 200, (1, 6))
        with torch.no_grad():
            tiny_llama(ids, use_cache=True)
        for layer in tiny_llama.model.layers:
            stash = layer.self_attn.attn_inputs
            assert set(("query", "key", "scaling")) <= set(stash)
            assert stash["query"].shape[-2] == 6
    finally:
        remove_tracer(tiny_llama, saved)
    assert not hasattr(tiny_llama.model.layers[0].self_attn, "attn_inputs")


def test_tracer_does_not_stash_on_single_token_decode(tiny_llama):
    saved = install_tracer(tiny_llama)
    try:
        with torch.no_grad():
            out = tiny_llama(torch.randint(1, 200, (1, 6)), use_cache=True)
            first = tiny_llama.model.layers[0].self_attn.attn_inputs["query"].shape[-2]
            tiny_llama(torch.randint(1, 200, (1, 1)), use_cache=True,
                       past_key_values=out.past_key_values)
            after = tiny_llama.model.layers[0].self_attn.attn_inputs["query"].shape[-2]
        assert first == after == 6, "decode step overwrote the prefill stash"
    finally:
        remove_tracer(tiny_llama, saved)


def _eager_twin(model, model_cls):
    """A same-config, same-weights copy running eager attention.

    sdpa never exposes attention weights, so this is the only trustworthy
    ground truth for "what did this forward actually attend to": construct a
    twin with attn_implementation="eager", load the same weights, and read
    `output_attentions=True`.
    """
    eager_config = copy.deepcopy(model.config)
    eager_config._attn_implementation = "eager"
    twin = model_cls(eager_config).to(torch.float32).eval()
    twin.load_state_dict(model.state_dict())
    return twin


def _assert_stash_reproduces_real_attention(model, twin, n_pos=6):
    """Recompute softmax(Q.K^T * scaling + causal_mask) from the tracer's own
    stash and check it against the attention weights the (same-weights, same
    input) eager twin actually computed. This is the numerical proof that
    query and key in the stash share a rotary basis: if they did not, this
    recomputation would not match, silently, with no crash -- exactly the
    D8-fix-round-1 bug (see docs/FINDINGS.md).
    """
    saved = install_tracer(model)
    try:
        ids = torch.randint(1, 200, (1, n_pos))
        with torch.no_grad():
            model(ids, use_cache=True)
            real_attn = twin(ids, use_cache=True, output_attentions=True).attentions

        causal = torch.zeros(n_pos, n_pos).masked_fill(
            torch.triu(torch.ones(n_pos, n_pos, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        # repeat_kv is architecture-specific in name only -- same body in both
        # modeling files -- resolved the same way tracer.py resolves RoPE.
        repeat_kv = import_module(type(model.model.layers[0].self_attn).__module__).repeat_kv

        for i, layer in enumerate(model.model.layers):
            stash = layer.self_attn.attn_inputs
            q, k, scaling = stash["query"], stash["key"], stash["scaling"]
            k_full = repeat_kv(k, layer.self_attn.num_key_value_groups)
            with torch.no_grad():
                scores = torch.matmul(q, k_full.transpose(2, 3)) * scaling + causal
                recomputed = torch.softmax(scores, dim=-1)
            torch.testing.assert_close(
                recomputed, real_attn[i], atol=1e-5, rtol=1e-4,
                msg=lambda m, i=i: f"layer {i}: {m}",
            )
    finally:
        remove_tracer(model, saved)


def test_stashed_query_key_reproduce_real_attention_weights_llama(tiny_llama):
    """D8-fix-round-1 regression: without RoPE on the stashed query, this would
    fail silently (no crash, just a wrong number) -- see the fix report for the
    captured before/after output."""
    twin = _eager_twin(tiny_llama, LlamaForCausalLM)
    _assert_stash_reproduces_real_attention(tiny_llama, twin)


def test_stashed_query_key_reproduce_real_attention_weights_qwen3(tiny_qwen3):
    """Same proof on Qwen3, which additionally applies q_norm before RoPE --
    covers the interaction the Llama case can't, since Llama has no q_norm."""
    twin = _eager_twin(tiny_qwen3, Qwen3ForCausalLM)
    _assert_stash_reproduces_real_attention(tiny_qwen3, twin)
