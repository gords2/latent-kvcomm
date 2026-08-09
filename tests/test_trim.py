import torch

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
