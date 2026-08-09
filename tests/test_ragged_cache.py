"""D8/D9: what does stock sdpa actually do with a ragged DynamicCache?

This test MEASURES the behaviour rather than asserting a belief about it. The
first test records the answer; the second asserts our own fix is causal.
"""
import torch

from lkv.kvcomm.trim import build_per_layer_masks, prepare_key_cache


def _prefill(model, n_pos):
    ids = torch.randint(1, 200, (1, n_pos))
    with torch.no_grad():
        return model(ids, use_cache=True).past_key_values


def test_ragged_cache_forward_does_not_crash(tiny_qwen3):
    """Establishes that the ragged path runs at all -- KVComm relies on it."""
    cache = prepare_key_cache(_prefill(tiny_qwen3, 12), layers_list=[2])
    q = torch.randint(1, 200, (1, 4))
    mask = torch.ones(1, 12 + 4, dtype=torch.long)
    with torch.no_grad():
        out = tiny_qwen3(q, attention_mask=mask, past_key_values=cache, use_cache=True)
    assert out.logits.shape == (1, 4, tiny_qwen3.config.vocab_size)


def _causality_drift(model, cache_builder):
    """Run 4 receiver tokens twice, changing ONLY the last one. If the FIRST
    position's logits move, information flowed backwards."""
    outs = []
    for q in (torch.tensor([[11, 12, 13, 14]]), torch.tensor([[11, 12, 13, 99]])):
        cache = cache_builder()
        lens = [int(k.shape[-2]) for k in cache.key_cache]
        mask = torch.ones(1, max(lens) + 4, dtype=torch.long)
        with torch.no_grad():
            outs.append(model(q, attention_mask=mask, past_key_values=cache,
                              use_cache=True).logits)
    return (outs[0][0, 0] - outs[1][0, 0]).abs().max().item()


def test_untrimmed_cache_is_causal(tiny_qwen3):
    """Control. Establishes the measurement itself reads zero when it should."""
    base = _prefill(tiny_qwen3, 12)
    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    drift = _causality_drift(tiny_qwen3, lambda: prepare_key_cache(base, all_layers))
    print(f"D8 MEASUREMENT: untrimmed control drift = {drift:.3e}")
    assert drift == 0.0, f"control drifted: {drift:.3e}"


def test_stock_path_is_NOT_causal_when_trimmed(tiny_qwen3):
    """D8, already confirmed by measurement before implementation began:
    cache lengths [12, 1, 12, 1] gave a first-position drift of 1.940e-01.

    This test pins the DEFECT so that a future transformers upgrade silently
    fixing it is visible rather than invisible. If it starts failing, stock
    sdpa began handling ragged caches causally -- update docs/FINDINGS.md and
    re-evaluate whether forward_with_ragged_cache is still needed.
    """
    base = _prefill(tiny_qwen3, 12)
    drift = _causality_drift(tiny_qwen3, lambda: prepare_key_cache(base, [2]))
    print(f"D8 MEASUREMENT: trimmed [12,1,12,1] stock-path drift = {drift:.3e}")
    assert drift > 1e-3, (
        f"stock ragged path now appears causal (drift={drift:.3e}); D8 may be "
        "fixed upstream -- see docs/FINDINGS.md"
    )


def test_per_layer_masks_are_causal_and_correctly_sized(tiny_qwen3):
    cache = prepare_key_cache(_prefill(tiny_qwen3, 12), layers_list=[2])
    masks = build_per_layer_masks(cache, q_len=4, device="cpu", dtype=torch.float32)
    assert len(masks) == 4
    for i, m in enumerate(masks):
        kv_len = int(cache.key_cache[i].shape[-2]) + 4
        assert m.shape[-1] == kv_len, (i, m.shape, kv_len)
        assert m.shape[-2] == 4
        # the receiver block is the last 4 columns; it must be lower-triangular
        block = m[0, 0, :, -4:]
        for r in range(4):
            for c in range(r + 1, 4):
                assert torch.isinf(block[r, c]) and block[r, c] < 0, (i, r, c)
            assert block[r, r] == 0.0


def test_fixed_forward_is_causal(tiny_qwen3):
    """With per-layer masks, changing the LAST receiver token must not move the
    FIRST receiver position's logits at all."""
    from lkv.kvcomm.trim import forward_with_ragged_cache

    base = _prefill(tiny_qwen3, 12)
    outs = []
    for q in (torch.tensor([[11, 12, 13, 14]]), torch.tensor([[11, 12, 13, 99]])):
        cache = prepare_key_cache(base, layers_list=[2])
        with torch.no_grad():
            outs.append(forward_with_ragged_cache(tiny_qwen3, q, cache))
    drift = (outs[0][0, 0] - outs[1][0, 0]).abs().max().item()
    print(f"D9 MEASUREMENT: forward_with_ragged_cache drift = {drift:.3e}")
    assert drift == 0.0, f"future token leaked into position 0: drift={drift:.3e}"


def test_forward_with_ragged_cache_matches_normal_forward_when_untrimmed(tiny_qwen3):
    """Self-review check (not in the brief): forward_with_ragged_cache is only a
    valid decoder reimplementation if, absent any trimming, it reproduces the
    exact logits of the model's own forward. A zero-drift result through a
    reimplementation that silently computes something else would be meaningless.
    """
    from lkv.kvcomm.trim import forward_with_ragged_cache

    base = _prefill(tiny_qwen3, 12)
    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    cache_for_custom = prepare_key_cache(base, all_layers)
    cache_for_stock = prepare_key_cache(base, all_layers)

    q = torch.tensor([[11, 12, 13, 14]])
    with torch.no_grad():
        custom_logits = forward_with_ragged_cache(tiny_qwen3, q, cache_for_custom)

        mask = torch.ones(1, 12 + 4, dtype=torch.long)
        stock_logits = tiny_qwen3(
            q, attention_mask=mask, past_key_values=cache_for_stock, use_cache=True
        ).logits

    torch.testing.assert_close(custom_logits, stock_logits, atol=1e-5, rtol=1e-4)
