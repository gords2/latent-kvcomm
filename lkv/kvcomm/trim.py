"""Layer-selective KV trimming, plus a correct mask for the resulting cache.

prepare_key_cache mirrors stock-kvcomm/models.py:95-108: selected layers are
copied whole, every other layer keeps only `sink` leading positions, and layer 0
is always retained in full (`if i in self.layers_list or i == 0`).

build_per_layer_masks / forward_with_ragged_cache are OURS. Upstream's default
path hands the ragged cache to the stock forward, where one full-length causal
mask is sliced down to each layer's key length -- see tests/test_ragged_cache.py
for the measurement (D8: drift 1.940e-01 on cache lengths [12, 1, 12, 1], versus
0.0 for an untrimmed control). Their opt-in `shift_back` path (models.py:253-359)
exists for the same reason but asserts at most two distinct cache lengths.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache


def layer_map(n_layers_a: int, n_layers_b: int) -> list[int]:
    return [round((i + 0.5) * n_layers_b / n_layers_a - 0.5) for i in range(n_layers_a)]


def prepare_key_cache(cache, layers_list, layer_map=None, sink: int = 1):
    keep = set(int(i) for i in layers_list)
    n = len(cache.key_cache)
    mapping = layer_map if layer_map is not None else list(range(n))
    out = DynamicCache()
    for i in range(n):
        k, v = cache.key_cache[i], cache.value_cache[i]
        if i in keep or i == 0:
            out.update(k, v, mapping[i])
        else:
            out.update(k[:, :, :sink, :], v[:, :, :sink, :], mapping[i])
    return out


def retention(cache) -> float:
    lengths = [int(k.shape[-2]) for k in cache.key_cache]
    full = max(lengths) * len(lengths)
    return (sum(lengths) / full) if full else 1.0


def build_per_layer_masks(cache, q_len: int, device, dtype, past_mask):
    """One additive causal mask per layer, each sized to that layer's kv length
    and to the cache's real batch size.

    `past_mask` is REQUIRED and is the ORIGINAL (untrimmed) historical
    attention_mask the full cache was built with -- shape (batch, full_len),
    1 where a cached position is real, 0 where it is a left-pad (every cache
    in this project comes from a batch tokenized by pad_batch, so this is not
    a hypothetical). prepare_key_cache/trim_delta always keep a PREFIX of each
    layer's positions (either the whole thing or the leading `sink`, see
    lkv/integrate.py's module docstring), so a layer whose own kv length is
    `past` corresponds to exactly that prefix of the original sequence,
    `past_mask[:, :past]`.

    This used to hard-code batch=1 and write an unconditional 0 (fully
    visible) for every historical column, which silently made every pad
    key/value attendable for any batch > 1 -- the same D12 defect already
    fixed on every other cached-span mask in this project (lkv/decode.py,
    lkv/latentmas.py, lkv/relay_calib.py, lkv/kvcomm/pair.py). There was no
    live caller to catch it because this function is only reachable from
    tests today (see docs/FINDINGS.md). A pad column now becomes a real -inf
    entry, per the mask's own batch row, instead of the unconditional 0 this
    function used to write regardless of `past_mask`.

    A trimmed layer's `sink` cached positions are visible wherever `past_mask`
    says they are real (they precede every receiver token); the receiver
    block is strictly lower-triangular.

    Uses literal -inf rather than torch.finfo(dtype).min: no row here is ever
    fully masked (every retained prefix keeps at least one real column by
    construction -- pad_batch never left-pads an entire row), so there is no
    softmax-of-all-(-inf) NaN risk, and an exact -inf keeps the causal
    boundary unambiguous under torch.isinf (see
    test_per_layer_masks_are_causal_and_correctly_sized).
    """
    batch = int(cache.key_cache[0].shape[0])
    if tuple(past_mask.shape[:1]) != (batch,):
        raise ValueError(
            f"past_mask has batch {tuple(past_mask.shape[:1])}, expected "
            f"({batch},) -- cache.key_cache[0].shape[0]."
        )
    future = torch.triu(
        torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1
    )
    causal_block = torch.zeros(q_len, q_len, device=device, dtype=dtype).masked_fill(
        future, float("-inf")
    ).view(1, 1, q_len, q_len).expand(batch, 1, q_len, q_len)

    masks = []
    for k in cache.key_cache:
        past = int(k.shape[-2])
        if past_mask.shape[1] < past:
            raise ValueError(
                f"past_mask covers {past_mask.shape[1]} positions but this "
                f"layer's cache holds {past} -- prepare_key_cache/trim_delta "
                "only ever keep a PREFIX of the original sequence, so "
                "past_mask must cover at least as many positions as the "
                "longest kept layer."
            )
        hist = past_mask[:, :past].to(device=device)
        pad = (hist == 0).view(batch, 1, 1, past)
        hist_block = torch.zeros(batch, 1, q_len, past, device=device, dtype=dtype)
        hist_block = hist_block.masked_fill(pad, float("-inf"))
        masks.append(torch.cat([hist_block, causal_block], dim=-1))
    return masks


def forward_with_ragged_cache(model, input_ids, cache, past_mask, sink: int = 1):
    """Run the decoder with a per-layer mask so causality holds in every layer.

    Reimplements Qwen3Model.forward's per-layer loop (transformers 4.53.3,
    modeling_qwen3.py:450-468) because that loop shares ONE causal mask and ONE
    `position_embeddings` across every layer, built from a single
    `cache_position` derived from `past_key_values.get_seq_length()` (which
    reads layer 0). That is exactly the D8 mechanism: a trimmed layer's shorter
    key length gets a mask sliced from the wrong (full-length) reference frame.
    Here every layer gets its own cache_position -- a trimmed layer places
    receiver tokens right after its sink, a retained layer places them after the
    full context -- so `position_embeddings` and the mask are both computed
    per layer. Upstream's shift_back does the same shift but only for two
    distinct lengths (models.py:240-249); this generalizes it to any number.

    `past_mask` is REQUIRED -- see build_per_layer_masks's docstring. There is
    no safe default: synthesizing all-ones for a left-padded ragged batch
    would silently re-enable attention to pad key/value vectors (D12), which
    is a real numerical effect, not inert filler.

    `sink` is accepted for interface symmetry with prepare_key_cache but is not
    used here: cache lengths are read directly off `cache`, which already
    reflects whatever sink size was used to build it.
    """
    q_len = int(input_ids.shape[1])
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    masks = build_per_layer_masks(cache, q_len, device, dtype, past_mask)

    hidden = model.model.embed_tokens(input_ids)
    for i, layer in enumerate(model.model.layers):
        past = int(cache.key_cache[i].shape[-2])
        cache_position = torch.arange(past, past + q_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        rotary = model.model.rotary_emb(hidden, position_ids)
        layer_out = layer(
            hidden,
            attention_mask=masks[i],
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=rotary,
        )
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)
