"""Causal multi-step decoding over a layer-trimmed (ragged) KV cache.

`trim.forward_with_ragged_cache` makes ONE forward pass causal. Generation needs
many, over a cache that grows, and three things change once it does:

1. **Per-layer position frames diverge as the cache grows.** A trimmed layer
   holds `[sink historical positions] ++ [everything generated since]`; a
   retained layer holds `[full history] ++ [the same tail]`. So a layer's mask
   cannot be recovered from its CURRENT length -- slicing `past_mask[:, :len]`
   would pull `len` positions of *history* for a trimmed layer, when all but
   `sink` of those are gone. This module therefore captures each layer's BASE
   length once, before the first forward, and thereafter builds every layer's
   column set explicitly as `past_mask[:, :base] ++ live ++ current`.

2. **The query block carries its own padding.** `build_per_layer_masks` writes a
   purely triangular receiver block, which is right for the batch-1 unpadded
   queries it is tested on but not for a left-padded batch: a real query must
   not attend to a pad position inside its own block. Here the mask is built
   over the whole concatenated column set, so padding and causality are applied
   uniformly and the distinction disappears.

3. **Rows finish at different steps.** `run_pair` generates 8-16 at a time.

Two deliberate matches to HuggingFace's own decoder, both required for
`test_untrimmed_ragged_decode_matches_generate_text_batch` to hold:

- **`torch.finfo(dtype).min`, not `-inf`.** `build_per_layer_masks` justifies
  literal `-inf` on the grounds that no row there is ever fully masked. That
  guarantee does not survive here: a left-pad query row attends to nothing in
  its own block, and an all-`-inf` softmax row is NaN. `generate()` uses
  `finfo.min` for exactly this reason.
- **`position_ids` from the mask's cumulative sum, not from `arange`.**
  `prepare_inputs_for_generation` computes
  `position_ids = attention_mask.cumsum(-1) - 1`, masked to 1 at pad, so a
  left-padded row's first real token is position 0 rather than position
  `n_pad`. `forward_with_ragged_cache` uses a bare `arange`, which is correct
  only for the unpadded batch-1 case it covers. Getting this wrong shifts RoPE
  for every padded row -- the same class of defect as D15, which scored a layer
  ranking on a query that had never had its rotary embedding applied.

The parameter order matches `lkv.decode.generate_text_batch` exactly, and the
return is the same `(texts, sequences)` pair, so `run_pair` swaps between them
on one flag. (The plan specified `-> list[str]`; that would have broken
`texts, _ = generate_text_batch(...)` at pair.py:125.)
"""
from __future__ import annotations

import torch


def _eos_ids(tokenizer, model) -> set[int]:
    """EOS may live on the tokenizer or the generation config, as int or list."""
    raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if raw is None:
        return set()
    return {int(x) for x in (raw if isinstance(raw, (list, tuple, set)) else [raw])}


def ragged_hidden(model, cache, base_lengths, past_mask, live_mask, cur_mask,
                  *, input_ids=None, inputs_embeds=None):
    """Last hidden state (POST final norm), causal under a ragged cache.

    Updates `cache` in place. `base_lengths[i]` is layer i's cache length BEFORE
    any of this pass's tokens were appended; `live_mask` covers everything
    appended since (shape (batch, n_live)); `cur_mask` is the block being run
    now. Layer i's columns are therefore
    `past_mask[:, :base_lengths[i]] ++ live_mask ++ cur_mask`, and its cache must
    already hold `base_lengths[i] + n_live` positions -- checked, because an
    off-by-one here produces fluent text conditioned on the wrong positions
    rather than an exception.

    Exactly one of `input_ids` / `inputs_embeds` is required. The embeds form is
    what LatentMAS's latent steps need: they feed a projected hidden state back
    in as a single position. Post-norm matches `output_hidden_states=True`,
    where transformers appends the final hidden state AFTER `model.norm`, so a
    caller reading `hidden_states[-1]` gets the same tensor this returns.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError(
            "pass exactly one of input_ids / inputs_embeds (got input_ids="
            f"{input_ids is not None}, inputs_embeds={inputs_embeds is not None})"
        )
    hidden = (
        model.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    )
    device = hidden.device
    dtype = next(model.parameters()).dtype
    batch, q_len = int(hidden.shape[0]), int(hidden.shape[1])
    n_live = int(live_mask.shape[1])
    neg = torch.finfo(dtype).min

    past_mask = past_mask.to(device=device, dtype=torch.long)
    live_mask = live_mask.to(device=device, dtype=torch.long)
    cur_mask = cur_mask.to(device=device, dtype=torch.long)
    future = torch.triu(
        torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1
    ).view(1, 1, q_len, q_len)

    # Layers that kept the SAME number of cached positions share a column set,
    # and therefore share a mask, position_ids and rotary embedding exactly.
    # prepare_key_cache leaves only two distinct lengths -- a kept layer holds
    # the full cache, a trimmed one holds `sink` (1) position -- so this turns
    # 32 mask allocations, cumsums and rotary_emb calls per decode step into 2.
    #
    # Pure memoisation, not an approximation: the key is the base length, which
    # fully determines all three values, and `rotary_emb(hidden, ...)` reads
    # only device and dtype off `hidden`, never its values, so sharing it across
    # layers cannot leak one layer's activations into another's.
    #
    # It matters only where generation is long. At Countries' 5 tokens the
    # overhead is invisible; at MBPP+'s 4096 it cost more than 10 minutes for a
    # single batch of 8, which put one sweep condition past 8 hours.
    per_base: dict[int, tuple] = {}

    for i, layer in enumerate(model.model.layers):
        base = int(base_lengths[i])
        held = int(cache.key_cache[i].shape[-2])
        if held != base + n_live:
            raise ValueError(
                f"layer {i} holds {held} cached positions but this decode "
                f"expects {base + n_live} (base {base} + {n_live} appended). "
                "base_lengths must be captured before the first forward and "
                "live_mask must track every token appended since."
            )
        if past_mask.shape[1] < base:
            raise ValueError(
                f"past_mask covers {past_mask.shape[1]} positions but layer {i} "
                f"kept {base} -- trimming only ever keeps a PREFIX of the "
                "original sequence, so past_mask must span the longest layer."
            )

        shared = per_base.get(base)
        if shared is None:
            cols = torch.cat([past_mask[:, :base], live_mask, cur_mask], dim=1)
            kv = int(cols.shape[1])

            mask = torch.zeros(batch, 1, q_len, kv, device=device, dtype=dtype)
            mask = mask.masked_fill((cols == 0).view(batch, 1, 1, kv), neg)
            block = mask[:, :, :, kv - q_len :].masked_fill(future, neg)
            mask = torch.cat([mask[:, :, :, : kv - q_len], block], dim=-1)

            # HF's own formula, applied to this layer's column set (module docstring).
            pos = (cols.cumsum(-1) - 1).masked_fill(cols == 0, 1)
            position_ids = pos[:, -q_len:]

            cache_position = torch.arange(
                base + n_live, base + n_live + q_len, device=device
            )
            rotary = model.model.rotary_emb(hidden, position_ids)
            per_base[base] = (mask, position_ids, cache_position, rotary)
        else:
            mask, position_ids, cache_position, rotary = shared
        out = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=rotary,
        )
        hidden = out[0] if isinstance(out, tuple) else out

    return model.model.norm(hidden)


def _ragged_forward(model, cache, base_lengths, past_mask, live_mask, cur_ids, cur_mask):
    """Logits form of `ragged_hidden`, for token-id input."""
    return model.lm_head(
        ragged_hidden(model, cache, base_lengths, past_mask, live_mask, cur_mask,
                      input_ids=cur_ids)
    )


def _next_tokens(logits, greedy: bool, temperature: float, top_p: float):
    """Greedy argmax, or temperature+top-p sampling in HuggingFace's warper
    order (TemperatureLogitsWarper then TopPLogitsWarper)."""
    if greedy:
        return torch.argmax(logits, dim=-1)

    scores = logits / temperature
    ordered, order = torch.sort(scores, descending=False)
    cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
    drop = cumulative <= (1 - top_p)
    drop[..., -1:] = False  # min_tokens_to_keep=1
    scores = scores.masked_fill(
        drop.scatter(1, order, drop), torch.finfo(scores.dtype).min
    )
    return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(-1)


def generate_with_ragged_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    greedy: bool,
    temperature: float = 0.6,
    top_p: float = 0.95,
    past_key_values=None,
    past_mask: torch.Tensor | None = None,
):
    """Drop-in causal replacement for `generate_text_batch` under a ragged cache.

    Pass the prompt ALONE for `input_ids`/`attention_mask`; this owns the
    widening onto the cached span, exactly as `generate_text_batch` does, and
    returns `(texts, sequences)` with `sequences` in the same layout -- the
    widened prompt followed by the generated tail.
    """
    if attention_mask.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"attention_mask width {attention_mask.shape[1]} != input_ids width "
            f"{input_ids.shape[1]}. Pass the prompt alone for both tensors -- "
            "generate_with_ragged_cache prepends the cached span itself; a "
            "caller that pre-widens either tensor gets the span counted twice."
        )
    if past_key_values is None:
        raise ValueError(
            "generate_with_ragged_cache exists to decode over a trimmed cache "
            "and requires one. With no cache there is nothing ragged to "
            "correct -- call lkv.decode.generate_text_batch instead."
        )
    if past_mask is None:
        raise ValueError(
            "past_key_values was given but past_mask is required and was not. "
            "The cached span can itself be left-padded (pad_batch left-pads "
            "every batch), so synthesizing an all-ones mask would let queries "
            "attend to pad positions the cache correctly masked out when it "
            "was written. Pass the exact attention_mask the cache was built with."
        )

    cache = past_key_values
    base_lengths = [int(k.shape[-2]) for k in cache.key_cache]
    full = max(base_lengths)
    batch = int(input_ids.shape[0])
    if tuple(past_mask.shape) != (batch, full):
        raise ValueError(
            f"past_mask has shape {tuple(past_mask.shape)}, expected "
            f"{(batch, full)} (batch size x the longest layer's cached length)."
        )

    device = input_ids.device
    eos = _eos_ids(tokenizer, model)

    # Same layout generate_text_batch produces: the cached span is represented
    # by inert pad ids so callers can slice at one prompt width for every row.
    sequences = torch.cat(
        [
            torch.full((batch, full), tokenizer.pad_token_id,
                       dtype=input_ids.dtype, device=device),
            input_ids,
        ],
        dim=1,
    )
    prompt_width = int(sequences.shape[1])

    live_mask = torch.zeros(batch, 0, dtype=torch.long, device=device)
    cur_ids, cur_mask = input_ids, attention_mask
    unfinished = torch.ones(batch, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _ragged_forward(
                model, cache, base_lengths, past_mask, live_mask, cur_ids, cur_mask
            )
            nxt = _next_tokens(logits[:, -1, :], greedy, temperature, top_p)
            # A finished row emits pad from here on, as generate() does.
            nxt = torch.where(
                unfinished, nxt, torch.full_like(nxt, tokenizer.pad_token_id)
            )
            sequences = torch.cat([sequences, nxt.unsqueeze(1)], dim=1)

            # Everything just run is now IN the cache, so it joins the live span.
            live_mask = torch.cat(
                [live_mask, cur_mask.to(device=device, dtype=torch.long)], dim=1
            )
            cur_ids = nxt.unsqueeze(1)
            cur_mask = torch.ones(batch, 1, dtype=torch.long, device=device)

            if eos:
                hit = torch.zeros_like(unfinished)
                for token in eos:
                    hit |= nxt == token
                unfinished &= ~hit
                if not bool(unfinished.any()):
                    break

    texts = [
        tokenizer.decode(sequences[i, prompt_width:], skip_special_tokens=True).strip()
        for i in range(batch)
    ]
    return texts, sequences
