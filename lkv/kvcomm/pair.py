"""KVComm: one sender prefill, one receiver generation, one hop.

The receiver never sees the context as text -- the cache is its only access,
which is what makes the Baseline (query only) the correct floor and the Skyline
(full text to one model) the correct ceiling.

Calibration runs at FULL retention (stock-kvcomm/models.py:58-59): importance is
defined over the attention a receiver pays to a complete context.

D8 (docs/FINDINGS.md): whenever `layers_list` is a strict subset of all layers,
`prepare_key_cache` returns a RAGGED cache, and the stock forward path attends
to it non-causally. `run_pair` now selects its decoder from `cfg.causal_ragged`:

  off (default) -- `generate_text_batch`, the stock path. KVComm's published
      table was produced on it, so reproducing those numbers requires it.
  on            -- `generate_with_ragged_cache`, per-layer causal masks. This is
      what the numbers SHOULD be, and what the integration must use, since for
      the integration no upstream result exists to be faithful to.

Task 21 Step 5 measures the gap between the two against the published
0.51/0.62/0.62 and needs a GPU.

`receiver_logits` stays on the stock path unconditionally: it is a
single-forward diagnostic with no live caller today (see docs/FINDINGS.md's D16
note). Calibration is unaffected either way -- it always runs at full retention,
so its cache is never ragged.
"""
from __future__ import annotations

import torch

from lkv.decode import generate_text_batch
from lkv.kvcomm.importance import attention_from_stash, layer_importance
from lkv.kvcomm.ragged_decode import generate_with_ragged_cache
from lkv.kvcomm.tracer import install_tracer, remove_tracer
from lkv.kvcomm.trim import prepare_key_cache
from lkv.modeling import chat_pad_batch
from lkv.prompts import (
    kvcomm_receiver_prompt,
    kvcomm_sender_prompt,
    kvcomm_skyline_prompt,
)


def _sender_cache(model_a, tokenizer, items, cfg):
    """Prefill Instruction+Context on the sender.

    Returns (cache, mask, ctx_len). `mask` is the sender's own attention_mask
    from chat_pad_batch, which left-pads ragged batches -- so it can hold real
    zeros. Every caller must thread this mask through as the historical mask
    for the cached span rather than synthesizing all-ones: a synthesized ones
    mask re-enables attention to pad key/value vectors, which are real,
    non-inert activations produced by a full forward pass (docs/FINDINGS.md
    D12), not zeros the way a padded input_id is.
    """
    texts = [kvcomm_sender_prompt(it["instruction"], it["context"]) for it in items]
    ids, mask = chat_pad_batch(tokenizer, texts, getattr(cfg, "enable_thinking", True))
    ids, mask = ids.to(model_a.device), mask.to(model_a.device)
    with torch.no_grad():
        out = model_a(input_ids=ids, attention_mask=mask, use_cache=True)
    return out.past_key_values, mask, int(ids.shape[1])


def _receiver_inputs(tokenizer, model_b, items, cfg):
    texts = [kvcomm_receiver_prompt(it["instruction"], it["question"]) for it in items]
    ids, mask = chat_pad_batch(tokenizer, texts, getattr(cfg, "enable_thinking", True))
    return ids.to(model_b.device), mask.to(model_b.device)


def calibrate(model, tokenizer, items, cfg) -> dict:
    """Importance totals over cfg.calib_size items, at full retention."""
    totals = {i: [] for i in range(model.config.num_hidden_layers)}
    saved = install_tracer(model)
    try:
        for item in items[: cfg.calib_size]:
            cache, sender_mask, _ = _sender_cache(model, tokenizer, [item], cfg)
            ids, mask = _receiver_inputs(tokenizer, model, [item], cfg)
            full_mask = torch.cat(
                [sender_mask.to(dtype=mask.dtype, device=mask.device), mask], dim=1)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=full_mask,
                      past_key_values=cache, use_cache=True)
            q_len = int(ids.shape[1])
            attn = {
                i: attention_from_stash(layer.self_attn, layer.self_attn.attn_inputs,
                                        q_len_receiver=q_len)
                for i, layer in enumerate(model.model.layers)
            }
            for i, value in layer_importance(attn, q_len=q_len).items():
                totals[i].append(float(value))
    finally:
        remove_tracer(model, saved)
    return totals


def receiver_logits(model_a, model_b, tokenizer, cfg, items, layers_list):
    """One forward pass of the receiver. layers_list=None means no cache.

    See the module docstring's D8 note: when layers_list is a strict subset of
    all layers the cache handed to this single forward pass is ragged, and this
    plain model call inherits the same non-causal-attention limitation as
    run_pair. Calls with layers_list=None or a full-layer list are unaffected.
    """
    ids, mask = _receiver_inputs(tokenizer, model_b, items, cfg)
    if layers_list is None:
        with torch.no_grad():
            return model_b(input_ids=ids, attention_mask=mask, use_cache=False).logits
    cache, sender_mask, _ = _sender_cache(model_a, tokenizer, items, cfg)
    trimmed = prepare_key_cache(cache, layers_list)
    full_mask = torch.cat(
        [sender_mask.to(dtype=mask.dtype, device=mask.device), mask], dim=1)
    with torch.no_grad():
        return model_b(input_ids=ids, attention_mask=full_mask,
                       past_key_values=trimmed, use_cache=True).logits


def run_pair(model_a, model_b, tokenizer, cfg, items, layers_list) -> list[str]:
    """One sender prefill, one receiver generation, one hop.

    `cfg.causal_ragged` picks the decoder -- see the module docstring. The two
    are interchangeable by construction: generate_with_ragged_cache takes the
    same arguments in the same order and returns the same (texts, sequences),
    and reproduces generate_text_batch token for token on an untrimmed cache
    (tests/test_ragged_decode.py).
    """
    decode = (
        generate_with_ragged_cache
        if getattr(cfg, "causal_ragged", False)
        else generate_text_batch
    )
    cache, sender_mask, _ = _sender_cache(model_a, tokenizer, items, cfg)
    trimmed = prepare_key_cache(cache, layers_list)
    ids, mask = _receiver_inputs(tokenizer, model_b, items, cfg)
    # generate_text_batch owns widening the cached span onto input_ids/mask
    # itself (lkv/decode.py's contract) -- pass the receiver prompt ALONE plus
    # past_mask, the sender's real historical mask. Pre-widening here (as a
    # literal reading of the brief's sketch would) hands generate_text_batch an
    # attention_mask wider than input_ids and it raises ValueError on the
    # width-mismatch guard; synthesizing past_mask as all-ones instead of
    # threading sender_mask reintroduces D12.
    texts, _ = decode(
        model_b, tokenizer, ids, mask,
        max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy,
        temperature=cfg.temperature, top_p=cfg.top_p,
        past_key_values=trimmed, past_mask=sender_mask,
    )
    return texts


def _plain(model, tokenizer, cfg, texts) -> list[str]:
    ids, mask = chat_pad_batch(tokenizer, texts, getattr(cfg, "enable_thinking", True))
    out, _ = generate_text_batch(
        model, tokenizer, ids.to(model.device), mask.to(model.device),
        max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy,
        temperature=cfg.temperature, top_p=cfg.top_p,
    )
    return out


def run_skyline(model, tokenizer, cfg, items) -> list[str]:
    return _plain(model, tokenizer, cfg, [
        kvcomm_skyline_prompt(it["instruction"], it["context"], it["question"])
        for it in items
    ])


def run_baseline(model, tokenizer, cfg, items) -> list[str]:
    return _plain(model, tokenizer, cfg, [
        kvcomm_receiver_prompt(it["instruction"], it["question"]) for it in items
    ])
