"""Calibrate layer importance at a LatentMAS agent boundary.

KVComm ranks layers by how much attention a RECEIVER pays to a SENDER's context
(stock-kvcomm/layer_importance.py:21). Here the receiver is the same model in its
next role and the context is the accumulated relay cache, so importance must be
measured there -- otherwise the ranking comes from a different distribution than
the one it will be applied to.
"""
from __future__ import annotations

import torch

from lkv.kvcomm.importance import attention_from_stash, layer_importance
from lkv.kvcomm.tracer import install_tracer, remove_tracer
from lkv.latentmas import generate_latent
from lkv.modeling import pad_batch
from lkv.prompts import ROLES, latent_agent_message
from lkv.realign import LatentProjector


def _render(tokenizer, cfg, item, role):
    text = tokenizer.apply_chat_template(
        latent_agent_message(role, item["question"], cfg.task),
        tokenize=False, add_generation_prompt=True,
    )
    return pad_batch(tokenizer, [text])


def calibrate_relay(model, tokenizer, cfg, items, hop_index: int = 0) -> dict:
    """Importance of each layer at the boundary AFTER agent `hop_index`."""
    totals = {i: [] for i in range(model.config.num_hidden_layers)}
    projector = LatentProjector(model, cfg.realign, cfg.renorm_latent, cfg.device)
    roles = list(ROLES[: cfg.n_agents - 1])
    if hop_index >= len(roles):
        raise ValueError(f"hop_index {hop_index} exceeds {len(roles)} latent agents")

    for item in items[: cfg.calib_size]:
        # Build the cache up to and including the agent at hop_index. The running
        # mask must be threaded, exactly as in run_latentmas -- the prompts are
        # left-padded, so an all-ones mask over the cached span would re-enable
        # attention to pad positions and skew the importance scores this whole
        # function exists to measure.
        cache = None
        run_mask = None
        for role in roles[: hop_index + 1]:
            ids, mask = _render(tokenizer, cfg, item, role)
            cache, run_mask, _ = generate_latent(
                model, projector, ids.to(model.device), mask.to(model.device),
                cfg.latent_steps, cache, past_mask=run_mask,
            )

        # Trace the NEXT role's prefill over that cache.
        next_role = roles[hop_index + 1] if hop_index + 1 < len(roles) else "judger"
        ids, mask = _render(tokenizer, cfg, item, next_role)
        ids, mask = ids.to(model.device), mask.to(model.device)
        # The mask handed to the traced forward must be built from run_mask, the
        # relay's REAL historical mask, not torch.ones(1, past) -- a synthesized
        # ones mask would re-enable attention to the left-padding gaps between
        # ragged prompts and skew the very importance scores this function
        # exists to measure (see lkv/latentmas.py's generate_latent docstring).
        full_mask = torch.cat([run_mask.to(dtype=mask.dtype, device=mask.device), mask],
                              dim=1)
        saved = install_tracer(model)
        try:
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
