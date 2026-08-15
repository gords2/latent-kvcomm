"""LatentMAS: one model plays N roles over a single growing KV cache.

Differences from stock-latentmas/methods/latent_mas.py, all deliberate:
  D2  The relay cache is ALWAYS passed to the judger. Upstream nulls it when
      latent_steps == 0 (`past_for_decoding = past_kv if self.latent_steps > 0
      else None`, :158), which turns the configuration into a single-agent run
      that still reports itself as LatentMAS.
  D5  Truncation is not a dead flag pair. It is a `trim_hook` argument, which
      is also the whole integration surface for KVComm -- no kvcomm import here.
  D7  No vLLM import. Upstream imports SamplingParams unguarded at module scope
      (:9), so the pure-HF path cannot run without vLLM installed.
  Also: past_length() takes the max over ALL layers. Upstream reads layer 0,
  which KVComm always keeps full -- so a layer-trimmed cache would look intact.
"""
from __future__ import annotations

import torch

from lkv.decode import generate_text_batch
from lkv.modeling import pad_batch
from lkv.prompts import ROLES, latent_agent_message


def past_length(past_kv) -> int:
    if past_kv is None:
        return 0
    return max(int(k.shape[-2]) for k in past_kv.key_cache)


def cumsum_positions(mask: torch.Tensor, q_len: int) -> torch.Tensor:
    """HuggingFace's own position convention, for the last `q_len` columns.

    `prepare_inputs_for_generation` computes
    `position_ids = attention_mask.cumsum(-1) - 1`, held at 1 where the mask is
    0, so a left-padded row's first REAL token is position 0 and pad columns
    consume no position index.

    A bare `model(...)` call does not route through that method and instead
    derives positions from `cache_position` alone -- a plain arange over the
    cache, in which pad columns DO consume indices. That distinction is
    invisible for an unpadded batch and wrong for a padded one (D16): every hop
    of the relay left-pads its prompt, inserting pad columns BETWEEN the cached
    history and the new real tokens, so under arange a row's content is pushed
    away from the previous hop's content by exactly its pad count. Since the pad
    count is set by the LONGEST item in the batch, an item's own representation
    would otherwise depend on who it is batched with -- and the judger, which
    goes through `generate()`, would use a different convention from the latent
    hops feeding it.
    """
    pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    return pos[:, -q_len:]


def generate_latent(model, projector, input_ids, attention_mask, latent_steps,
                    past_key_values, past_mask=None):
    """Prefill the prompt, then roll `latent_steps` last-hidden-states back in.

    Returns (cache, running_mask, positions_added). `running_mask` spans the whole
    cache and MUST be threaded into the next hop: the prompts are left-padded and
    ragged, so those pad positions have to stay masked for the rest of the relay.
    Synthesizing ones later re-enables attention to them and silently changes the
    output -- measured max logit delta ~0.53 on the tiny fixture.
    """
    before = past_length(past_key_values)
    if before and past_mask is None:
        raise ValueError(
            f"past_mask is required when the cache already holds {before} "
            "positions; the relay's earlier prompts were left-padded."
        )
    # The mask handed to the model must cover cache + this prompt.
    mask = attention_mask if not before else torch.cat(
        [past_mask.to(dtype=attention_mask.dtype, device=attention_mask.device),
         attention_mask], dim=1)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            # D16: required, not optional. Without it this forward derives
            # positions from cache_position (a bare arange), which counts the
            # left-padding between hops as real distance -- see
            # cumsum_positions. The judger's generate() already uses this
            # convention, so omitting it also makes the two halves of one relay
            # disagree about where the same tokens sit.
            position_ids=cumsum_positions(mask, int(input_ids.shape[1])),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        cache = out.past_key_values
        for _ in range(latent_steps):
            hidden = out.hidden_states[-1][:, -1, :]
            embed = projector(hidden).unsqueeze(1)
            # A latent position is always real (never a pad), so exactly one
            # `1` column is appended per step -- never re-derived from shape.
            mask = torch.cat(
                [mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)],
                dim=1,
            )
            out = model(
                inputs_embeds=embed,
                attention_mask=mask,
                position_ids=cumsum_positions(mask, 1),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = out.past_key_values
    return cache, mask, past_length(cache) - before


def _render(tokenizer, cfg, items, role):
    texts = [
        tokenizer.apply_chat_template(
            latent_agent_message(role, item["question"], cfg.task),
            tokenize=False, add_generation_prompt=True,
        )
        for item in items
    ]
    return pad_batch(tokenizer, texts)


def run_latentmas(model, tokenizer, cfg, items, trim_hook=None, projector=None):
    """Returns (final_texts, hop_records).

    trim_hook(cache, hop_index, inherited_len) -> cache is called after each
    non-final agent. It is the only place KVComm enters the relay.
    """
    if projector is None:
        from lkv.realign import LatentProjector
        projector = LatentProjector(model, cfg.realign, cfg.renorm_latent, cfg.device)

    roles = list(ROLES[: cfg.n_agents - 1]) + ["judger"] if cfg.n_agents > 1 else ["judger"]
    cache = None
    run_mask = None          # spans the whole cache; threaded, never synthesized
    hops: list[dict] = []
    final_texts: list[str] = []

    for hop_index, role in enumerate(roles):
        ids, mask = _render(tokenizer, cfg, items, role)
        ids = ids.to(model.device)
        mask = mask.to(model.device)
        before = past_length(cache)

        if role == "judger":
            # Pass the judger prompt ALONE. generate_text_batch owns widening for
            # the cached span -- see its contract (lkv/decode.py). Hand-rolling
            # the widening here duplicates the cached span and desyncs
            # cache_position from input_ids, raising IndexError inside generate().
            final_texts, _ = generate_text_batch(
                model, tokenizer, ids, mask,
                max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy,
                temperature=cfg.temperature, top_p=cfg.top_p,
                past_key_values=cache,   # D2: never nulled
                past_mask=run_mask,      # the relay's real mask, not ones
            )
            # generate() mutates `cache` IN PLACE (every DynamicCache.update()
            # call during decoding appends to it), so re-reading past_length
            # here reports the cache's real post-generation length -- unlike
            # the pre-fix version, which wrote `before` again under the
            # `positions_after` key and reported entry state as if nothing
            # had grown.
            hops.append({"hop": hop_index, "role": role, "latent_steps": 0,
                         "positions_before": before, "positions_after": past_length(cache)})
            continue

        cache, run_mask, _ = generate_latent(
            model, projector, ids, mask, cfg.latent_steps, cache, past_mask=run_mask
        )
        after = past_length(cache)
        if trim_hook is not None:
            cache = trim_hook(cache, hop_index, before)
            # A trim shortens the cache, so the running mask must be re-cut to
            # match or the next hop's widths disagree. Trimming keeps a PREFIX of
            # each layer, so the surviving mask is the matching prefix.
            run_mask = run_mask[:, : past_length(cache)]
        hops.append({"hop": hop_index, "role": role, "latent_steps": cfg.latent_steps,
                     "positions_before": before, "positions_after": after,
                     "positions_after_trim": past_length(cache)})

    return final_texts, hops
