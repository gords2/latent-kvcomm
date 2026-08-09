"""Batched text and latent generation.

The output slice is taken at the PADDED prompt width, not each row's unpadded
length. With left padding those differ, and using the unpadded length -- as
upstream does at stock-latentmas/models.py:230,260-267 -- lands inside the
prompt for every row that was padded.

Callers that pass `past_key_values` must pass the remaining prompt ALONE
(not pre-widened to the full input_ids/mask width). `generate()` computes
`cache_position = arange(input_ids.shape[1])[past_len:]` and forwards only
`input_ids[:, past_len:]` to the model -- so a prompt-only `input_ids` from a
relay boundary underflows that slice (IndexError) unless `input_ids` is first
widened to span the cached prefix too. This function owns that widening so
every caller gets it once, correctly.
"""
from __future__ import annotations

import torch


def generate_text_batch(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    greedy: bool,
    temperature: float = 0.6,
    top_p: float = 0.95,
    past_key_values=None,
):
    if attention_mask.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"attention_mask width {attention_mask.shape[1]} != input_ids width "
            f"{input_ids.shape[1]}. Pass the prompt alone for both tensors -- "
            "generate_text_batch prepends the cached span itself; a caller "
            "that pre-widens either tensor gets the cached span counted twice."
        )

    past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
    if past_len:
        # generate() only ever reads input_ids[:, past_len:] (see module
        # docstring): the first `past_len` columns are never embedded or
        # looked at, they exist purely so arange(input_ids.shape[1])[past_len:]
        # lines up with the cache's real length. Verified directly on this
        # stack: true ids, all-PAD, and random garbage in this span produce
        # byte-identical output. pad_token_id is therefore safe, inert
        # filler -- this is not a bug, it is dead weight by construction.
        pad_prefix = torch.full(
            (input_ids.shape[0], past_len), tokenizer.pad_token_id,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        # These columns represent real, already-cached content, not padding
        # to be masked out -- so the mask prefix is 1s, not 0s.
        mask_prefix = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        input_ids = torch.cat([pad_prefix, input_ids], dim=1)
        attention_mask = torch.cat([mask_prefix, attention_mask], dim=1)

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        return_dict_in_generate=True,
    )
    if greedy:
        kwargs.update(do_sample=False, num_beams=1)
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    with torch.no_grad():
        out = model.generate(**kwargs)

    sequences = out.sequences
    # Left padding means every row's prompt ends at the same column. When a
    # cache was supplied, input_ids was widened above to include the cached
    # span, so this is still the full (possibly widened) prompt width.
    prompt_width = int(input_ids.shape[1])
    texts = [
        tokenizer.decode(sequences[i, prompt_width:], skip_special_tokens=True).strip()
        for i in range(sequences.shape[0])
    ]
    return texts, sequences
