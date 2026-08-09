"""Batched text and latent generation.

The output slice is taken at the PADDED prompt width, not each row's unpadded
length. With left padding those differ, and using the unpadded length -- as
upstream does at stock-latentmas/models.py:230,260-267 -- lands inside the
prompt for every row that was padded.
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
    # Left padding means every row's prompt ends at the same column.
    prompt_width = int(input_ids.shape[1])
    texts = [
        tokenizer.decode(sequences[i, prompt_width:], skip_special_tokens=True).strip()
        for i in range(sequences.shape[0])
    ]
    return texts, sequences
