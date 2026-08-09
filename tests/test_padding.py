"""D1: generation must be invariant to batching.

If padding_side is 'right', generate() continues from a PAD token for every
row shorter than the batch max, and the output slice lands inside the prompt.
The decisive check is that a row's greedy continuation is identical whether it
was generated alone or inside a ragged batch.

Also covers the cache-widening fix in generate_text_batch: callers on a relay
path pass the remaining prompt ALONE alongside a real past_key_values, and
generate_text_batch must widen input_ids/attention_mask to span the cached
prefix itself (see lkv/decode.py module docstring for why).
"""
import copy

import pytest
import torch

from lkv.decode import generate_text_batch
from lkv.modeling import pad_batch


class _IdTok:
    """Tokenizer stub whose decode() just echoes token ids back as strings."""
    pad_token_id = 0
    padding_side = "left"

    def decode(self, t, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in t)


def _greedy(model, ids, mask, n=6):
    with torch.no_grad():
        return model.generate(
            ids, attention_mask=mask, max_new_tokens=n,
            do_sample=False, num_beams=1,
            pad_token_id=model.config.pad_token_id,
        )


def test_batched_greedy_matches_unbatched(tiny_qwen3, tiny_ids, short_long_solo):
    ids, mask = tiny_ids
    short, long = short_long_solo

    batched = _greedy(tiny_qwen3, ids, mask)
    solo_short = _greedy(tiny_qwen3, short, torch.ones_like(short))
    solo_long = _greedy(tiny_qwen3, long, torch.ones_like(long))

    assert torch.equal(batched[0, -6:], solo_short[0, -6:]), (
        "short row diverged inside the batch -- padding side is wrong"
    )
    assert torch.equal(batched[1, -6:], solo_long[0, -6:])


def test_generate_text_batch_returns_only_the_continuation(tiny_qwen3):
    """The returned string must not contain any prompt token."""
    ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    mask = torch.ones_like(ids)

    texts, seqs = generate_text_batch(
        tiny_qwen3, _IdTok(), ids, mask, max_new_tokens=5, greedy=True,
    )
    assert len(texts) == 1
    assert seqs.shape[1] == 4 + 5
    # the continuation is exactly the last 5 ids, nothing more
    assert texts[0].split() == [str(int(x)) for x in seqs[0, 4:]]


def test_no_cache_path_is_unaffected(tiny_qwen3):
    """Regression: adding the widening logic must not touch the no-cache path.

    Same inputs as test_generate_text_batch_returns_only_the_continuation,
    but here we additionally cross-check against calling model.generate
    directly with identical arguments -- the two must be byte-identical.
    """
    ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    mask = torch.ones_like(ids)

    texts, seqs = generate_text_batch(
        tiny_qwen3, _IdTok(), ids, mask, max_new_tokens=5, greedy=True,
    )

    reference = _greedy(tiny_qwen3, ids, mask, n=5)

    assert torch.equal(seqs, reference)
    assert seqs.shape[1] == 4 + 5
    assert texts[0].split() == [str(int(x)) for x in seqs[0, 4:]]


def test_prewidened_mask_is_rejected(tiny_qwen3):
    """A caller must pass the prompt alone; generate_text_batch does the
    widening itself. A pre-widened mask (or ids) is silently double-counted
    otherwise, so this must fail loudly instead."""
    ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    mask = torch.ones((1, 10), dtype=torch.long)  # caller pre-widened the mask

    with pytest.raises(ValueError, match="prompt alone"):
        generate_text_batch(
            tiny_qwen3, _IdTok(), ids, mask, max_new_tokens=3, greedy=True,
        )


def test_cached_generation_matches_uncached_reference(tiny_qwen3):
    """Continuing from a real past_key_values plus the remaining prompt must
    match running the whole sequence through generate() with no cache at all.

    This is the decisive check for the widening fix: without it, the
    prompt-only input_ids underflows generate()'s internal
    input_ids[:, past_len:] slice.
    """
    full_ids = torch.tensor([[5, 6, 7, 8, 9, 10, 20, 21, 22]], dtype=torch.long)
    prefix, prompt = full_ids[:, :6], full_ids[:, 6:]
    full_mask = torch.ones_like(full_ids)

    with torch.no_grad():
        reference = _greedy(tiny_qwen3, full_ids, full_mask, n=4)
    reference_continuation = reference[0, full_ids.shape[1]:]

    with torch.no_grad():
        prefix_out = tiny_qwen3(prefix, use_cache=True)
    cache = prefix_out.past_key_values  # generate() mutates its cache argument

    prompt_mask = torch.ones_like(prompt)
    texts, seqs = generate_text_batch(
        tiny_qwen3, _IdTok(), prompt, prompt_mask, max_new_tokens=4, greedy=True,
        past_key_values=copy.deepcopy(cache), past_mask=torch.ones_like(prefix),
    )
    cached_continuation = seqs[0, -4:]

    assert torch.equal(cached_continuation, reference_continuation)


def test_cache_without_past_mask_is_rejected(tiny_qwen3):
    """past_mask has no safe default -- a missing one must be a hard error,
    not a silent all-ones guess (that guess is exactly what D-round-2 found
    unsafe for a padded cached history)."""
    prefix = torch.tensor([[5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        prefix_out = tiny_qwen3(prefix, use_cache=True)
    cache = prefix_out.past_key_values

    prompt = torch.tensor([[8, 9]], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt)

    with pytest.raises(ValueError, match="past_mask is required"):
        generate_text_batch(
            tiny_qwen3, _IdTok(), prompt, prompt_mask, max_new_tokens=3, greedy=True,
            past_key_values=cache,
        )


def test_cached_generation_honours_a_padded_history(tiny_qwen3):
    """The cached span can itself be left-padded -- every cache in this
    project comes from a pad_batch'd batch. Synthesizing an all-ones mask for
    that span (instead of the cache's true historical mask) lets later
    queries attend to pad positions that were correctly excluded when the
    cache was written. This is the decisive check for that regression.
    """
    # 2 pad + 5 real prefix tokens (left-padded), then a 3-token prompt.
    prefix_ids = torch.tensor([[0, 0, 11, 12, 13, 14, 15]], dtype=torch.long)
    prefix_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    prompt = torch.tensor([[20, 21, 22]], dtype=torch.long)

    full_ids = torch.cat([prefix_ids, prompt], dim=1)
    full_mask = torch.cat([prefix_mask, torch.ones_like(prompt)], dim=1)

    with torch.no_grad():
        reference = _greedy(tiny_qwen3, full_ids, full_mask, n=6)
    reference_continuation = reference[0, full_ids.shape[1]:]

    # Prime the cache with the same left-pad position convention generate()
    # uses internally (mask.cumsum(-1) - 1, masked_fill 1 at pad slots). A
    # plain forward() call defaults position_ids to a bare arange() and
    # ignores the mask entirely, which would desync the comparison for a
    # reason unrelated to the fix under test.
    position_ids = prefix_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(prefix_mask == 0, 1)
    with torch.no_grad():
        prefix_out = tiny_qwen3(
            prefix_ids, attention_mask=prefix_mask, position_ids=position_ids,
            use_cache=True,
        )
    cache = prefix_out.past_key_values  # generate() mutates its cache argument

    prompt_mask = torch.ones_like(prompt)
    texts, seqs = generate_text_batch(
        tiny_qwen3, _IdTok(), prompt, prompt_mask, max_new_tokens=6, greedy=True,
        past_key_values=copy.deepcopy(cache), past_mask=prefix_mask,
    )
    cached_continuation = seqs[0, -6:]

    assert torch.equal(cached_continuation, reference_continuation), (
        "cached generation diverged from the uncached reference -- "
        "past_mask was not honoured correctly"
    )


def test_pad_batch_is_left_padded():
    class _Tok:
        pad_token_id = 0
        padding_side = "right"          # deliberately wrong on the way in

        def __call__(self, texts, return_tensors=None, padding=None,
                     add_special_tokens=None):
            seqs = [[5, 6, 7], [5, 6, 7, 8, 9]]
            n = max(len(s) for s in seqs)
            if self.padding_side == "left":
                ids = [[0] * (n - len(s)) + s for s in seqs]
                am = [[0] * (n - len(s)) + [1] * len(s) for s in seqs]
            else:
                ids = [s + [0] * (n - len(s)) for s in seqs]
                am = [[1] * len(s) + [0] * (n - len(s)) for s in seqs]
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(am)}

    tok = _Tok()
    ids, mask = pad_batch(tok, ["a", "b"])
    assert tok.padding_side == "left", "pad_batch must force left padding"
    assert mask[0, 0].item() == 0
