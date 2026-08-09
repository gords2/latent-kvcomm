"""D1: generation must be invariant to batching.

If padding_side is 'right', generate() continues from a PAD token for every
row shorter than the batch max, and the output slice lands inside the prompt.
The decisive check is that a row's greedy continuation is identical whether it
was generated alone or inside a ragged batch.
"""
import torch

from lkv.decode import generate_text_batch
from lkv.modeling import pad_batch


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

    class _IdTok:
        pad_token_id = 0
        padding_side = "left"

        def decode(self, t, skip_special_tokens=True):
            return " ".join(str(int(x)) for x in t)

    texts, seqs = generate_text_batch(
        tiny_qwen3, _IdTok(), ids, mask, max_new_tokens=5, greedy=True,
    )
    assert len(texts) == 1
    assert seqs.shape[1] == 4 + 5
    # the continuation is exactly the last 5 ids, nothing more
    assert texts[0].split() == [str(int(x)) for x in seqs[0, 4:]]


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
