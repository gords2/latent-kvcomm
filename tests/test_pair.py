import torch

from lkv.config import RunConfig
from lkv.kvcomm.pair import calibrate, run_baseline, run_pair, run_skyline


class _Tok:
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=None):
        # length proportional to text length so context/question differ in size
        seqs = [[7] * max(2, min(12, len(t) // 8)) for t in texts]
        n = max(len(s) for s in seqs)
        ids = [[0] * (n - len(s)) + s for s in seqs]
        am = [[0] * (n - len(s)) + [1] * len(s) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(am)}

    def decode(self, t, skip_special_tokens=True):
        return "paris"


def _cfg(**kw):
    base = dict(system="kvcomm", task="hotpotqa",
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                model_b="meta-llama/Llama-3.1-8B-Instruct",
                top_layers=0.5, alpha=1.0, mu=0.5, sigma=10.0, calib_size=1,
                greedy=True, max_new_tokens=4, device="cpu", generate_bs=2,
                latent_steps=0)
    base.update(kw)
    return RunConfig(**base)


_ITEMS = [
    {"instruction": "Answer using the context.",
     "context": "Paris is the capital of France. " * 6,
     "question": "What is the capital of France?", "gold": "Paris"},
    {"instruction": "Answer using the context.",
     "context": "Rome is the capital of Italy. " * 6,
     "question": "What is the capital of Italy?", "gold": "Rome"},
]


def test_calibrate_returns_one_entry_per_layer(tiny_llama):
    totals = calibrate(tiny_llama, _Tok(), _ITEMS, _cfg())
    assert sorted(totals) == [0, 1, 2, 3]
    assert all(len(v) == 1 for v in totals.values()), "calib_size=1 -> one sample"
    assert all(isinstance(v[0], float) for v in totals.values())


def test_calibrate_honours_calib_size(tiny_llama):
    totals = calibrate(tiny_llama, _Tok(), _ITEMS, _cfg(calib_size=2))
    assert all(len(v) == 2 for v in totals.values())


def test_run_pair_produces_one_answer_per_item(tiny_llama):
    outs = run_pair(tiny_llama, tiny_llama, _Tok(), _cfg(), _ITEMS, layers_list=[2])
    assert len(outs) == len(_ITEMS)
    assert all(isinstance(o, str) for o in outs)


def test_baseline_never_sees_the_context(tiny_llama, monkeypatch):
    seen = []

    class _Spy(_Tok):
        def __call__(self, texts, **kw):
            seen.extend(texts)
            return super().__call__(texts, **kw)

    run_baseline(tiny_llama, _Spy(), _cfg(), _ITEMS)
    joined = " ".join(seen)
    assert "capital of France" in joined      # the question
    assert "Paris is the capital" not in joined  # the context


def test_skyline_sees_both(tiny_llama):
    seen = []

    class _Spy(_Tok):
        def __call__(self, texts, **kw):
            seen.extend(texts)
            return super().__call__(texts, **kw)

    run_skyline(tiny_llama, _Spy(), _cfg(), _ITEMS)
    joined = " ".join(seen)
    assert "Paris is the capital" in joined and "capital of France" in joined


class _DistinctTok(_Tok):
    """Like _Tok, but token ids vary by position instead of all being the
    literal constant `7`.

    _Tok's `[7] * k` scheme makes every position's post-embedding value vector
    IDENTICAL (same id everywhere -> same embed_tokens() row everywhere). A
    softmax-weighted average of identical value vectors returns that same
    vector NO MATTER what the attention weights are -- so with _Tok, whether
    the receiver's cache participates correctly, incorrectly, or NOT AT ALL,
    the output is mathematically forced to the same answer. That degeneracy
    was measured directly: with _Tok, with-cache vs no-cache differed by
    1.8e-07 (float rounding) even with correct code, which would have made
    this sanity check fail on GOOD code and given false confidence on code
    that silently drops the cache. Distinct, position-varying ids break the
    degeneracy: value vectors genuinely differ across positions, so this test
    can actually tell "the cache changed the output" apart from "nothing
    could have changed the output regardless of correctness".
    """

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=None):
        seqs = [
            [2 + (sum(map(ord, t)) % 50) + i for i in range(max(2, min(12, len(t) // 8)))]
            for t in texts
        ]
        n = max(len(s) for s in seqs)
        ids = [[0] * (n - len(s)) + s for s in seqs]
        am = [[0] * (n - len(s)) + [1] * len(s) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(am)}


def test_full_retention_pair_differs_from_baseline(tiny_llama):
    """Sanity: transmitting all layers must change the receiver's logits versus
    transmitting nothing. If not, the cache is not reaching the receiver -- the
    single most important thing to catch in this file, since a silently-ignored
    cache would make every KVComm number equal the Baseline while looking like a
    real measurement.

    Uses _DistinctTok, not _Tok (see its docstring): _Tok's repeated-constant
    tokens make the check pass trivially regardless of correctness.

    Asserts a MINIMUM MAGNITUDE, not mere inequality: a numerical-noise-sized
    difference (e.g. float rounding, order 1e-6) must not be able to pass this
    check. 0.05 sits comfortably below every real-signal measurement taken
    while writing this test (0.12-0.83 mean/max across several non-degenerate
    fixtures) and far above the ~1e-7 floor observed for float32 noise.
    """
    from lkv.kvcomm.pair import receiver_logits
    all_layers = list(range(tiny_llama.config.num_hidden_layers))
    with_cache = receiver_logits(tiny_llama, tiny_llama, _DistinctTok(), _cfg(),
                                 _ITEMS[:1], layers_list=all_layers)
    no_cache = receiver_logits(tiny_llama, tiny_llama, _DistinctTok(), _cfg(),
                               _ITEMS[:1], layers_list=None)
    assert not torch.allclose(with_cache, no_cache, atol=1e-4)
    diff = (with_cache - no_cache).abs().max().item()
    print(f"full-retention-vs-no-cache max logit delta = {diff:.3e}")
    assert diff > 0.05, f"logit delta {diff:.3e} is too small to rule out noise"


def test_sender_mask_is_threaded_not_synthesized(tiny_llama):
    """D12-class regression, specific to pair.py's own cache-building path.

    _sender_cache's batch is left-padded whenever items have different context
    lengths (pad_batch's own contract -- see lkv/modeling.py). A synthesized
    all-ones mask over that cached span (instead of the real historical mask)
    would let the receiver attend to pad key/value vectors, which are real,
    non-inert activations (docs/FINDINGS.md's D12). This measures exactly that
    class of bug for pair.py's own cache plumbing, the way
    tests/test_padding.py::test_cached_generation_honours_a_padded_history does
    for lkv/decode.py.

    The oracle is the shorter-context item run ALONE: pad_batch never pads a
    batch of one, so a solo call is untouched by any padding bug and gives an
    unimpeachable ground truth for that same row's logits inside a batch of two.
    """
    from lkv.kvcomm.pair import receiver_logits

    short_item = {"instruction": "Answer.", "context": "Rome.",
                  "question": "Q?", "gold": "Rome"}
    long_item = {"instruction": "Answer.",
                 "context": "Paris is the capital of France.",
                 "question": "Q?", "gold": "Paris"}
    all_layers = list(range(tiny_llama.config.num_hidden_layers))

    batched = receiver_logits(tiny_llama, tiny_llama, _Tok(), _cfg(),
                              [short_item, long_item], layers_list=all_layers)
    solo = receiver_logits(tiny_llama, tiny_llama, _Tok(), _cfg(),
                           [short_item], layers_list=all_layers)

    torch.testing.assert_close(batched[0], solo[0], atol=1e-4, rtol=1e-4)
