"""Task 21: multi-step generation over a ragged cache.

Task 12's `forward_with_ragged_cache` made ONE forward pass causal under a
layer-trimmed cache. Generation needs many passes over a cache that GROWS, and
that is where the new failure modes live: each layer's positions advance from
its own base length, so a trimmed layer's tokens sit right after its `sink`
while a retained layer's sit after the full context. Get that bookkeeping wrong
and the model still produces fluent text -- just conditioned on the wrong
positions. This suite is built so that cannot pass unnoticed.

Note on the plan's stated test 2 ("changing the last prompt token must not move
the first generated token's logits"): that property is false for any correct
decoder. The last prompt token is the immediate predecessor of the first
generated token, so it MUST move it. The causality probe that actually holds --
and the one Task 12 used -- is that changing the LAST query token must not move
the FIRST query position's logits. That is what is asserted here.
"""
import pytest
import torch

from lkv.decode import generate_text_batch
from lkv.kvcomm.ragged_decode import _ragged_forward, generate_with_ragged_cache
from lkv.kvcomm.trim import prepare_key_cache

PAD_ID = 0


class _Tok:
    """Faithful stand-in: decode is injective so a text comparison cannot pass
    vacuously the way a constant-string stub would."""

    pad_token_id = PAD_ID
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


def _prefill(model, ids, mask):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask, use_cache=True).past_key_values


def _padded_history(model, batch=2, width=12, pad_rows=((1, 3),)):
    """A left-padded historical batch and the mask it was really built with."""
    torch.manual_seed(7)
    ids = torch.randint(1, 200, (batch, width))
    mask = torch.ones(batch, width, dtype=torch.long)
    for row, n_pad in pad_rows:
        ids[row, :n_pad] = PAD_ID
        mask[row, :n_pad] = 0
    return ids, mask


def _padded_prompt(batch=2, width=5, pad_rows=((0, 2),)):
    torch.manual_seed(8)
    ids = torch.randint(1, 200, (batch, width))
    mask = torch.ones(batch, width, dtype=torch.long)
    for row, n_pad in pad_rows:
        ids[row, :n_pad] = PAD_ID
        mask[row, :n_pad] = 0
    return ids, mask


def _slice_cache(cache, row):
    """One batch row of a cache, as its own cache."""
    from transformers import DynamicCache

    out = DynamicCache()
    for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        out.update(k[row : row + 1].clone(), v[row : row + 1].clone(), i)
    return out


# --------------------------------------------------------------------------
# 1. The load-bearing test: untrimmed, the two paths must agree EXACTLY.
# --------------------------------------------------------------------------


def test_untrimmed_ragged_decode_matches_generate_text_batch(tiny_qwen3):
    """With a FULL (unragged) cache the ragged loop must reproduce
    generate_text_batch token for token.

    If it does not, the loop is not a faithful decoder and every trimmed number
    it later produces is meaningless -- a fluent-but-wrong decoder is exactly
    the silent-failure mode this project keeps hitting. Compared on TOKEN IDS,
    not decoded text: a text comparison can pass on a stub tokenizer that maps
    everything to one string.

    Deliberately run at batch 2 with padding on BOTH sides -- a left-padded
    history and a left-padded prompt -- because a mask bug that broadcasts row
    0 over the batch is invisible at batch 1.
    """
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()

    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)

    _, ref_seq = generate_text_batch(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        max_new_tokens=6, greedy=True,
        past_key_values=prepare_key_cache(base, all_layers), past_mask=hist_mask,
    )
    _, got_seq = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        max_new_tokens=6, greedy=True,
        past_key_values=prepare_key_cache(base, all_layers), past_mask=hist_mask,
    )

    assert got_seq.shape == ref_seq.shape, (got_seq.shape, ref_seq.shape)
    assert torch.equal(got_seq, ref_seq), (
        "ragged decode diverged from generate_text_batch on an UNTRIMMED cache:\n"
        f"  reference: {ref_seq.tolist()}\n"
        f"  ragged   : {got_seq.tolist()}"
    )


def test_untrimmed_ragged_forward_matches_stock_logits(tiny_qwen3):
    """The token-level test above is a DISCRETE comparison: argmax quantizes
    away small logit drift, so a subtly wrong mask can still produce identical
    tokens on a tiny random model. (Measured: neutralizing the query block's own
    padding leaves the generated tokens unchanged here while moving the logits.)
    This pins the continuous quantity instead -- with an untrimmed cache the
    ragged forward must reproduce the stock forward's logits numerically, at
    batch 2 with padding on both the history and the query.

    THE REFERENCE MUST CARRY EXPLICIT position_ids. A bare
    `model(ids, attention_mask=..., past_key_values=...)` call does NOT route
    through `prepare_inputs_for_generation`, so it derives positions from
    `cache_position` alone -- a plain arange that ignores left padding. On this
    fixture that gives row 1 positions [12..16] when its 3 pad positions mean
    the true positions are [9..13], and the resulting logits differ by
    1.930e-01. `generate()` (and therefore `generate_text_batch`, the contract
    this module must match) uses the cumsum form. Measured against the correct
    reference the difference is exactly 0.0.
    """
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()
    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)
    q_len = int(prompt_ids.shape[1])

    cache_mine = prepare_key_cache(base, all_layers)
    base_lengths = [int(k.shape[-2]) for k in cache_mine.key_cache]
    full_mask = torch.cat([hist_mask, prompt_mask], dim=1)
    position_ids = (full_mask.cumsum(-1) - 1).masked_fill(full_mask == 0, 1)[:, -q_len:]

    with torch.no_grad():
        mine = _ragged_forward(
            tiny_qwen3, cache_mine, base_lengths, hist_mask,
            torch.zeros(prompt_ids.shape[0], 0, dtype=torch.long),
            prompt_ids, prompt_mask,
        )
        stock = tiny_qwen3(
            prompt_ids,
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=prepare_key_cache(base, all_layers),
            use_cache=True,
        ).logits

    # Compare only the rows that are real queries: a fully-masked pad-query row
    # is uniform-attention garbage in both paths and carries no information.
    real = prompt_mask.bool()
    torch.testing.assert_close(mine[real], stock[real], atol=1e-5, rtol=1e-4)


def test_untrimmed_ragged_decode_returns_generate_text_batch_shaped_output(tiny_qwen3):
    """run_pair unpacks `texts, _ = generate_text_batch(...)`, so the swap is
    only one flag if this returns the same (texts, sequences) pair. The plan
    specified `-> list[str]`, which would have broken that call site."""
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)

    out = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        max_new_tokens=3, greedy=True,
        past_key_values=prepare_key_cache(base, [2]), past_mask=hist_mask,
    )
    assert isinstance(out, tuple) and len(out) == 2
    texts, seq = out
    assert isinstance(texts, list) and len(texts) == 2
    assert all(isinstance(t, str) for t in texts)
    # sequences carry the widened prompt plus the generated tail
    assert seq.shape[1] == hist_ids.shape[1] + prompt_ids.shape[1] + 3


# --------------------------------------------------------------------------
# 2. Causality -- at every growth stage, not just the first forward.
# --------------------------------------------------------------------------


def _first_position_drift(model, cache_builder, base_mask, q_pair, n_live):
    """Run a query block twice changing ONLY its last token. If the FIRST
    position's logits move, a later token leaked backwards."""
    outs = []
    for q in q_pair:
        cache = cache_builder()
        base_lengths = [int(k.shape[-2]) for k in cache.key_cache]
        live = torch.ones(q.shape[0], n_live, dtype=torch.long)
        if n_live:
            # grow the cache by n_live real tokens first
            filler = torch.full((q.shape[0], n_live), 42, dtype=torch.long)
            with torch.no_grad():
                _ragged_forward(model, cache, base_lengths, base_mask,
                                torch.zeros(q.shape[0], 0, dtype=torch.long),
                                filler, live)
        cur_mask = torch.ones_like(q)
        with torch.no_grad():
            outs.append(_ragged_forward(model, cache, base_lengths, base_mask,
                                        live, q, cur_mask))
    return (outs[0][0, 0] - outs[1][0, 0]).abs().max().item()


def test_trimmed_ragged_decode_is_causal_at_every_growth_stage(tiny_qwen3):
    """The multi-step-specific risk. Task 12 proved the FIRST forward is causal;
    the new failure mode is per-layer offsets drifting as the cache grows, so
    this probes causality after 0, 3 and 7 tokens have already been appended.

    The stock path drifts ~1.9e-01 on this cache shape (tests/test_ragged_cache.py
    pins that measurement); here it must be exactly 0.0.
    """
    hist_ids, hist_mask = _padded_history(tiny_qwen3, batch=1, pad_rows=())
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)
    q_pair = (torch.tensor([[11, 12, 13, 14]]), torch.tensor([[11, 12, 13, 99]]))

    for n_live in (0, 3, 7):
        drift = _first_position_drift(
            tiny_qwen3, lambda: prepare_key_cache(base, [2]), hist_mask, q_pair, n_live
        )
        print(f"D8 MEASUREMENT: ragged decode drift after {n_live} appended = {drift:.3e}")
        assert drift == 0.0, (
            f"future token leaked into position 0 after {n_live} appended "
            f"tokens: drift={drift:.3e}"
        )


def test_ragged_decode_is_step_consistent_as_the_cache_grows(tiny_qwen3):
    """Growth bookkeeping, tested end to end.

    Generating M tokens in one call must agree with generating 1 token, folding
    it into the prompt, and generating M-1 more. If per-layer cache_position or
    the per-layer masks drift as the cache grows, the two disagree -- and this
    catches it without needing a causal ground truth for a trimmed cache, which
    does not exist.
    """
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)
    hist_w = hist_ids.shape[1]

    _, seq_all = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        max_new_tokens=4, greedy=True,
        past_key_values=prepare_key_cache(base, [2]), past_mask=hist_mask,
    )
    _, seq_one = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        max_new_tokens=1, greedy=True,
        past_key_values=prepare_key_cache(base, [2]), past_mask=hist_mask,
    )
    # fold the first generated token into the prompt, then continue
    grown_ids = torch.cat([prompt_ids, seq_one[:, -1:]], dim=1)
    grown_mask = torch.cat([prompt_mask, torch.ones(prompt_mask.shape[0], 1,
                                                    dtype=prompt_mask.dtype)], dim=1)
    _, seq_rest = generate_with_ragged_cache(
        tiny_qwen3, tok, grown_ids, grown_mask,
        max_new_tokens=3, greedy=True,
        past_key_values=prepare_key_cache(base, [2]), past_mask=hist_mask,
    )

    assert torch.equal(seq_all[:, hist_w:], seq_rest[:, hist_w:]), (
        "one-shot and resumed generation disagree -- per-layer offsets drift as "
        f"the cache grows:\n  one-shot: {seq_all[:, hist_w:].tolist()}\n"
        f"  resumed : {seq_rest[:, hist_w:].tolist()}"
    )


# --------------------------------------------------------------------------
# 3. D12 across every step, not just the first.
# --------------------------------------------------------------------------


def test_ragged_decode_respects_a_padded_history(tiny_qwen3):
    """A left-padded cached span must keep its pad positions masked at EVERY
    generated step.

    Tested as batch invariance: row 1 decoded inside a batch must match row 1
    decoded alone with the identical (still padded) history. A mask that
    broadcasts row 0's padding over the batch, or that silently synthesizes
    all-ones for the cached span, makes these differ -- that is D12, which has
    already been found four separate times in this codebase.
    """
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3, batch=2, pad_rows=((1, 4),))
    prompt_ids, prompt_mask = _padded_prompt(batch=2, pad_rows=((0, 2),))
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)

    _, batched = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        max_new_tokens=5, greedy=True,
        past_key_values=prepare_key_cache(base, [2]), past_mask=hist_mask,
    )
    _, solo = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids[1:2], prompt_mask[1:2],
        max_new_tokens=5, greedy=True,
        past_key_values=_slice_cache(prepare_key_cache(base, [2]), 1),
        past_mask=hist_mask[1:2],
    )

    assert torch.equal(batched[1:2, -5:], solo[:, -5:]), (
        "row 1 decodes differently inside a batch than alone -- the cached "
        f"span's padding is not being honoured per row:\n"
        f"  batched: {batched[1:2, -5:].tolist()}\n  solo   : {solo[:, -5:].tolist()}"
    )


# --------------------------------------------------------------------------
# 4. The --causal_ragged switch.
# --------------------------------------------------------------------------


def test_causal_ragged_flag_selects_the_causal_decoder(tiny_qwen3, monkeypatch):
    """run_pair must actually route through the ragged decoder when the flag is
    set, and through generate_text_batch when it is not.

    A flag that parses but changes no behaviour is the exact shape of defect
    this project keeps finding, so this asserts on which function was CALLED
    rather than on the output -- both paths return plausible text either way.
    """
    import lkv.kvcomm.pair as pair
    from tests.test_pair import _Tok as PairTok, _cfg

    called = []

    def spy(name, real):
        def wrapped(*a, **kw):
            called.append(name)
            return real(*a, **kw)
        return wrapped

    monkeypatch.setattr(pair, "generate_text_batch",
                        spy("stock", pair.generate_text_batch))
    monkeypatch.setattr(pair, "generate_with_ragged_cache",
                        spy("causal", pair.generate_with_ragged_cache))

    item = [{"instruction": "i" * 40, "context": "c" * 90, "question": "q" * 20}]
    for flag, expected in ((False, "stock"), (True, "causal")):
        called.clear()
        pair.run_pair(tiny_qwen3, tiny_qwen3, PairTok(),
                      _cfg(causal_ragged=flag, top_layers=0.5),
                      item, layers_list=[2])
        assert called == [expected], (
            f"causal_ragged={flag} routed to {called}, expected [{expected!r}]")


def test_causal_ragged_is_rejected_for_systems_that_ignore_it():
    """Accepting the flag where it is not wired would stamp
    ragged_path=causal-ragged onto a run decoded non-causally -- a result file
    that lies about how it was produced."""
    from lkv.config import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--system", "latentmas", "--task", "gsm8k",
                    "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "10",
                    "--top_layers", "0.5", "--causal_ragged"])


# --------------------------------------------------------------------------
# 5. Contract guards and batch EOS.
# --------------------------------------------------------------------------


def test_missing_past_mask_is_a_hard_error(tiny_qwen3):
    """Same contract as generate_text_batch: there is no safe default, because
    synthesizing all-ones re-enables attention to pad key/value vectors."""
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)

    with pytest.raises(ValueError, match="past_mask"):
        generate_with_ragged_cache(
            tiny_qwen3, tok, prompt_ids, prompt_mask,
            max_new_tokens=2, greedy=True,
            past_key_values=prepare_key_cache(base, [2]), past_mask=None,
        )


def test_prewidened_inputs_are_rejected(tiny_qwen3):
    """generate_text_batch owns the widening and so does this; a caller that
    pre-widens gets the cached span counted twice."""
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)

    with pytest.raises(ValueError, match="width"):
        generate_with_ragged_cache(
            tiny_qwen3, tok, prompt_ids,
            torch.cat([hist_mask, prompt_mask], dim=1),
            max_new_tokens=2, greedy=True,
            past_key_values=prepare_key_cache(base, [2]), past_mask=hist_mask,
        )


def test_per_row_eos_stops_that_row_only(tiny_qwen3):
    """run_pair generates batches of 8-16, so rows hit EOS at different steps.
    A finished row must emit pad from then on while its neighbours continue --
    matching generate()'s own behaviour."""
    tok = _Tok()
    hist_ids, hist_mask = _padded_history(tiny_qwen3)
    prompt_ids, prompt_mask = _padded_prompt()
    base = _prefill(tiny_qwen3, hist_ids, hist_mask)

    kw = dict(max_new_tokens=5, greedy=True, past_mask=hist_mask)
    _, free = generate_with_ragged_cache(
        tiny_qwen3, tok, prompt_ids, prompt_mask,
        past_key_values=prepare_key_cache(base, [2]), **kw)
    tail = free[:, -5:]

    # Make row 0's SECOND generated token an EOS; row 1's tokens differ from it.
    eos = int(tail[0, 1])
    assert int(tail[1, 1]) != eos or int(tail[0, 0]) != int(tail[1, 0]), (
        "rows generated identically; this fixture cannot distinguish per-row stopping"
    )

    class _EosTok(_Tok):
        eos_token_id = eos

    _, stopped = generate_with_ragged_cache(
        tiny_qwen3, _EosTok(), prompt_ids, prompt_mask,
        past_key_values=prepare_key_cache(base, [2]), **kw)
    got = stopped[:, -5:]

    assert int(got[0, 1]) == eos, "row 0 should still emit the EOS token itself"
    assert (got[0, 2:] == PAD_ID).all(), (
        f"row 0 must be padded after EOS, got {got[0].tolist()}")
    # row 0 stopping must not truncate row 1
    assert torch.equal(got[1, :2], tail[1, :2]), "row 1 changed when row 0 stopped"
