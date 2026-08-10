# Findings

Running log. Every gate result and every deviation from upstream goes here, whether it flatters the project or not.

---

## Environment (verified 2026-08-09, dev laptop, CPU)

The plan's Task 1 pin resolves. Verified installed set:

```
python 3.13 (/usr/local/bin/python3.13)
torch        2.8.0
transformers 4.53.3
```

`python3.11` was not on the host; `python3.13` worked without needing the conda fallback. The GPU box must use the same combination.

---

## Pre-implementation measurements

Both were run before any of `lkv/` existed, on a randomly-initialized 4-layer Qwen3 (`vocab 256, hidden 64, 8 query heads, 2 KV heads, head_dim 8, untied embeddings`), CPU, float32, `attn_implementation="sdpa"`. Random weights are adequate because both claims are about masks and indices, not learned behaviour.

### D1 — right padding destroys batched generation. CONFIRMED.

Method: greedily continue the 3-token prompt `[5, 6, 7]` for 6 tokens, three ways — alone, left-padded inside a batch with a 7-token sequence, and right-padded inside the same batch.

| condition | continuation | matches unbatched |
|---|---|---|
| unbatched | `[110, 125, 188, 119, 110, 162]` | — |
| left padding | `[110, 125, 188, 119, 110, 162]` | **yes** |
| right padding | `[239, 188, 239, 188, 125, 53]` | **no** |

transformers emits its own warning on the right-padded call: "A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'`."

**Consequence for upstream LatentMAS.** `stock-latentmas` never sets `padding_side` (`models.py:15-20` sets only `pad_token`), so the tokenizer default applies — right, for both Qwen and Llama. Every row shorter than its batch maximum therefore generates from a PAD token, and the output slice at `models.py:230,260-267` compounds it by using each row's *unpadded* length. At the repo default `generate_bs=20`, on a length-heterogeneous task like MBPP+, this affects most items in most batches. First thing to fix, and plausibly the largest single source of any reproduction gap.

### D8 — attention is not causal under a layer-trimmed cache. CONFIRMED.

Method: prefill 12 context tokens, apply KVComm's `prepare_key_cache` semantics (selected layers whole, others truncated to position 0, layer 0 always whole), then run 4 receiver tokens twice, changing **only the last one**. If the *first* position's logits move, information flowed backwards.

| condition | cache lengths | first-position logit drift | verdict |
|---|---|---|---|
| untrimmed control | `[12, 12, 12, 12]` | `0.000e+00` | causal |
| trimmed, keep layer 2 | `[12, 1, 12, 1]` | `1.940e-01` | **not causal** |

**Mechanism.** `DynamicCache.get_seq_length()` reads layer 0, and KVComm forces layer 0 to be retained in full (`stock-kvcomm/models.py:101`, `if i in self.layers_list or i == 0`). The causal mask is therefore built at the full length `N+q`, and sdpa slices it down to each layer's actual key length. For a layer holding `1+q` positions, the first `1+q` columns of that mask are all-visible to every query row, so the receiver's own tokens see each other in both directions.

**Consequences.**

1. KVComm's default path (`shift_back=False`) is affected, and that is the path its README commands use. Their published numbers were produced with non-causal attention in every trimmed layer. Their opt-in `shift_back` implementation (`models.py:253-359`) exists to correct positions and asserts at most two distinct cache lengths (`:240-249`) — consistent with this being a known-difficult area for them.
2. Our reproduction must state which path it used. Reproducing their table may *require* the uncorrected path; matching their numbers with a corrected forward would be a different measurement, and the gap between the two is itself a result worth reporting.
3. Any LatentMAS integration must use corrected per-layer masks, because the relay has three boundaries rather than one, so the error would compound.

The plan's `lkv/kvcomm/trim.py:build_per_layer_masks` and `forward_with_ragged_cache` (Task 12) address this. Task 15 step 3 item 3 compares the two paths on real weights.

### D8 (fix verified) — the per-layer-mask correction is exactly causal.

Task 12 implemented `build_per_layer_masks` and `forward_with_ragged_cache`
(`lkv/kvcomm/trim.py`), which give every decoder layer its own additive mask
sized to that layer's own key length, and its own `position_embeddings`
computed from that layer's own cache length, instead of the one shared
mask/position pair the stock `Qwen3Model.forward` loop builds from layer 0's
length (`transformers/models/qwen3/modeling_qwen3.py:450-468` at 4.53.3).

Re-running the D8 measurement (`tests/test_ragged_cache.py`) through the fixed
path instead of the stock path, on the same `[12, 1, 12, 1]` cache:

| path | first-position logit drift |
|---|---|
| stock (D8, re-measured in-suite) | `1.5e-01`–`2.1e-01`, always > `1e-3` |
| `forward_with_ragged_cache` | `0.000e+00`, exactly, every run |

The stock-path number is not a single fixed value: `tests/test_ragged_cache.py`
seeds the model once per session but draws the receiver-token ids with an
unseeded `torch.randint`, so the exact drift depends on how much of the global
RNG stream earlier tests in the same process consumed. Isolated, it reproduces
at `1.537e-01`; inside the full two-file run it was `2.075e-01`. The magnitude
moves, the qualitative result does not: every observed run is non-causal by at
least two orders of magnitude above the `1e-3` threshold, and
`forward_with_ragged_cache`'s drift is `0.0` to the bit, every run, because the
receiver block's mask column for a future token is a literal `-inf`, not an
approximation. `test_stock_path_is_NOT_causal_when_trimmed` pins the defect
range rather than a single digit for exactly this reason.

**Consequence.** D8's open question #2 is resolved for Phase 4: KVComm
reproduction work here must run through `forward_with_ragged_cache`, not the
stock forward, whenever a trimmed cache is used with more than one distinct
cache length. The two-path comparison upstream's `shift_back` flag gestures at
is now a real, measured gap (D8 vs. D9's tables above), not a hypothesis.

### D8 — compliance gap in Task 14's pair harness. NOT YET CLOSED.

Task 14 (`lkv/kvcomm/pair.py`) builds `run_pair`/`receiver_logits` on
`generate_text_batch` and a plain model call, i.e. the stock forward path, not
`forward_with_ragged_cache`. Whenever `layers_list` passed to `run_pair` or
`receiver_logits` is a strict subset of all layers (the entire point of
`top_layers < 1.0`), `prepare_key_cache` returns a ragged cache and this
requirement is violated: the run is subject to the same non-causal attention
D8 measured above.

This is not a new defect distinct from D8 -- it is the stated requirement
above not yet being implemented. `forward_with_ragged_cache` computes exactly
one forward pass; `run_pair` needs autoregressive generation (up to
`cfg.max_new_tokens` steps) over a batched, left-padded receiver prompt.
Extending `forward_with_ragged_cache` into a decode loop with that shape is
new work, not a fix to something Task 14 broke, so it was flagged rather than
built silently (see `task-14-report.md`). Calibration is unaffected: it always
runs at full retention, so `calibrate`'s cache is never ragged.

**Consequence.** Any Phase 4 run (Task 15/16) with `top_layers < 1.0` is
currently measuring the stock (non-causal) path, not the fix. This must be
resolved -- either by building the decode loop or by an explicit, recorded
decision to reproduce upstream's own uncorrected path on purpose (D8's
consequence #2 above) -- before any such number is reported as a KVComm
measurement rather than a known-non-causal baseline.

### D8 — compliance gap continues into Task 17's LatentMAS trim hook. NOT YET CLOSED.

Predicted by `task-14-report.md` ("`run_latentmas` has this exact same shape of
limitation once Task 17 wires real KVComm trimming into it") and now confirmed.
`lkv/integrate.py`'s `make_trim_hook` returns exactly the same kind of ragged
`DynamicCache` as `prepare_key_cache`/`trim_delta` whenever `top_layers < 1.0`
in either scope. `run_latentmas` (`lkv/latentmas.py`) hands that cache straight
into `generate_latent`'s plain `model(...)` call for every later latent hop,
and into `generate_text_batch` for the judger -- both the stock forward path,
never `forward_with_ragged_cache`. This is the SAME gap as the section above,
one call site deeper, and D8's own consequence #3 (recorded when
`forward_with_ragged_cache` was built, before any integration existed) named it
in advance: *"Any LatentMAS integration must use corrected per-layer masks,
because the relay has three boundaries rather than one, so the error would
compound."* It has not been built; the relay now has up to three boundaries at
which this can compound, not KVComm's original one.

`lkv/latentmas.py` cannot import `lkv/kvcomm/` (Task 17's own code-organization
rule: `lkv/integrate.py` is the only module allowed to depend on both), so
closing this gap is not a small edit inside the hook.

**This is not the same gap Task 21 (Phase 6, already planned) closes.** Task 21
builds `generate_with_ragged_cache` and wires it into `run_pair` via a
`causal_ragged` flag -- its own Files/Interfaces list only
`lkv/kvcomm/ragged_decode.py`, `lkv/kvcomm/pair.py`, `lkv/config.py`. Nothing in
Task 21 as planned touches `lkv/latentmas.py` or `lkv/integrate.py`, so even
after Task 21 lands, `run_pair`'s causality gap is closed while
`run_latentmas`'s (this section's) is not. `generate_with_ragged_cache` is
declared "signature-compatible with `generate_text_batch`" specifically so
`run_pair` can switch on one flag -- the same compatibility would let a future
task thread it into `generate_latent`'s per-step calls and into
`run_latentmas`'s judger call, but that task does not exist yet in the plan.

**Consequence.** Any Phase 5 sweep (Task 20) with `top_layers < 1.0`, in either
`trim_scope`, is currently measuring the stock (non-causal) path at every
boundary after the first trim -- not a corrected KVComm-in-LatentMAS
measurement, and NOT something Task 21 as currently scoped will fix.
`trim_scope="all"` vs `"delta"` and `every_hop` can still be compared against
EACH OTHER on this shared, uncorrected footing, but neither should be reported
as what a causally-correct trim would produce until a Task 21-equivalent
extension exists for `run_latentmas`. `top_layers=1.0` (no trim built at all)
is unaffected.

---

## Not yet measured

Everything below needs GPU and is not yet known. Do not cite any number for these until it appears in this file.

- L1 — GSM8K 8B latent-step sweep against the published 93.8 / 81.1.
- L2 — MBPP+ 8B against 74.6 / 64.8, and 4B / 14B GSM8K.
- K1 — KVComm Countries against 0.00 / 0.62 / 0.51 / 0.62 / 0.62, plus the D10 metric decision (mean F1 versus fraction with F1 > 0.5).
- K2 — KVComm HotpotQA and Tipsheets.
- Phase 5 — the integration sweep across `trim_scope` and retention ratio.

The correct `latent_steps` value is not recorded anywhere in the upstream repo: its argparse default is `0` (degenerate — the relay cache is discarded), and the two shipped example logs show `0` and `10`, neither of which matches the table cell it accompanies. It has to be searched over `[0, 80]`.

### D12 — a cached span's mask must be the real historical mask. CONFIRMED.

Discovered during implementation of Task 3, in a fix that itself synthesized
`torch.ones` for the cached span.

When a KV cache is built from a **left-padded ragged batch** — which every
LatentMAS relay batch and every KVComm sender batch is — the pad positions were
correctly masked at write time. Any later call that hands `generate` an all-ones
mask over the cached span re-enables attention to those pad key/value vectors.

Measured on transformers 4.53.3 with the tiny fixture, a 5-real-token prefix
left-padded by 2, then a 3-token prompt:

| mask over cached span | greedy continuation |
|---|---|
| true historical mask (reference) | `[188, 178, 229, 81, 210, 79]` |
| synthesized all-ones | `[188, 236, 188, 236, 159, 180]` |

Divergence held across randomized trials at pad counts 1–3 and multiple seeds.
A raw forward pass put the max logit difference at ≈0.53, comparable to a typical
top-1/top-2 gap — so this is not numerically inert. Note the contrast with the
`input_ids` filler for the same span, which **is** provably inert because
`generate` never re-embeds it.

**Consequence.** `generate_text_batch` and `generate_latent` now both *require* a
`past_mask` whenever the cache is non-empty, and refuse rather than guess. The
relay threads one running mask across all four agents; a trim re-cuts it to the
surviving prefix. Upstream LatentMAS synthesizes the prepended block as ones
(`stock-latentmas/models.py:374-382`), so it has this defect too, compounding
with D1.

### D13 / D14 — two harness-fidelity gaps found while reviewing Task 7

Neither is an upstream *defect* — both are places where our reimplementation
diverged from what KVComm actually measures. Caught by review before any GPU
time was spent. Verified against the upstream clone, not inferred.

**D13 — the HotpotQA sender context is supporting-fact sentences only.**
Upstream's `construct_support` (`stock-kvcomm/dataloader/hotpotqa.py:4-14`) walks
`supporting_facts` and pulls exactly those sentences out of the matching context
paragraphs, then joins with newlines to form `prompt_A` (`:34-35`). Our first
implementation joined **all ten distractor paragraphs**. That context is roughly
an order of magnitude larger and mostly irrelevant, so every KVComm score would
have come out low — and it would have looked like the method failing rather than
the harness being wrong. Fixed in the plan and in `lkv/data/hotpotqa.py`.

**D14 — KVComm's F1 lemmatizes both sides before comparing.**
`stock-kvcomm/utils/f1.py:12-13` applies `lemmatize_text(normalize_answer(x))` to
reference and candidate, using nltk's WordNetLemmatizer with a per-token POS tag
(`utils/metric_tools.py:188-210`). Our F1 normalized but did not lemmatize, which
depresses F1 systematically across every condition — enough to make the K1/K2
gates miss for a reason unrelated to KV transfer. Added `lkv/lemma.py` and
`nltk`, and it raises rather than silently degrading to unlemmatized text.

Also confirmed harmless, so recorded to avoid re-litigation: upstream's `precision`
and `recall` are transposed relative to the usual convention (`utils/f1.py:18-27`
divides by the reference and candidate set sizes respectively). F1 is symmetric in
the two, so the resulting value is identical. No change needed.

### D15 — the traced query must be post-RoPE. CONFIRMED against upstream.

Found by review during Task 12; the defect was in this project's plan, not in
upstream.

Our tracer stashed the query straight out of `q_proj` (plus `q_norm`), with **no
rotary embedding applied**, while the key was read back from the KV cache where
RoPE had already been applied. Query and key therefore lived in different rotary
bases, making every recomputed `Q·Kᵀ` meaningless — and since those attention
weights *are* the layer-importance statistic, the entire KVComm ranking would
have been scored on noise. It would not have crashed or looked wrong; it would
have produced a plausible ranking that meant nothing.

Upstream gets this right by construction: `stock-kvcomm/model_attn.py:37-40`
projects Q, K, V and then calls `apply_rotary_pos_emb` **before** stashing
`query_states` and `key_states` at `:50-58`. It overrides the attention forward
wholesale rather than wrapping it, so the post-RoPE tensors are simply in scope.

Our wrapper design cannot see upstream's local variables, so the tracer now
applies RoPE itself using the `position_embeddings` (cos, sin) passed into the
forward, resolving `apply_rotary_pos_emb` dynamically from the attention
module's own modeling file — Qwen3 and Llama each ship an identical copy, and
hard-coding one family's import path would break the other.

Recorded because this class of bug is invisible downstream: nothing about a
wrongly-rotated query produces an error, only worse numbers.
