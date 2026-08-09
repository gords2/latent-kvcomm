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

---

## Not yet measured

Everything below needs GPU and is not yet known. Do not cite any number for these until it appears in this file.

- L1 — GSM8K 8B latent-step sweep against the published 93.8 / 81.1.
- L2 — MBPP+ 8B against 74.6 / 64.8, and 4B / 14B GSM8K.
- K1 — KVComm Countries against 0.00 / 0.62 / 0.51 / 0.62 / 0.62, plus the D10 metric decision (mean F1 versus fraction with F1 > 0.5).
- K2 — KVComm HotpotQA and Tipsheets.
- Phase 5 — the integration sweep across `trim_scope` and retention ratio.

The correct `latent_steps` value is not recorded anywhere in the upstream repo: its argparse default is `0` (degenerate — the relay cache is discarded), and the two shipped example logs show `0` and `10`, neither of which matches the table cell it accompanies. It has to be searched over `[0, 80]`.
