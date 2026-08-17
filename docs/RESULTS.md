# Results

Everything measured, in one place. Sources: `runs/*.json` (51 records, all
committed). Method and defects: `docs/FINDINGS.md`. **Read the caveats — several
numbers mean less than they look like they mean.**

Hardware: AMD MI300X (ROCm 6.4), `transformers==4.53.3`, torch 2.8.0+rocm6.4.
Suite: 173 tests, CPU-only, no network.

---

## 1. LatentMAS reproduction (L1 / L2)

Qwen3-8B, sampled decoding (temperature 0.6, top_p 0.95) as the paper used.
McNemar is paired on the same seeded slice.

| task | thinking | relay | single | delta | p | |
|---|---|---|---|---|---|---|
| GSM8K n=200 | on | 0.9500 | 0.9600 | −0.0100 | 0.6875 | ns |
| GSM8K n=200 | off | 0.9150 | 0.8900 | +0.0250 | 0.3018 | ns |
| MBPP+ n=378 | on | 0.7487 | 0.7566 | −0.0079 | 0.7201 | ns |
| **MBPP+ n=378** | **off** | **0.7302** | **0.6931** | **+0.0370** | **0.0201** | **significant** |

Paper targets: GSM8K relay 0.938 / single 0.811; MBPP+ relay 0.746 / single 0.648.

**The paper's relay numbers reproduce; its baselines do not.** 0.9500 vs 0.938
and 0.7487 vs 0.746 are close. Our single-agent baselines are 11–15 points
stronger than published, which is what erases the advantage.

**LatentMAS beats one agent in exactly one of four cells** — MBPP+ with thinking
disabled, +3.70 points, 23 items won against 9 lost. Everywhere else the
difference is indistinguishable from noise. The pattern is consistent with
saturation: GSM8K sits at 0.89–0.96 for both arms, leaving nothing to win.

`latent_steps` was swept over {0, 10, 20, 40, 80} on GSM8K: 0.950 / 0.910 /
0.945 / 0.925 / 0.940. Flat within 1–2 SE — no evidence any value helps. Also
`lm_gsm8k_4b` 0.9250 and `lm_gsm8k_14b` 0.9300.

---

## 2. KVComm reproduction (K1 / K2)

`NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror), greedy,
`--causal_ragged`, `calib_size=1`, `alpha=1.0`. Metric is the fraction with
F1 > 0.5, matching upstream's `f1_match` threshold.

| task | baseline | skyline | r30 | r50 | r70 | rand50 |
|---|---|---|---|---|---|---|
| Countries n=200 | 0.000 | 0.635 | 0.315 | 0.615 | 0.620 | 0.305 |
| *published* | *0.00* | *0.62* | *0.51* | *0.62* | *0.62* | — |
| HotpotQA n=500 | 0.248 | 0.766 | 0.264 | 0.532 | 0.598 | 0.450 |
| Tipsheets n=500 | 0.008 | 0.928 | **0.930** | **0.930** | 0.908 | 0.790 |

**Tipsheets is the cleanest reproduction of KVComm's core claim**: at 0.28 layer
retention the receiver matches its full-context skyline (0.930 vs 0.928) while
the no-context baseline sits at 0.008. Roughly 72% of KV layers are never
transmitted and nothing is lost.

**The importance ranking beats random selection on every task** — 0.615 vs 0.305,
0.532 vs 0.450, 0.930 vs 0.790 — which is the specific claim KVComm makes for
layer selection.

Gate status: **K1 passes 4 of 5** (only `r30 ≥ 0.42` fails, at 0.315);
**K2 HotpotQA passes 5 of 6** (only `r70 ≥ 0.62` fails, at 0.598).

**r30 on Countries is an open discrepancy with no identified cause.** Two
hypotheses were tested and rejected: calibration-set size (0.315/0.345/0.315 at
`calib_size` 1/4/16 — the ranking is stable) and layer-0 accounting (already
matches upstream). Since r50 and r70 land within 0.005 of published, the
transmission path is not in question. It was left open rather than tuned shut:
fitting a knob until 0.51 appears would void the reproduction.

---

## 3. D8 — the causal correction is load-bearing

Countries, n=200, greedy. Same layer selection; only the decode path differs.

| retention | stock (non-causal) | causal-ragged | published |
|---|---|---|---|
| 0.3 | 0.030 | 0.315 | 0.51 |
| 0.5 | 0.150 | 0.615 | 0.62 |
| 0.7 | 0.155 | 0.620 | 0.62 |

**This reversed a standing project ruling.** The repo previously asserted that
the stock path reproduced KVComm as published. It reproduces nothing; the
corrected path lands on two of three published cells. `--causal_ragged` is now
the default in `scripts/gate_kvcomm.sh`.

---

## 4. The integration (Phase 5)

MBPP+ n=378, Qwen3-8B, `--greedy --no_thinking --causal_ragged`,
`latent_steps=0`, `max_new_tokens=640`. Every condition including both
references shares those settings.

| condition | accuracy | layer_ret | bytes/ref |
|---|---|---|---|
| relay, untrimmed | **0.7196** | 1.000 | 1.000 |
| single agent | **0.7116** | n/a | n/a |
| all, r70 | 0.5661 | 0.723 | 0.723 |
| all, r50 | 0.5000 | 0.529 | 0.528 |
| all, r30 | 0.3386 | 0.307 | 0.306 |
| delta, r70 | 0.5794 | 0.723 | 0.723 |
| delta, r50 | 0.5053 | 0.529 | 0.529 |
| delta, r30 | 0.3333 | 0.308 | 0.308 |
| r50, one-shot | 0.6455 | 0.713 | 0.844 |
| r50, random layers | 0.3651 | 0.501 | 0.501 |

| comparison | delta | p | |
|---|---|---|---|
| untrimmed vs trimmed r50 | +0.2196 | 4.7e-20 | significant |
| trimmed r50 vs single | −0.2116 | 3.1e-17 | significant |
| ranked r50 vs random r50 | +0.1349 | 5.0e-07 | significant |
| one-shot vs every-hop r50 | +0.1455 | 1.2e-09 | significant |
| scope all vs delta r50 | −0.0053 | 0.73 | ns |
| untrimmed relay vs single | +0.0079 | 0.72 | ns |

**The integration does not pay off.** At r50 it saves 47% of relay cache bytes
and loses 22 accuracy points. The trimmed relay (0.5000) is beaten by a single
agent carrying no relay cache at all (0.7116). No ratio tested reaches the
single-agent score.

Three secondary results are solid and more informative than the headline:

1. **The layer ranking transfers to the relay** — ranked beats random by 13.5
   points at identical retention (p=5.0e-07). The selection mechanism works;
   the setting it is applied in is what fails.
2. **Compounding is the dominant cost, not trimming.** Trimming once scores
   0.6455 against 0.5000 for trimming at every boundary (p=1.2e-09) — about two
   thirds of the damage comes from repetition.
3. **`trim_scope` all vs delta is indistinguishable** (p=0.73) at every ratio.
   The knob can be dropped.

This is coherent with §2 rather than contradicting it: one-shot cache transfer
tolerates heavy trimming (Tipsheets at 0.28 retention loses nothing), a
multi-hop relay does not, because each hop re-trims a cache that already lost
information at the previous hop.

**The load-bearing caveat.** Under greedy decoding the untrimmed relay beats one
agent by only +0.0079 (ns). The significant +0.0370 exists only under sampled
decoding, and internal A/B rules forbid mixing the two. So §4 measures the cost
of trimming a relay that, in this decoding regime, was not buying anything. A
sampled-decoding sweep is the obvious follow-up and has not been run.

---

## What would need doing next

1. **Sampled-decoding integration sweep** — the one configuration where a relay
   advantage exists to protect. Requires re-deriving the references under
   sampling.
2. **Countries r30** — the open discrepancy; the remaining candidate is that the
   Nous mirror differs from the weights KVComm used.
3. **HotpotQA r70** (0.598 vs 0.62) — the other missed threshold.
4. **A cheaper causal decoder.** `ragged_hidden` is ~27x slower than stock
   generate; memoising by cache length already bought 3.35x (718s → 214s per
   MBPP+ batch). The remaining cost is the per-layer Python loop.
5. **Tipsheets / HotpotQA integration sweeps** — both carry a `context` field,
   so unlike GSM8K and MBPP+ they can run the `--calib_source pair` control.
