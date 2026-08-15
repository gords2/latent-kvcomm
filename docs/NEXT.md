# Where to pick up

Read this first. It is the handoff for the GPU phase.

**State as of 2026-08-15:** all CPU-testable work is done, reviewed, merged to `main`, and pushed. Suite is **158/158** green with no GPU or network required. Nothing is half-finished — the remaining tasks were deliberately deferred because they need a GPU.

Two things changed after this file was first written:

- **Task 21's CPU half landed.** The causal ragged decoder exists and is wired to *both* call sites behind `--causal_ragged`, off by default. Only its Step 5 measurement still needs a GPU.
- **D16 was found and fixed.** The relay assigned positions with a bare arange, so left padding between hops counted as real distance and an item's own representation depended on the length of its batch peers — `--generate_bs` was silently an experimental variable. Read D16 in `docs/FINDINGS.md` before running anything.

- The spec is `docs/superpowers/plans/2026-08-09-latentmas-kvcomm.md` (21 tasks).
- Measured defects and deviations are in `docs/FINDINGS.md`. **Read it before citing any number.**
- The per-task process log is `.superpowers/sdd/2026-08-09-latentmas-kvcomm/progress.md` (317 lines, git-ignored, local only). It holds every parked ruling.

---

## First action tomorrow

You need a GPU box. Nothing below can start without one.

1. The user creates the droplet and supplies the IP. Claude cannot create or destroy droplets, and must not handle API tokens.
2. The user runs `hf auth login` themselves. **Never paste an HF token into chat.**
3. Then the prerequisites below, then Task 10.

Budget expectation: setup is dominated by downloading ~51 GB of weights. Tasks 10 and 11 together are roughly 2–3 GPU hours; the full remaining set roughly 8–12.

---

## Prerequisites on the GPU box

Three things are missing that the CPU phase did not need.

**1. Environment.** Target Python 3.11 (the dev laptop used 3.13 because 3.11 was absent; either works, but keep one).

```bash
PY=python3.11 ./scripts/setup_env.sh
```

The `transformers==4.53.3` pin is load-bearing — KVComm's own requirements say "Must be 4.53.3, model definition changed in 4.54.0". Never bump it to silence a warning.

**2. nltk corpora.** KVComm's F1 lemmatizes both sides (see D14). `lkv/lemma.py` probes capabilities functionally, downloads only what is missing, and raises rather than silently scoring without lemmatization. Pre-seed so the first gate run does not stall:

```bash
./.venv/bin/python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng
./.venv/bin/python -c "from lkv.lemma import lemmatize_text; print(lemmatize_text('cities were running faster mice'))"
# must print: city be run faster mouse
```

**3. Two bundled datasets that are not on the Hub.** `lkv/data/files/` does not exist yet. Copy from a KVComm clone:

```bash
git clone --depth 1 https://github.com/Zephyroam/KVComm.git /tmp/kvcomm
mkdir -p lkv/data/files
cp /tmp/kvcomm/dataloader/data/countries.jsonl /tmp/kvcomm/dataloader/data/tipsheets.jsonl lkv/data/files/
```

`lkv/data/countries.py` raises `FileNotFoundError` if absent — deliberately, because returning `[]` would report accuracy 0.0 as though it were a measurement. Note `tipsheets` is **not yet registered** in `lkv/data/__init__.py`; registering it is part of Task 16.

---

## Task order

Run in this order. Each gate must pass before the next phase means anything.

| Task | What it does | Gate |
|---|---|---|
| **10** | GSM8K 8B sweep over `latent_steps` ∈ {0,10,20,40,80}, plus single-agent. Writes `scripts/gate_latentmas.sh`. | **L1**: some `latent_steps` ≥ 0.91 at n=200, **and** single-agent within [0.76, 0.86]. Targets 93.8 / 81.1. |
| **11** | MBPP+ 8B at 378 items, plus 4B and 14B GSM8K. Uses the winning `latent_steps` as `S_BEST`. | **L2**: MBPP+ ≥ 0.70, **and** relay > single-agent by paired McNemar. Targets 74.6 / 64.8. |
| **15** | KVComm Countries (200 items — cheapest, widest dynamic range). Also **decides the metric**: mean F1 vs fraction with F1 > 0.5. Writes `scripts/gate_kvcomm.sh`. | **K1**: baseline ≤ 0.05, skyline ∈ [0.55, 0.69], r30 ≥ 0.42, r50 ≥ 0.55, monotone r30 ≤ r50 ≤ r70 (0.02 tolerance). |
| **16** | KVComm HotpotQA (500) and Tipsheets (500). Registers the tipsheets loader. | **K2**: baseline ∈ [0.13, 0.25], skyline ≥ 0.66, r30 ≥ 0.26, r50 ≥ 0.53, r70 ≥ 0.62, **strictly** increasing. |
| **20** | Integration sweep: 2 scopes × 3 ratios, plus random-selection and pair-calibration controls. Run every trimmed condition **twice**, `--causal_ragged` off and on. | — |
| **21** | ~~Causal multi-step decoding, both call sites.~~ **Code and tests are DONE** (`lkv/kvcomm/ragged_decode.py`, 14 tests). Only **Step 5** remains: re-run the K1 Countries conditions at 0.3/0.5/0.7 with `--causal_ragged` and record `ratio \| stock \| corrected \| delta` against the published 0.51/0.62/0.62. | Untrimmed path matches `generate_text_batch` exactly — already verified on CPU. |

Start Task 20 only once L1, L2, K1 and K2 have all passed. If a gate fails, its diagnostic ladder is in the task text — work it in order and record each outcome in `docs/FINDINGS.md`.

Task 21's Step 5 write-up must state the interpretation plainly whichever way it falls. If the corrected path scores **higher**, KVComm's published numbers understate the method. If **lower**, their numbers depend on the bug — the more uncomfortable finding, and it must be reported, not buried. If the two are within noise, say so: that would mean D8 is real but immaterial at these retention ratios, which is also worth knowing.

---

## Rulings that bind — do not re-litigate

Decided with evidence. Re-opening them wastes GPU time.

1. **Task 20's analyzer must not use the plan's literal `or` form.** Write the retention lookup as `1.0 if d.get("layer_retention") is None else d["layer_retention"]`. The plan snippet says `(d.get('layer_retention') or 1.0)`, which treats a legitimate `0.0` as missing and would print full retention for a run that achieved zero. Unreachable in the planned sweep, but the snippet gets copied verbatim if this is not read.
2. **`latent_steps` has no default and must be searched.** Upstream's default is `0`, which discards the relay cache. The value behind the published numbers is recorded nowhere; the two shipped example logs show `0` and `10`, neither matching the table cell it accompanies.
3. **`--calib_source` defaults to `relay`, which is *not* KVComm-as-published.** Deliberate. `--calib_source pair` gives the drop-in comparison and Task 20 reports both. `pair` is rejected at parse time for GSM8K/MBPP+ because their items carry no `context`.
4. **The verbatim upstream reference block in `tests/test_importance.py` stays.** Approved differential-testing choice. Do not deduplicate it.
5. **Reproduction uses sampled decoding; internal A/B uses `--greedy`.** Never mix the two inside one comparison.
6. **Regenerate a task brief after any plan amendment** — `scripts/task-brief PLAN N` from the SDD skill. Two implementers worked from stale briefs before this rule was written down.

---

## Hazards that will bite

**D8 — the fix now exists, but it is OFF by default.** `--causal_ragged` decodes a trimmed cache with per-layer causal masks, at both call sites. Default off is deliberate: KVComm's published table was produced on the non-causal path, so reproducing *their* numbers requires it. For the integration there is no upstream to be faithful to and the error compounds over three boundaries, so **every integration number must be produced with `--causal_ragged`**, or Task 20 must report both columns with the stock one labelled non-causal. `build_record` stamps `ragged_path` (`n/a` / `stock-noncausal` / `causal-ragged`) so this travels with the result files, and `parse_args` rejects the flag for systems that never trim, so the label cannot claim a correction that never applied.

Measured on the tiny fixture, one relay hop over a ragged cache: stock vs causal = **4.333** max delta. Whatever D8 is worth at scale, it is not nothing.

**D16 is fixed but its lesson stands.** The relay was reproducible only by accident: an item's representation depended on the length of its batch peers, so `--generate_bs` and item order silently changed results. Fixed and pinned by a test. The lesson is that `lkv` calls the model directly in several places, and a bare `model(...)` does **not** get `generate()`'s position handling. Any new direct forward needs explicit `position_ids` — see `lkv/latentmas.cumsum_positions`.

**The failure mode of this codebase is silent wrongness, not crashes.** Fourteen defects found during the CPU phase were functions that ran fine and returned plausible, meaningless numbers — a query missing its rotary embedding, a scope that stopped trimming after one hop, a validity test that passed regardless of correctness. A green suite is weak evidence. When a gate result looks clean, ask what would have to be true for it to be wrong.

**Two parked minors, both narrow:** `ragged_path` and `layer_retention` can contradict each other under `--system latentmas --n_agents 1 --calib_source pair` on a context task; and the analyzer `or` gotcha in ruling 1. Neither is reachable from the planned sweep.

---

## Already true — needs no re-checking

Verified mechanically, with evidence in `docs/FINDINGS.md` and the ledger:

- LatentMAS: 4 roles in order, question re-supplied as text to all 4, latents skipped for the judger, relay cache never nulled, latent input is the last hidden state fed back as `inputs_embeds`.
- KVComm: ranking **bit-identical** to the upstream formula across α ∈ {1.0, 0.8, 0.5, 0.0}; retained counts `[9, 16, 22]` at 0.3/0.5/0.7 on 32 layers; trim keeps layer 0 full and others at `sink=1`; full selection is the identity.
- Defaults reproduce upstream: `realign=False, renorm_latent=True`, `n_agents=4`, `temperature=0.6`, `top_p=0.95`, `sequential`, `seed=42`, `mu=0.5`, `sigma=10.0`, `calib_size=1`, α per family (Llama 1.0, Qwen/Falcon 0.8).
- No cross-task interface drift across the 15 implemented tasks.

- Task 21's decoder: reproduces `generate_text_batch` token for token on an untrimmed cache at batch 2 with padding on both sides; causal (drift exactly 0.0) after 0, 3 and 7 appended tokens; batch-invariant under a padded history; per-row EOS stops one row without truncating its neighbour. Verified by mutation — reverting positions to arange, ignoring the query block's padding, broadcasting the mask from row 0, and rebuilding masks from current instead of base lengths are each caught.

**Not verified:** that any of it reproduces the published accuracy. That is exactly what Tasks 10/11/15/16 establish, and no GPU has run.
