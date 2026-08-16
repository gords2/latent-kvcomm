# Where to pick up

Read this first. It is the handoff for the GPU phase.

**State as of 2026-08-16, ~01:30.** The first GPU session happened. The
environment is solved, both LatentMAS gates ran, and **K1 is broken in a way that
must be fixed before any KVComm number means anything**. All 18 run records are
committed under `runs/`. Both droplets were destroyed.

- Suite: **165/165** green on CPU, no GPU or network needed.
- Measured defects and deviations: `docs/FINDINGS.md`. **Read before citing any number.**
- The spec: `docs/superpowers/plans/2026-08-09-latentmas-kvcomm.md` (21 tasks).
- Per-task process log: `.superpowers/sdd/2026-08-09-latentmas-kvcomm/progress.md` (git-ignored, local only).

---

## First action tomorrow

1. **The user creates a GPU droplet and supplies the IP.** Claude cannot create
   or destroy droplets and must not handle API tokens.
2. **No Hugging Face account is needed.** This was wrong yesterday and cost time:
   the Qwen3 repos are Apache-2.0 and download anonymously; only `meta-llama/*`
   is gated. `NousResearch/Meta-Llama-3.1-8B-Instruct` is in `MODEL_CHOICES` and
   is what the KVComm gates use. **Never ask the user to `hf auth login`.**
3. Then the setup below, then **Task 0**, which blocks everything else.

Budget: the whole remaining set is roughly 3–5 GPU hours if nothing else breaks.

---

## Setup on a fresh MI300X box — known good, follow exactly

The first SSH lands in `Please wait while we get your droplet ready...` and the
connection closes. That is DigitalOcean's first-boot script, not a fault. Poll
until a real shell answers; it took ~9 minutes.

**Do not infer missing hardware from a missing tool.** Yesterday `nvidia-smi:
command not found` plus an `lspci` that was not installed produced a confident,
wrong "this droplet has no GPU" — on a working MI300X. The definitive test is the
ROCm device node:

```bash
ssh root@<IP> 'ls -l /dev/kfd && lsmod | grep amdgpu && rocm-smi --showproductname'
```

**Ship the repo with rsync, not a clone** (it is private, and this avoids any
credential handling):

```bash
rsync -az --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  ./ root@<IP>:/root/latent-kvcomm/
```

**Environment.** Python 3.11 is absent on this image, and 3.12 lacks `ensurepip`,
so the venv fails with a misleading message unless the package is installed
first. `rocm6.4` is the **only** wheel index carrying a torch inside
`requirements.txt`'s `>=2.4,<2.9` pin — rocm7.x jumps to 2.10+.

```bash
apt-get update -qq && apt-get install -y -qq python3.12-venv
cd /root/latent-kvcomm
PY=python3 TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.4 ./scripts/setup_env.sh
```

Expect `torch 2.8.0+rocm6.4 | ROCm/HIP 6.4`, `accelerator available: True`,
`[0] AMD Instinct MI300X VF 192 GiB`. No `lkv` code change is needed for AMD —
torch's ROCm build maps the CUDA API onto HIP, so `--device cuda:0` and the
`torch.cuda.*` calls in `lkv/instrument.py` work unmodified.

**nltk corpora** (F1 lemmatises both sides, see D14):

```bash
./.venv/bin/python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng
./.venv/bin/python -c "from lkv.lemma import lemmatize_text; print(lemmatize_text('cities were running faster mice'))"
# must print: city be run faster mouse
```

**Two bundled datasets** (not on the Hub; `lkv/data/files/` is absent by design):

```bash
git clone --depth 1 https://github.com/Zephyroam/KVComm.git /tmp/kvcomm
mkdir -p lkv/data/files
cp /tmp/kvcomm/dataloader/data/countries.jsonl /tmp/kvcomm/dataloader/data/tipsheets.jsonl lkv/data/files/
```

**Verify before running anything:** `./.venv/bin/python -m pytest -q` → 165 passed.

---

## Task 0 — FIX K1 BEFORE ANY OTHER GPU WORK

**All six Countries conditions scored exactly 0.0000, skyline included.** A
skyline run is handed the full context *and* the question, so zero is impossible
unless the harness is at fault. HotpotQA's baseline came in at 0.0020, so it is
probably the same defect. Every KVComm number from yesterday is void.

**Step 1 — store the predictions. DONE, no GPU needed.** `per_item` now carries
`prediction`, `prediction_chars` and `gold` alongside `correct`/`i`/`score`
(`lkv/run.py`). Scoring still runs on the full string; only the stored copy is
cut, middle-elided to 600 chars a side behind a visible `...[N chars elided]...`
marker — GSM8K reads `numbers[-1]` and MBPP+ takes the LAST ```python block, so
head-only storage would have hidden both answers. `prediction_chars` is the
**true** pre-elision length: that is the field separating "the model emitted
nothing" from "the model ran into `max_new_tokens` mid-answer". Countries at 5
tokens and HotpotQA at 48 are stored whole.

**Step 2 — look at what the model actually emitted** for `kv_countries_skyline`.
Re-run it first: every record now in `runs/` predates Step 1 and carries no
predictions.
Two leading hypotheses, in order:

1. **`max_new_tokens=5` truncates before an answer appears.** Countries sets 5,
   matching upstream's `CountriesEvaluator.max_tokens`. If the model emits a
   preamble, a newline, or a chat-template artefact first, the answer never lands
   inside the budget. Test by re-running skyline at `--max_new_tokens 32`; if it
   jumps, that is the cause. Note this would be a deviation from upstream's 5 and
   must be recorded as one.
2. **Scoring mismatch.** `gold` is a country name like `United States`. Walk
   `lkv/scoring.py`'s F1 path by hand on one real (prediction, gold) pair.

**Step 3 — only once skyline is plausible (~0.55–0.69) does K1 mean anything.**
Then re-run the full gate. Countries is cheap: 200 items at 5 tokens took **6
seconds** per condition.

---

## Results so far — all committed in `runs/`

**L1, GSM8K Qwen3-8B, n=200, sampled** (targets: relay 0.938, single 0.811):

| latent_steps | 0 | 10 | 20 | 40 | 80 | single | single no-think |
|---|---|---|---|---|---|---|---|
| accuracy | **0.950** | 0.910 | 0.945 | 0.925 | 0.940 | **0.960** | 0.890 |

Also `lm_gsm8k_4b` 0.9250 and `lm_gsm8k_14b` 0.9300.

**L2, MBPP+ n=378** (targets: relay 0.746, single 0.648):

| | relay | single |
|---|---|---|
| accuracy | **0.7487** | **0.7566** |

**K1, Countries** — all six conditions `0.0000`. **Void, see Task 0.**
**K2, HotpotQA** — baseline `0.0020` only; the rest never ran.

### What this means, stated carefully

On two independent tasks **the paper's LatentMAS numbers reproduce closely**
(0.950 vs 0.938; 0.7487 vs 0.746) while **its baselines do not** — ours are 15
and 11 points stronger — and the relay lands marginally *behind* a single agent
both times.

The internal comparison is matched: both arms share decoding, prompts, seed,
items and thinking setting. So within this harness, LatentMAS shows no advantage
over one agent on either task.

Two things **not** established, which must not be asserted:

- **That the method fails.** GSM8K is saturated for Qwen3-8B (0.89–0.96), so it
  cannot test the claim at this scale. MBPP+ is not saturated and the relay still
  reached 0.7487, so the relay is functioning.
- **That thinking mode explains the baseline gap.** It explains part: GSM8K
  single-agent falls 0.9600 → 0.8900 with `--no_thinking`, still well above
  0.811. Roughly half the gap is unaccounted for.

The missing experiment is the matched no-thinking pair. Three runs were in flight
when the box was destroyed and produced no files: `lm_gsm8k_8b_s0_nothink`,
`lm_mbppplus_8b_nothink`, `lm_mbppplus_8b_single_nothink`. **Re-run these** —
they complete the 2×2 of {thinking, no-thinking} × {relay, single} and are the
cleanest test of whether the published advantage survives a fair baseline.

One more angle worth checking, not yet investigated: upstream ships a `--think`
flag documented as "manually add think token in the prompt **for LatentMAS**",
applied only in `methods/latent_mas.py`. If the paper used it for the relay but
not the baseline, that asymmetry alone could produce the reported gap. Their
README commands do not use it. This is a hypothesis, not a finding.

---

## Task order tomorrow

| | what | gate |
|---|---|---|
| **0** | ~~Store predictions~~ (done, on CPU); diagnose and fix the Countries zeros. | skyline plausible (~0.55–0.69) |
| **1** | Re-run K1 Countries (6 conditions, ~1 min total). | **K1**: baseline ≤ 0.05, skyline ∈ [0.55, 0.69], r30 ≥ 0.42, r50 ≥ 0.55, monotone r30 ≤ r50 ≤ r70 (0.02 tol) |
| **2** | The three no-thinking runs above (~50 min, run concurrently). | completes the 2×2 |
| **3** | K2 HotpotQA (500) and Tipsheets (500). Tipsheets needs its loader registered in `lkv/data/__init__.py` — that is Task 16 proper. | **K2**: baseline ∈ [0.13, 0.25], skyline ≥ 0.66, r30 ≥ 0.26, r50 ≥ 0.53, r70 ≥ 0.62, strictly increasing |
| **4** | Task 20 integration sweep: 2 scopes × 3 ratios + random and pair-calibration controls, **each run twice, `--causal_ragged` off and on**. | — |
| **5** | Task 21 Step 5: re-run K1 conditions with `--causal_ragged`; record `ratio \| stock \| corrected \| delta` against the published 0.51/0.62/0.62. | — |

`scripts/gate_latentmas.sh`, `scripts/gate_kvcomm.sh` and `scripts/chain_gates.sh`
all exist and skip any condition whose JSON is present, so they are safe to
re-run after an interruption.

---

## Operational lessons — do not relearn these

**Pull results off the box continuously, not at the end.** `runs/*.json` is now
tracked in git (`.gitignore` carries an explicit exception). Rsync and commit
after every gate, not when the user announces they are about to destroy the box.

**Concurrency is the speedup lever, not batch size.** Runs peak at 16–38 GB of
192 GB at `bs=16`, so 4–5 fit at once. Do **not** raise `--generate_bs` on
sampled runs: batch size changes the order items consume the RNG stream, so
results stop being comparable with everything already measured. It is safe on
`--greedy` runs, which is what the KVComm gates use.

**Timings measured:** GSM8K 200 items ≈ 14–20 min; MBPP+ 378 items ≈ 100–141 min;
Countries 200 items ≈ 6 s; no-thinking runs ≈ 4× faster than thinking. Weights
download at ~16 GB in 13 s, so downloads are never the bottleneck.

**A `| tail` in a shell pipeline swallows the real exit status.** A setup failure
reported exit 0 that way yesterday.

---

## Rulings that bind — do not re-litigate

1. **Task 20's analyzer must not use the plan's `or` form.** Write
   `1.0 if d.get("layer_retention") is None else d["layer_retention"]`. The same
   trap applies to `.get(key, default)`: `baseline` and `skyline` records carry
   `layer_retention` **with an explicit None**, and `.get`'s default applies only
   when the key is absent — so the plan's `d.get('layer_retention', 1.0)` returns
   None and `f"{None:10.4f}"` raises. `scripts/gate_kvcomm.sh` already handles
   this; the plan text still contains the bug.
2. **`latent_steps` has no default and must be searched.** Upstream's default 0
   discards the relay cache. Measured: the sweep is flat within 1–2 SE, so there
   is no evidence any value helps.
3. **`--calib_source` defaults to `relay`, which is not KVComm-as-published.**
   Deliberate. `pair` gives the drop-in comparison; it is rejected at parse time
   for GSM8K/MBPP+, whose items carry no `context`.
4. **The verbatim upstream reference block in `tests/test_importance.py` stays.**
5. **Reproduction uses sampled decoding; internal A/B uses `--greedy`.** The
   LatentMAS gates sample; the KVComm gates are greedy (upstream's `eval.py` sets
   `do_sample=False`). Never mix within one comparison.
6. **`--causal_ragged` defaults OFF.** Off reproduces KVComm as published, whose
   table came from the non-causal path. On is correct and is what every
   integration number must use.
7. **The Llama mirror is a mirror, not asserted byte-identical.** If K1/K2 miss
   after the harness is fixed, "the mirror differs from KVComm's weights" belongs
   on the diagnostic ladder — below the harness causes, above concluding the
   method does not reproduce.

---

## Hazards

**The failure mode of this codebase is silent wrongness, not crashes.** Yesterday
alone: a loader that read three keys the file does not contain (D17), a relay
whose results depended on batch composition (D16), and six accuracy scores of
exactly 0.0000 that a green test suite had nothing to say about. When a result
looks clean, ask what would have to be true for it to be wrong.

**D8 remains open for the integration.** Every `top_layers < 1.0` run without
`--causal_ragged` measures non-causal attention at each boundary. Measured on the
tiny fixture: stock vs causal = 4.333 max delta on one relay hop. `ragged_path`
in each record says which path produced it.

**Not verified:** that any KVComm number reproduces. K1 has never produced a
valid measurement.
