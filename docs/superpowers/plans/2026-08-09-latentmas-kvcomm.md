# LatentMAS × KVComm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean-room implementation in which LatentMAS reproduces its published accuracy and KVComm reproduces its published accuracy, then apply KVComm's layer-selective KV trimming at LatentMAS's inter-agent cache handoff and measure the accuracy/efficiency tradeoff.

**Architecture:** One Python package `lkv/` with a single shared model wrapper. Two independent systems sit on top of it — `lkv/latentmas.py` (LatentMAS's 4-role relay over one continuously-growing KV cache) and `lkv/kvcomm/` (KVComm's sender→receiver single-hop cache transfer). Both are gated against their own paper's numbers *before* they are composed. Integration then inserts KVComm's `prepare_key_cache` at each LatentMAS agent boundary. Critically, we do **not** register a custom attention backend — we follow KVComm's approach of subclassing the attention module while leaving `config._attn_implementation == "sdpa"` untouched, which sidesteps an entire class of causal-mask bug.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace `transformers==4.53.3` (hard pin — see Global Constraints), `datasets`, `pytest`. Models: Qwen3-4B/8B/14B (LatentMAS), Llama-3.1-8B-Instruct (KVComm). CPU-only correctness tests via a randomly-initialized 4-layer model; GPU only for accuracy gates.

---

## Global Constraints

These apply to every task. Values are copied verbatim from the two upstream repos and the two papers.

- **`transformers==4.53.3` exactly.** KVComm's `requirements.txt:2` states: `transformers==4.53.3 # Must be 4.53.3, model definition changed in 4.54.0`. Qwen3 is supported at this version, so one environment serves both systems. Never upgrade to silence a deprecation warning.
- **Python `>=3.10,<3.12`** on the GPU box (LatentMAS README specifies `python=3.10`). The CPU test suite must additionally pass on Python 3.13 so it can run on the dev laptop.
- **dtype:** `torch.bfloat16` on CUDA, `torch.float32` on CPU. Both upstream repos do this (`stock-latentmas/models.py:57-60`, `stock-kvcomm/com.py:112-113`).
- **Attention implementation:** always `attn_implementation="sdpa"`, set explicitly at load. Never register anything into `ALL_ATTENTION_FUNCTIONS` or `ALL_MASK_ATTENTION_FUNCTIONS`. Rationale: in transformers ≥4.52, `masking_utils` treats an unregistered `_attn_implementation` as a custom backend that builds its own mask, and returns `attention_mask=None` — which silently makes attention fully bidirectional. KVComm avoids this by subclassing `LlamaAttention`/`Qwen2Attention` and hot-swapping `block.self_attn` while leaving the config string alone (`stock-kvcomm/model_attn.py:17`, `stock-kvcomm/models.py:71-93`). We do the same.
- **`padding_side="left"` on every tokenizer, always.** Non-negotiable. See Task 3.
- **KVComm hyperparameters** (paper §B.2, cross-checked against code): `mu = 0.5` as a fraction of depth so `mu_abs = mu * (n_layers - 1)`; `sigma = 10.0`; `calib_size = 1`; `alpha = 1.0` for the Llama family and `alpha = 0.8` for Qwen and Falcon families. Retention ratios swept: `0.3`, `0.5`, `0.7`.
- **LatentMAS hyperparameters** (repo argparse): `max_new_tokens` 2048 for GSM8K and 4096 for MBPP+, `temperature=0.6`, `top_p=0.95`, `do_sample=True`, 4 agents (Planner, Critic, Refiner, Judger), `prompt="sequential"`, `seed=42`. `latent_steps` is a free hyperparameter the paper tunes over `[0, 80]`.
- **Every accuracy run must write a `per_item` array** of `{"i": int, "correct": bool, "score": float}` so any two runs on the same slice can be compared with a paired test (McNemar) rather than an unpaired one.
- **Determinism:** a `--greedy` flag must exist that sets `do_sample=False` and drops `temperature`/`top_p`. Paper reproduction uses sampling at `temperature=0.6`; all *internal* A/B comparisons use `--greedy`. Never mix the two within one comparison.
- **No results from prior experiments are inputs.** Earlier work in `../latentmas-kvcomm/` is reference-only for known hazards. Every number in this plan's gates must be produced by this codebase.

### Published targets

LatentMAS, Table 1 (sequential), accuracy %:

| Task | 4B Single / Text / Latent | 8B Single / Text / Latent | 14B Single / Text / Latent |
|---|---|---|---|
| GSM8K | 82.4 / 89.8 / 88.2 | 81.1 / 92.3 / **93.8** | 83.7 / 93.8 / **95.2** |
| MBPP+ | 63.5 / 69.8 / **73.5** | 64.8 / 69.5 / **74.6** | 68.5 / 72.8 / **75.7** |

KVComm, Table 6, Llama-3.1-8B-Instruct as both sender and receiver:

| Method | Countries | Tipsheets | HotpotQA |
|---|---|---|---|
| Baseline (query only) | 0.00 | 0.05 | 0.19 |
| Skyline (full text) | 0.62 | 0.92 | 0.74 |
| KVComm (0.3) | 0.51 | 0.93 | 0.33 |
| KVComm (0.5) | 0.62 | 0.95 | 0.60 |
| KVComm (0.7) | 0.62 | 0.96 | 0.69 |

Sample counts: Countries 200, Tipsheets 500, HotpotQA 500. `max_new_tokens`: Countries 5, Tipsheets 10, HotpotQA 48. Note there is **no full-cache (1.0) row anywhere in the paper** — the "3× communication reduction" claim is the layer ratio `1/0.3`, not a measured comparison.

### Known upstream defects this plan must fix

| # | Defect | Evidence | Fixed in |
|---|---|---|---|
| D1 | `padding_side` is never set, yet `generate` is called on ragged batches and the output slice assumes left padding. Right padding is the tokenizer default for most checkpoints, so generation continues from PAD and the slice is wrong. **CONFIRMED by measurement**: the same 3-token prompt greedily continued `[110, 125, 188, 119, 110, 162]` alone and left-padded in a batch, but `[239, 188, 239, 188, 125, 53]` right-padded — a completely different completion. transformers itself warns: "A decoder-only architecture is being used, but right-padding was detected!" | `stock-latentmas/models.py:230,260-267`; `_ensure_pad_token` at `:15-20` sets only `pad_token` | Task 3 |
| D2 | `--latent_steps` defaults to `0` and no README command passes it. At `0`, `past_for_decoding = past_kv if self.latent_steps > 0 else None` discards the entire relay cache, making LatentMAS a plain single-agent call. | `stock-latentmas/run.py:102`, `methods/latent_mas.py:158` | Task 8 |
| D3 | `Qwen/Qwen3-8B` is absent from the `--model_name` choices (`Qwen/Qwen3-4B` appears twice), so the README's own 8B command fails argparse. | `stock-latentmas/run.py:91` | Task 4 |
| D4 | Realignment: when `--latent_space_realign` is off the matrix is replaced by the identity, but the L2 rescale to `target_norm` is applied **unconditionally** on every latent step. "Realign off" is therefore not "no transform". | `stock-latentmas/models.py:179-183` vs `:204-213`, called at `:324`,`:394` | Task 5 |
| D5 | `_truncate_past` is gated on `sequential_info_only`/`latent_only`, read via `getattr(args, ..., False)` but never added to argparse — permanently dead. | `stock-latentmas/methods/latent_mas.py:40-41,61-79` | Task 8 |
| D6 | `--text_mas_context_length` defaults to `-1` and is used as `context[:args.text_mas_context_length]`, silently dropping the last character instead of disabling truncation. | `stock-latentmas/run.py:106`, `prompts.py:349` | Task 6 |
| D7 | `from vllm import SamplingParams` is unguarded at module import, so the pure-HF path hard-requires vLLM. | `stock-latentmas/methods/latent_mas.py:9` | Task 8 |
| D8 | KVComm's default (`shift_back=False`) hands a ragged `DynamicCache` to the stock forward. `get_seq_length()` reads layer 0, which KVComm always keeps full, so the causal mask is sized for the full length; sdpa then slices it to the shorter key length, making every column visible to every query. **CONFIRMED by measurement** on transformers 4.53.3 / torch 2.8.0: with cache lengths `[12, 1, 12, 1]`, changing only the *last* of four receiver tokens moved the *first* position's logits by `1.940e-01`. The untrimmed control drifts `0.000e+00`. Attention in trimmed layers is not causal. | `stock-kvcomm/models.py:95-108` (layer 0 forced full via `or i == 0`), `:146-150` (ragged cache passed through); `shift_back` at `:253-359` exists to correct positions | Task 12 |
| D9 | KVComm's `CVCommunicator.forward` declares an `attention_mask` parameter and never forwards it to `self.B`. | `stock-kvcomm/models.py:113` vs `:146-150` | Task 12 |
| D10 | KVComm's implemented metric is `f1_match(...) -> bool`, i.e. the **fraction of examples whose word-set F1 exceeds 0.5**, while paper §4.1 states plain "F1 score". The tables cannot be reproduced without resolving which was used. | `stock-kvcomm/utils/f1.py:65`, `base_evaluator.py:36-48` vs paper §4.1 | Task 7, resolved in Task 13 |
| D11 | LatentMAS re-supplies the full question as text to all four agents; `context` is accepted by both latent prompt builders and never interpolated. | `stock-latentmas/prompts.py:12,20,36,51`; `methods/latent_mas.py:95,100` | Recorded, deliberately **not** changed (Task 6) — changing it would no longer be LatentMAS |

---

## File Structure

```
latent-kvcomm/
├── requirements.txt            exact pins; transformers==4.53.3
├── pyproject.toml              package metadata, pytest config
├── lkv/
│   ├── __init__.py
│   ├── config.py               RunConfig dataclass + argparse; single source of truth for flags
│   ├── modeling.py             load_model(): dtype, sdpa, left padding, pad token
│   ├── decode.py               generate_text_batch / generate_latent_batch — correct slicing
│   ├── realign.py              LatentMAS Wa matrix; realign and renorm as SEPARATE flags
│   ├── prompts.py              LatentMAS 4 role prompts; KVComm sender/receiver/skyline templates
│   ├── latentmas.py            the 4-role relay loop; accepts an optional trim hook
│   ├── scoring.py              GSM8K boxed extraction, MBPP+ exec, F1 (mean and thresholded)
│   ├── stats.py                McNemar, exact binomial, bootstrap CI
│   ├── instrument.py           cache bytes, peak memory, attention FLOPs, wall clock
│   ├── run.py                  entry point; writes one JSON per run incl. per_item
│   ├── kvcomm/
│   │   ├── __init__.py
│   │   ├── tracer.py           attention subclass that stashes prefill Q/K; sdpa preserved
│   │   ├── importance.py       calc_layer_importance, get_layer_ranking
│   │   ├── trim.py             prepare_key_cache + ragged-cache mask correction
│   │   └── pair.py             single-hop sender→receiver harness (KVComm reproduction)
│   └── data/
│       ├── __init__.py         load_task(name) dispatch
│       ├── gsm8k.py            HF "gsm8k"/"main"/test
│       ├── mbppplus.py         HF "evalplus/mbppplus"/test, 378 items
│       ├── hotpotqa.py         HF "hotpotqa/hotpot_qa"/"distractor"/validation, 500 subsample
│       └── countries.py        200-item context/question set
├── tests/
│   ├── conftest.py             tiny random Qwen3 + Llama fixtures, CPU, no download
│   ├── test_padding.py         D1: batch-invariance of greedy generation
│   ├── test_decode.py          latent-step count, cache growth, slice correctness
│   ├── test_realign.py         D4: ridge solve correctness; renorm separability
│   ├── test_importance.py      bit-exactness vs verbatim upstream reference
│   ├── test_trim.py            retention counts, layer-0 rule, layer_map
│   ├── test_ragged_cache.py    D8/D9: causality under a ragged cache
│   ├── test_scoring.py         extraction, exec sandbox, both F1 variants
│   ├── test_latentmas_loop.py  4 roles, 3×latent_steps, cache monotonicity
│   ├── test_integration.py     trim hook fires at each boundary; scope modes
│   └── test_stats.py           McNemar against hand-computed values
├── scripts/
│   ├── setup_env.sh            environment bootstrap
│   ├── gate_latentmas.sh       Phase 2 accuracy gates
│   ├── gate_kvcomm.sh          Phase 4 accuracy gates
│   └── sweep_integration.sh    Phase 5 sweep
└── docs/
    ├── superpowers/plans/2026-08-09-latentmas-kvcomm.md   this file
    └── FINDINGS.md             running log: every gate result, every deviation
```

Rationale for the split: `modeling`/`decode`/`realign` are the three places the upstream LatentMAS defects live, and each is independently testable on CPU with a tiny model. `kvcomm/` is a subpackage because its four modules are only meaningful together. `instrument.py` is separate so efficiency measurement can change without touching correctness code.

---

## Phase 0 — Environment and test harness

### Task 1: Repository scaffold and pinned environment

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `lkv/__init__.py`, `lkv/kvcomm/__init__.py`, `lkv/data/__init__.py`, `scripts/setup_env.sh`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `lkv` package; a `.venv` with `transformers==4.53.3`; `lkv.__version__` as a `str`.

- [ ] **Step 1: Write the failing test**

`tests/test_env.py`:
```python
import importlib


def test_transformers_is_pinned_to_4_53_3():
    import transformers
    assert transformers.__version__ == "4.53.3", (
        "KVComm requires exactly 4.53.3 (model definition changed in 4.54.0); "
        f"got {transformers.__version__}"
    )


def test_lkv_imports():
    lkv = importlib.import_module("lkv")
    assert isinstance(lkv.__version__, str)


def test_torch_available_cpu():
    import torch
    assert float(torch.ones(2, 2).sum()) == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py -v`
Expected: FAIL — the venv does not exist, so the command errors with "no such file or directory".

- [ ] **Step 3: Create the environment and package files**

`requirements.txt`:
```
torch>=2.4,<2.9
transformers==4.53.3
tokenizers>=0.21,<0.22
datasets>=2.19,<4
accelerate>=0.30
numpy>=1.26,<3
tqdm>=4.66
scipy>=1.11
pytest>=8.0
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lkv"
version = "0.1.0"
description = "KVComm layer-selective KV trimming applied to LatentMAS inter-agent handoff"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["lkv*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = [
    "gpu: requires a CUDA device and downloaded weights",
    "slow: takes more than 30 seconds",
]
```

`lkv/__init__.py`:
```python
__version__ = "0.1.0"
```

`lkv/kvcomm/__init__.py` and `lkv/data/__init__.py`: empty files.

`scripts/setup_env.sh`:
```bash
#!/usr/bin/env bash
# Bootstraps the environment. Works on the dev laptop (CPU) and the GPU box.
# The transformers pin is load-bearing -- see docs Global Constraints.
set -euo pipefail

PY=${PY:-python3.11}
command -v "$PY" >/dev/null || { echo "need $PY on PATH; set PY=..."; exit 1; }

"$PY" -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/pip install -q -e .

./.venv/bin/python - <<'EOF'
import torch, transformers
assert transformers.__version__ == "4.53.3", transformers.__version__
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo "env OK"
```

- [ ] **Step 4: Run the bootstrap, then the test**

Run:
```bash
chmod +x scripts/setup_env.sh && PY=python3.11 ./scripts/setup_env.sh && ./.venv/bin/python -m pytest tests/test_env.py -v
```
Expected: PASS, 3 tests.

Fallbacks, in order, if `python3.11` is absent or the pin will not resolve: (a) `PY=python3.13 ./scripts/setup_env.sh`; (b) `conda create -y -p ./.venv python=3.11 && ./.venv/bin/pip install -r requirements.txt -e .`. Record which path worked in `docs/FINDINGS.md` — the GPU box must use the same one.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pyproject.toml lkv/__init__.py lkv/kvcomm/__init__.py lkv/data/__init__.py scripts/setup_env.sh tests/test_env.py
git commit -m "chore: pinned environment and package scaffold"
```

---

### Task 2: CPU test fixtures — tiny randomly-initialized models

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_fixtures.py`

**Interfaces:**
- Consumes: `lkv` from Task 1.
- Produces: session fixtures `tiny_qwen3` and `tiny_llama` (a `PreTrainedModel` in eval mode on CPU, `config._attn_implementation == "sdpa"`, `num_hidden_layers == 4`, `tie_word_embeddings is False`); `tiny_ids` returning `(input_ids, attention_mask)` as a left-padded ragged batch of 2; `short_long_solo` returning the same two sequences unbatched. Every later CPU test depends on these.

Random init is deliberate: every assertion in this suite is about masks, cache lengths, slice indices, and matrix algebra, none of which depend on learned weights. `tie_word_embeddings=False` is required so the realignment matrix in Task 5 is a genuine solve rather than trivially the identity.

- [ ] **Step 1: Write the failing test**

`tests/test_fixtures.py`:
```python
import torch


def test_tiny_qwen3_shape(tiny_qwen3):
    assert tiny_qwen3.config.num_hidden_layers == 4
    assert tiny_qwen3.config.num_key_value_heads == 2
    assert tiny_qwen3.config.num_attention_heads == 8
    assert tiny_qwen3.config._attn_implementation == "sdpa"
    assert tiny_qwen3.config.tie_word_embeddings is False


def test_tiny_qwen3_forward_produces_cache(tiny_qwen3):
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        out = tiny_qwen3(ids, use_cache=True)
    assert out.logits.shape == (1, 4, tiny_qwen3.config.vocab_size)
    assert len(out.past_key_values.key_cache) == 4
    assert out.past_key_values.key_cache[0].shape[-2] == 4


def test_tiny_llama_shape(tiny_llama):
    assert tiny_llama.config.num_hidden_layers == 4
    assert tiny_llama.config._attn_implementation == "sdpa"


def test_tiny_ids_is_left_padded(tiny_ids):
    ids, mask = tiny_ids
    assert ids.shape[0] == 2
    assert mask[0, 0].item() == 0     # short row: padding on the LEFT
    assert mask[0, -1].item() == 1
    assert mask[1].min().item() == 1  # long row: fully attended
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_fixtures.py -v`
Expected: FAIL — `fixture 'tiny_qwen3' not found`.

- [ ] **Step 3: Write the fixtures**

`tests/conftest.py`:
```python
"""CPU fixtures for mechanical correctness tests.

Random weights on purpose: every assertion in this suite is about masks,
cache lengths, slice indices, and matrix algebra, none of which depend on
what the model has learned. A 4-layer model needs no download and runs in
milliseconds.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

PAD_ID = 0


def _tiny_kwargs():
    return dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
        # Untied so the realignment matrix is a genuine solve, not I.
        tie_word_embeddings=False,
        pad_token_id=PAD_ID,
        attn_implementation="sdpa",
    )


@pytest.fixture(scope="session")
def tiny_qwen3():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(head_dim=8, **_tiny_kwargs()))
    model = model.to(torch.float32).eval()
    assert model.config._attn_implementation == "sdpa"
    return model


@pytest.fixture(scope="session")
def tiny_llama():
    torch.manual_seed(1)
    model = LlamaForCausalLM(LlamaConfig(**_tiny_kwargs())).to(torch.float32).eval()
    assert model.config._attn_implementation == "sdpa"
    return model


@pytest.fixture
def tiny_ids():
    """A ragged batch, LEFT padded: row 0 has 3 real tokens, row 1 has 7."""
    short, long = [5, 6, 7], [5, 6, 7, 8, 9, 10, 11]
    gap = len(long) - len(short)
    ids = torch.tensor([[PAD_ID] * gap + short, long], dtype=torch.long)
    mask = torch.tensor([[0] * gap + [1] * len(short), [1] * len(long)], dtype=torch.long)
    return ids, mask


@pytest.fixture
def short_long_solo():
    """The same two sequences unbatched, for batch-invariance comparisons."""
    return (
        torch.tensor([[5, 6, 7]], dtype=torch.long),
        torch.tensor([[5, 6, 7, 8, 9, 10, 11]], dtype=torch.long),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_fixtures.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_fixtures.py
git commit -m "test: CPU fixtures with tiny random Qwen3 and Llama"
```

---

## Phase 1 — LatentMAS correctness (all CPU-testable)

### Task 3: Model loading and batch-invariant decoding — fixes D1

This is the highest-value task in the plan. Upstream computes `prompt_lengths = attention_mask.sum(dim=1).tolist()` and then slices `sequences[idx, length:]` (`stock-latentmas/models.py:230,260-267`). That slice is only correct under **left** padding, but `padding_side` is never set anywhere in the repo, and the HF default for Qwen and Llama tokenizers is `"right"`. Under right padding, `generate` continues from a PAD token for every row shorter than the batch maximum, and the slice starts inside the prompt. At the repo's default `generate_bs=20` on a length-heterogeneous task like MBPP+, most rows in most batches are affected.

**Files:**
- Create: `lkv/modeling.py`
- Create: `lkv/decode.py`
- Test: `tests/test_padding.py`

**Interfaces:**
- Consumes: fixtures `tiny_qwen3`, `tiny_ids`, `short_long_solo` from Task 2.
- Produces:
  - `load_model(model_name: str, device: str, dtype: torch.dtype | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]` — tokenizer always has `padding_side == "left"` and a non-None `pad_token_id`; model always has `config._attn_implementation == "sdpa"` and `config.use_cache is True`.
  - `pad_batch(tokenizer, texts: list[str]) -> tuple[Tensor, Tensor]` returning `(input_ids, attention_mask)`, left-padded.
  - `generate_text_batch(model, tokenizer, input_ids: Tensor, attention_mask: Tensor, max_new_tokens: int, greedy: bool, temperature: float = 0.6, top_p: float = 0.95, past_key_values=None, past_mask: Tensor | None = None) -> tuple[list[str], Tensor]` returning decoded continuations and the full `sequences` tensor. Callers pass the prompt ALONE for `input_ids` and `attention_mask`; the function prepends the cached span itself. `past_mask` is REQUIRED whenever the cache is non-empty, and must be the mask that was in force when the cache was written — synthesizing ones re-enables attention to pad positions.
  - Task 5 consumes `load_model`; Tasks 5, 8, 14 consume `generate_text_batch`.

- [ ] **Step 1: Write the failing test**

`tests/test_padding.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_padding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.modeling'`.

- [ ] **Step 3: Write the implementation**

`lkv/modeling.py`:
```python
"""Model and tokenizer loading with the two settings upstream gets wrong.

1. padding_side="left". Upstream never sets it (stock-latentmas/models.py:15-20
   sets only pad_token), yet slices generate() output as if it were left-padded
   (models.py:230,260-267). See tests/test_padding.py.
2. attn_implementation="sdpa", set explicitly. Never register a custom backend:
   in transformers >=4.52 an unregistered _attn_implementation makes
   masking_utils return attention_mask=None, which is silently bidirectional.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if str(device).startswith("cuda") else torch.float32


def load_model(model_name: str, device: str = "cuda:0", dtype: torch.dtype | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError(f"{model_name}: no pad_token and no eos_token to borrow")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if dtype is not None else default_dtype(device),
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = True
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config._attn_implementation == "sdpa", model.config._attn_implementation
    return model, tokenizer


def pad_batch(tokenizer, texts: list[str]):
    """Tokenize a batch, forcing left padding regardless of tokenizer state."""
    tokenizer.padding_side = "left"
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False,
    )
    return enc["input_ids"], enc["attention_mask"]
```

`lkv/decode.py`:
```python
"""Batched text and latent generation.

Two slicing hazards, both of which upstream gets wrong:

1. The output slice is taken at the PADDED prompt width, not each row's unpadded
   length. With left padding those differ, and using the unpadded length -- as
   upstream does at stock-latentmas/models.py:230,260-267 -- lands inside the
   prompt for every row that was padded.

2. `generate()` will not accept a prompt-only `input_ids` alongside a cache. It
   computes `cache_position = arange(input_ids.shape[1])[past_len:]` and forwards
   only `input_ids[:, past_len:]`, so a prompt-only tensor underflows that slice
   and raises IndexError. This function therefore owns the widening: callers
   always pass their prompt alone, for BOTH input_ids and attention_mask, and it
   prepends the cached span itself.

3. The cached span's mask must be the REAL historical mask, never all-ones. If the
   cache was built from a left-padded ragged batch -- which every relay batch is --
   then synthesizing ones re-enables attention to pad positions that were correctly
   masked when the cache was written. Measured on transformers 4.53.3: greedy
   continuations diverge from the correct-mask reference in the large majority of
   randomized trials, with max logit differences around 0.53, comparable to a
   typical top-1/top-2 gap. The `input_ids` filler above IS inert; the mask is not.
   `past_mask` is therefore REQUIRED whenever the cache is non-empty. A caller
   holding a genuinely unpadded cache passes explicit ones -- one line, and it
   documents the claim instead of assuming it.

CONTRACT: input_ids and attention_mask must have the same width on entry -- the
prompt only, never pre-widened for the cache. Validated below, because a caller
that pre-widens would otherwise be double-counted into silently wrong tokens.
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
    past_mask: torch.Tensor | None = None,
):
    if attention_mask.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"pass the prompt alone: input_ids width {input_ids.shape[1]} != "
            f"attention_mask width {attention_mask.shape[1]}. This function "
            "prepends the cached span itself; do not pre-widen either tensor."
        )

    past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
    if past_len:
        batch = input_ids.shape[0]
        if past_mask is None:
            raise ValueError(
                f"past_mask is required when past_key_values holds {past_len} "
                "positions. Synthesizing all-ones would let queries attend to "
                "pad positions the cache masked when it was written. Pass the "
                "mask that was in force for the cached span; if it is genuinely "
                "unpadded, pass explicit ones."
            )
        if tuple(past_mask.shape) != (batch, past_len):
            raise ValueError(
                f"past_mask shape {tuple(past_mask.shape)} != expected "
                f"{(batch, past_len)} (batch, cached positions)."
            )
        # The prepended ids are never embedded -- generate forwards only
        # input_ids[:, past_len:] -- so pad_token_id is inert filler that exists
        # purely to satisfy the length arithmetic. Verified on transformers
        # 4.53.3: true ids, all-PAD, and random garbage in the cached span all
        # produce byte-identical output, matching the no-cache reference.
        input_ids = torch.cat(
            [torch.full((batch, past_len), tokenizer.pad_token_id,
                        dtype=input_ids.dtype, device=input_ids.device), input_ids],
            dim=1,
        )
        attention_mask = torch.cat(
            [past_mask.to(dtype=attention_mask.dtype, device=attention_mask.device),
             attention_mask],
            dim=1,
        )

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
    # Left padding means every row's prompt ends at the same column; after the
    # widening above, that column accounts for the cached span too.
    prompt_width = int(input_ids.shape[1])
    texts = [
        tokenizer.decode(sequences[i, prompt_width:], skip_special_tokens=True).strip()
        for i in range(sequences.shape[0])
    ]
    return texts, sequences
```

Three more tests belong in `tests/test_padding.py` for the cache path:

```python
def test_cached_generation_matches_uncached_reference(tiny_qwen3):
    """A cache plus the remaining prompt must equal running the whole thing."""
    import copy
    ctx = torch.tensor([[5, 6, 7, 8, 9, 10]])
    new = torch.tensor([[11, 12, 13]])
    tok = _IdTok()
    with torch.no_grad():
        cache = tiny_qwen3(ctx, use_cache=True).past_key_values

    full = torch.cat([ctx, new], dim=1)
    ref, _ = generate_text_batch(tiny_qwen3, tok, full, torch.ones_like(full),
                                 max_new_tokens=4, greedy=True)
    got, _ = generate_text_batch(tiny_qwen3, tok, new, torch.ones_like(new),
                                 max_new_tokens=4, greedy=True,
                                 past_key_values=copy.deepcopy(cache),
                                 past_mask=torch.ones_like(ctx))
    assert got == ref, (got, ref)


def test_cached_generation_honours_a_padded_history(tiny_qwen3):
    """THE decisive test. A cache built from a LEFT-PADDED row must keep those
    pad positions masked. Synthesizing all-ones instead diverges from the
    correct-mask reference -- measured max logit delta ~0.53."""
    prefix_ids = torch.tensor([[0, 0, 11, 12, 13, 14, 15]], dtype=torch.long)
    prefix_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    prompt = torch.tensor([[20, 21, 22]], dtype=torch.long)

    full_ids = torch.cat([prefix_ids, prompt], dim=1)
    full_mask = torch.cat([prefix_mask, torch.ones_like(prompt)], dim=1)
    with torch.no_grad():
        reference = _greedy(tiny_qwen3, full_ids, full_mask, n=6)
    reference_continuation = reference[0, full_ids.shape[1]:]

    # Prime the cache with the same left-pad position convention generate() uses
    # internally. A plain forward() defaults position_ids to a bare arange() and
    # ignores the mask, which would desync the comparison for a reason unrelated
    # to the fix under test. This step is easy to omit and the resulting failure
    # looks exactly like a real regression.
    position_ids = prefix_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(prefix_mask == 0, 1)
    with torch.no_grad():
        cache = tiny_qwen3(prefix_ids, attention_mask=prefix_mask,
                           position_ids=position_ids, use_cache=True).past_key_values

    _, seqs = generate_text_batch(
        tiny_qwen3, _IdTok(), prompt, torch.ones_like(prompt),
        max_new_tokens=6, greedy=True,
        past_key_values=copy.deepcopy(cache), past_mask=prefix_mask,
    )
    assert torch.equal(seqs[0, -6:], reference_continuation), (
        "cached generation diverged from the uncached reference -- "
        "past_mask was not honoured correctly"
    )


def test_cache_without_past_mask_is_rejected(tiny_qwen3):
    """Silently guessing all-ones is the defect; refusing is the fix."""
    import copy
    import pytest
    ctx = torch.tensor([[5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        cache = tiny_qwen3(ctx, use_cache=True).past_key_values
    new = torch.tensor([[11, 12, 13]])
    with pytest.raises(ValueError, match="past_mask is required"):
        generate_text_batch(tiny_qwen3, _IdTok(), new, torch.ones_like(new),
                            max_new_tokens=2, greedy=True,
                            past_key_values=copy.deepcopy(cache))


def test_prewidened_mask_is_rejected(tiny_qwen3):
    """The contract is prompt-only; a pre-widened mask must raise, not silently
    produce wrong tokens."""
    import pytest
    new = torch.tensor([[11, 12, 13]])
    with pytest.raises(ValueError, match="prompt alone"):
        generate_text_batch(tiny_qwen3, _IdTok(), new,
                            torch.ones(1, 9, dtype=torch.long),
                            max_new_tokens=2, greedy=True)


def test_no_cache_path_is_unaffected(tiny_qwen3):
    ids = torch.tensor([[5, 6, 7]])
    texts, seqs = generate_text_batch(tiny_qwen3, _IdTok(), ids,
                                      torch.ones_like(ids), max_new_tokens=4,
                                      greedy=True)
    assert seqs.shape[1] == 3 + 4
    assert texts[0].split() == [str(int(x)) for x in seqs[0, 3:]]
```

One more test is required, and it is the one D1 and D12 both argue for — every
cached-path test above is single-row, yet both defects were specifically
ragged-multi-row bugs:

```python
def test_cached_generation_matches_per_row_references_in_a_ragged_batch(tiny_qwen3):
    """Batch invariance across the cache boundary. Rows whose real prefixes and
    real prompts differ in length, left-padded to common widths, must each
    reproduce the answer they get alone. Tasks 8/14/18 run batches of 8-16, so a
    batch-dimension error here would corrupt most rows of most batches -- exactly
    the D1 failure mode, one layer deeper."""
```

Build it so each row's oracle is that row run **alone, unpadded, with no cache** —
a per-row reference cannot be contaminated by the batching or the widening under
test. Prime the cache with the same `position_ids` convention as above. Prove the
test is not vacuous by injecting a row-0-mask-broadcast error once and confirming
it fails while the single-row cached tests still pass.

`_IdTok` is the identity-decoding stub already used by
`test_generate_text_batch_returns_only_the_continuation`; lift it to module scope
so every test shares one definition. `copy` and `pytest` are imported at module
scope rather than inside each test.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_padding.py -v`
Expected: PASS, 3 tests. If `test_batched_greedy_matches_unbatched` still fails, the cause is inside `generate`, not the slice — do not weaken the assertion; investigate `position_ids` handling and record findings in `docs/FINDINGS.md`.

- [ ] **Step 5: Commit**

```bash
git add lkv/modeling.py lkv/decode.py tests/test_padding.py
git commit -m "fix: left-pad batches and slice at padded prompt width (D1)"
```

---

### Task 4: Configuration — one source of truth for every flag, fixes D3

**Files:**
- Create: `lkv/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunConfig` dataclass and `parse_args(argv: list[str] | None = None) -> RunConfig`. Field names that later tasks reference by name: `system`, `model_name`, `model_b`, `task`, `max_samples`, `seed`, `greedy`, `temperature`, `top_p`, `max_new_tokens`, `generate_bs`, `device`, `n_agents`, `latent_steps`, `realign`, `renorm_latent`, `prompt`, `top_layers`, `alpha`, `mu`, `sigma`, `calib_size`, `trim_scope`, `trim_every_hop`, `random_selection`, `out_json`. `MODEL_CHOICES` must include `Qwen/Qwen3-8B` (D3).

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import pytest

from lkv.config import MODEL_CHOICES, RunConfig, parse_args


def test_qwen3_8b_is_selectable():
    """D3: upstream run.py:91 lists Qwen/Qwen3-4B twice and omits 8B."""
    assert "Qwen/Qwen3-8B" in MODEL_CHOICES
    assert len(MODEL_CHOICES) == len(set(MODEL_CHOICES)), "duplicate model choice"


def test_latent_steps_has_no_silent_zero_default():
    """D2: a 0 default silently disables the whole relay, so require it."""
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40"])
    assert cfg.latent_steps == 40
    with pytest.raises(SystemExit):
        parse_args(["--system", "latentmas", "--task", "gsm8k",
                    "--model_name", "Qwen/Qwen3-8B"])


def test_kvcomm_defaults_match_paper():
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
                      "--top_layers", "0.3"])
    assert cfg.mu == 0.5 and cfg.sigma == 10.0 and cfg.calib_size == 1
    assert cfg.alpha == 1.0            # Llama family
    assert cfg.model_b == cfg.model_name  # receiver defaults to sender


def test_alpha_autoselects_08_for_qwen_and_falcon():
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "Qwen/Qwen2.5-7B-Instruct",
                      "--top_layers", "0.5"])
    assert cfg.alpha == 0.8


def test_greedy_and_sampling_are_mutually_exclusive_in_effect():
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B",
                      "--latent_steps", "40", "--greedy"])
    assert cfg.greedy is True


def test_defaults_are_paper_defaults():
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40"])
    assert cfg.n_agents == 4
    assert cfg.temperature == 0.6 and cfg.top_p == 0.95
    assert cfg.prompt == "sequential"
    assert cfg.seed == 42
    assert cfg.realign is False        # store_true upstream
    assert cfg.renorm_latent is True   # D4: upstream applies it unconditionally
    assert isinstance(cfg, RunConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.config'`.

- [ ] **Step 3: Write the implementation**

`lkv/config.py`:
```python
"""Every runtime knob, in one place.

Two deliberate departures from upstream argparse:
  D3  Qwen/Qwen3-8B is included (upstream run.py:91 lists 4B twice, omits 8B).
  D2  --latent_steps is REQUIRED for the latentmas system. Upstream defaults it
      to 0, and at 0 methods/latent_mas.py:158 throws the relay cache away, so
      the default silently degrades LatentMAS into a single-agent call.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

MODEL_CHOICES = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
]

TASK_CHOICES = ["gsm8k", "mbppplus", "hotpotqa", "countries", "tipsheets"]

# Paper B.2: alpha = 1 for the Llama family, 0.8 for Qwen and Falcon.
_ALPHA_BY_FAMILY = {"qwen": 0.8, "falcon": 0.8, "llama": 1.0}


def default_alpha(model_name: str) -> float:
    lowered = model_name.lower()
    for family, value in _ALPHA_BY_FAMILY.items():
        if family in lowered:
            return value
    return 1.0


@dataclass
class RunConfig:
    system: str
    task: str
    model_name: str
    model_b: str = ""
    max_samples: int = -1
    seed: int = 42
    device: str = "cuda:0"
    greedy: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 0          # 0 -> per-task default, see task_defaults()
    generate_bs: int = 16
    # LatentMAS
    n_agents: int = 4
    latent_steps: int = 0
    realign: bool = False
    renorm_latent: bool = True
    prompt: str = "sequential"
    # KVComm
    top_layers: float = 1.0
    alpha: float = -1.0              # -1 -> resolve from model family
    mu: float = 0.5
    sigma: float = 10.0
    calib_size: int = 1
    random_selection: bool = False
    # Integration
    trim_scope: str = "all"          # "all" | "delta"
    trim_every_hop: bool = False
    out_json: str = ""
    extra: dict = field(default_factory=dict)


# max_new_tokens per task: LatentMAS README uses 2048 for GSM8K and the repo
# default 4096 for code; KVComm dataloader/__init__.py fixes short budgets.
_MAX_NEW_TOKENS = {
    "gsm8k": 2048, "mbppplus": 4096,
    "hotpotqa": 48, "countries": 5, "tipsheets": 10,
}


def task_defaults(task: str) -> int:
    return _MAX_NEW_TOKENS[task]


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(prog="lkv")
    p.add_argument("--system", required=True, choices=["latentmas", "kvcomm", "single", "textmas"])
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--model_name", required=True, choices=MODEL_CHOICES)
    p.add_argument("--model_b", default="", help="KVComm receiver; defaults to --model_name")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--greedy", action="store_true",
                   help="do_sample=False; use for every internal A/B comparison")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=0)
    p.add_argument("--generate_bs", type=int, default=16)
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--latent_steps", type=int, default=None)
    p.add_argument("--realign", action="store_true")
    p.add_argument("--no_renorm_latent", dest="renorm_latent", action="store_false")
    p.add_argument("--prompt", default="sequential", choices=["sequential", "hierarchical"])
    p.add_argument("--top_layers", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=-1.0)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--calib_size", type=int, default=1)
    p.add_argument("--random_selection", action="store_true")
    p.add_argument("--trim_scope", default="all", choices=["all", "delta"])
    p.add_argument("--trim_every_hop", action="store_true")
    p.add_argument("--out_json", default="")
    ns = p.parse_args(argv)

    if ns.system in ("latentmas", "textmas") and ns.latent_steps is None:
        p.error("--latent_steps is required for --system latentmas/textmas "
                "(a 0 default silently disables the relay; see D2)")
    if ns.latent_steps is None:
        ns.latent_steps = 0

    kw = vars(ns)
    if not kw["model_b"]:
        kw["model_b"] = kw["model_name"]
    if kw["alpha"] < 0:
        kw["alpha"] = default_alpha(kw["model_name"])
    if kw["max_new_tokens"] == 0:
        kw["max_new_tokens"] = task_defaults(kw["task"])
    return RunConfig(**kw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add lkv/config.py tests/test_config.py
git commit -m "feat: RunConfig with Qwen3-8B choice and required latent_steps (D2, D3)"
```

---

### Task 5: Latent-space realignment — fixes D4

Upstream conflates two transforms behind one flag. `_build_latent_realign_matrix` solves a ridge-regularized least-squares problem for `W` such that `E_out W ≈ E_in`, then **discards it for the identity when the flag is off** (`stock-latentmas/models.py:179-183`). But `_apply_latent_realignment` also rescales the vector to `target_norm`, and it is called unconditionally on every latent step (`:204-213`, called at `:324`,`:394`). So "realign off" still renormalizes. Any ablation of the flag is therefore confounded. We split them into `realign` and `renorm_latent`.

**Files:**
- Create: `lkv/realign.py`
- Test: `tests/test_realign.py`

**Interfaces:**
- Consumes: `tiny_qwen3` from Task 2.
- Produces:
  - `build_realign_matrix(model, device, dtype=torch.float32) -> tuple[Tensor, Tensor]` returning `(W, target_norm)` where `W` has shape `[D, D]` and `target_norm` is a 0-dim tensor.
  - `LatentProjector(model, realign: bool, renorm: bool, device)` with method `__call__(hidden: Tensor) -> Tensor`, shape-preserving on `[B, D]`.
  - Task 8 constructs one `LatentProjector` per run and calls it on each latent step.

- [ ] **Step 1: Write the failing test**

`tests/test_realign.py`:
```python
"""D4: realignment and renormalization are separate transforms.

Upstream ties both to one flag and applies the renorm unconditionally
(stock-latentmas/models.py:179-183 vs :204-213), so ablating the flag
ablates only the matmul. These tests pin the two apart.
"""
import torch

from lkv.realign import LatentProjector, build_realign_matrix


def test_matrix_solves_the_least_squares_system(tiny_qwen3):
    W, target_norm = build_realign_matrix(tiny_qwen3, "cpu")
    e_in = tiny_qwen3.get_input_embeddings().weight.detach().float()
    e_out = tiny_qwen3.get_output_embeddings().weight.detach().float()
    dim = e_in.shape[1]
    assert W.shape == (dim, dim)

    resid_w = (e_out @ W - e_in).norm()
    assert resid_w < (e_out - e_in).norm()      # beats the identity

    # Beating the identity is necessary but NOT sufficient, and this is the trap:
    # solving the TRANSPOSED problem also beats it (measured 2.97 vs 3.65 on this
    # fixture), so an orientation slip would pass an identity-only check. Compare
    # against the transposed solve directly.
    gram_t = e_in.T @ e_in + 1e-5 * torch.eye(dim)
    w_flipped = torch.linalg.solve(gram_t, e_in.T @ e_out)
    resid_flipped = (e_out @ w_flipped - e_in).norm()
    assert resid_w < resid_flipped, (resid_w.item(), resid_flipped.item())

    # target_norm is the mean row norm of the INPUT embedding
    assert torch.allclose(target_norm, e_in.norm(dim=1).mean(), atol=1e-4)


def test_realign_off_renorm_off_is_exactly_identity(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=False, device="cpu")
    h = torch.randn(3, tiny_qwen3.config.hidden_size)
    assert torch.equal(proj(h), h)
    # And the [D, D] solve must never have run for this configuration.
    assert proj.builds == 0


def test_renorm_alone_rescales_but_preserves_direction(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    h = torch.randn(3, tiny_qwen3.config.hidden_size)
    out = proj(h)
    _, target_norm = build_realign_matrix(tiny_qwen3, "cpu")
    assert torch.allclose(out.norm(dim=-1), target_norm.expand(3), atol=1e-3)
    cos = torch.nn.functional.cosine_similarity(out.float(), h.float(), dim=-1)
    assert torch.allclose(cos, torch.ones(3), atol=1e-5)


def test_realign_alone_changes_direction(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=True, renorm=False, device="cpu")
    h = torch.randn(3, tiny_qwen3.config.hidden_size)
    out = proj(h)
    cos = torch.nn.functional.cosine_similarity(out.float(), h.float(), dim=-1)
    assert (cos < 0.999).all(), "matmul had no effect -- matrix is the identity"


def test_shape_and_dtype_preserved(tiny_qwen3):
    for realign in (False, True):
        for renorm in (False, True):
            proj = LatentProjector(tiny_qwen3, realign=realign, renorm=renorm, device="cpu")
            h = torch.randn(2, tiny_qwen3.config.hidden_size, dtype=torch.float32)
            out = proj(h)
            assert out.shape == h.shape and out.dtype == h.dtype


def test_matrix_is_built_once(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=True, renorm=True, device="cpu")
    h = torch.randn(1, tiny_qwen3.config.hidden_size)
    proj(h); proj(h)
    assert proj.builds == 1, "matrix rebuilt per call -- D x D solve is expensive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_realign.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.realign'`.

- [ ] **Step 3: Write the implementation**

`lkv/realign.py`:
```python
"""LatentMAS latent-space realignment, with the renorm split out.

The matrix is the ridge-regularized least-squares solution of E_out W ~= E_in:
    W = (E_out^T E_out + 1e-5 I)^-1 (E_out^T E_in)
solved in fp32, exactly as stock-latentmas/models.py:158-185. target_norm is the
mean L2 row norm of the INPUT embedding (models.py:177).

Upstream applies the renorm on every latent step regardless of the
--latent_space_realign flag; here `realign` gates the matmul and `renorm` gates
the rescale, independently, so either can be ablated without the other.
"""
from __future__ import annotations

import torch


def build_realign_matrix(model, device, dtype: torch.dtype = torch.float32):
    input_embeds = model.get_input_embeddings()
    output_embeds = model.get_output_embeddings() or getattr(model, "lm_head", None)
    if input_embeds is None or output_embeds is None:
        raise RuntimeError("cannot build realign matrix: embedding weights not accessible")

    e_in = input_embeds.weight.detach().to(device=device, dtype=dtype)
    e_out = output_embeds.weight.detach().to(device=device, dtype=dtype)

    gram = e_out.T @ e_out
    gram = gram + 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = e_out.T @ e_in
    w = torch.linalg.solve(gram, rhs)
    target_norm = e_in.norm(dim=1).mean().detach()
    return w, target_norm


class LatentProjector:
    """Applies (optionally) the realign matmul and (optionally) the renorm."""

    def __init__(self, model, realign: bool, renorm: bool, device):
        self._model = model
        self.realign = realign
        self.renorm = renorm
        self.device = device
        self._w = None
        self._target_norm = None
        self.builds = 0

    def _ensure(self):
        if self._w is None and (self.realign or self.renorm):
            self._w, self._target_norm = build_realign_matrix(self._model, self.device)
            self.builds += 1

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.realign and not self.renorm:
            return hidden
        self._ensure()
        out_dtype = hidden.dtype
        x = hidden.to(torch.float32)
        if self.realign:
            x = x @ self._w.to(x.device)
        if self.renorm:
            scale = self._target_norm.to(x.device) / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x * scale
        return x.to(out_dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_realign.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add lkv/realign.py tests/test_realign.py
git commit -m "feat: split realign matmul from latent renorm (D4)"
```

---

### Task 6: Prompts — fixes D6, records D11

Port LatentMAS's four sequential role prompts and KVComm's three templates. Two notes:

- **D6:** upstream truncates the TextMAS context with `context[: args.text_mas_context_length]` where the default is `-1`, silently dropping the last character. We use `None` for "no truncation".
- **D11:** LatentMAS interpolates the full question into all four role prompts and never uses the `context` parameter of the latent builders (`stock-latentmas/prompts.py:12,20,36,51`; `methods/latent_mas.py:95,100`). This is **preserved deliberately** — removing it would no longer be the system the paper measured. It is recorded here so the redundancy is a known property, not an accident.

**Files:**
- Create: `lkv/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ROLES: tuple[str, ...] == ("planner", "critic", "refiner", "judger")`
  - `latent_agent_message(role: str, question: str, task: str) -> list[dict]` returning chat messages (`[{"role": "system"|"user", "content": str}, ...]`).
  - `truncate_context(context: str, limit: int | None) -> str`.
  - `kvcomm_sender_prompt(instruction: str, context: str) -> str`, `kvcomm_receiver_prompt(instruction: str, question: str) -> str`, `kvcomm_skyline_prompt(instruction: str, context: str, question: str) -> str`.
  - Task 8 consumes `ROLES` and `latent_agent_message`; Task 14 consumes the three `kvcomm_*` functions.

- [ ] **Step 1: Write the failing test**

`tests/test_prompts.py`:
```python
import pytest

from lkv.prompts import (
    ROLES,
    kvcomm_receiver_prompt,
    kvcomm_sender_prompt,
    kvcomm_skyline_prompt,
    latent_agent_message,
    truncate_context,
)


def test_roles_are_the_paper_four_in_order():
    assert ROLES == ("planner", "critic", "refiner", "judger")


@pytest.mark.parametrize("role", ROLES)
def test_every_role_embeds_the_question(role):
    """D11: LatentMAS re-supplies the question as text to all four agents.
    Preserved on purpose -- removing it would not be LatentMAS."""
    msgs = latent_agent_message(role, "What is 2+2?", task="gsm8k")
    joined = " ".join(m["content"] for m in msgs)
    assert "What is 2+2?" in joined
    assert msgs[-1]["role"] == "user"
    assert all(m["role"] in ("system", "user") for m in msgs)


def test_judger_prompt_requests_boxed_answer_for_gsm8k():
    msgs = latent_agent_message("judger", "q", task="gsm8k")
    assert "\\boxed" in " ".join(m["content"] for m in msgs)


def test_judger_prompt_requests_python_block_for_mbppplus():
    msgs = latent_agent_message("judger", "q", task="mbppplus")
    assert "```python" in " ".join(m["content"] for m in msgs)


def test_truncate_context_none_means_no_truncation():
    """D6: upstream's -1 default silently drops the last character."""
    text = "abcdef"
    assert truncate_context(text, None) == text
    assert truncate_context(text, 3) == "abc"
    with pytest.raises(ValueError):
        truncate_context(text, -1)


def test_kvcomm_templates_separate_context_from_question():
    s = kvcomm_sender_prompt("INSTR", "CTX")
    r = kvcomm_receiver_prompt("INSTR", "Q")
    k = kvcomm_skyline_prompt("INSTR", "CTX", "Q")
    assert "CTX" in s and "Q" not in s
    assert "Q" in r and "CTX" not in r
    assert "CTX" in k and "Q" in k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.prompts'`.

- [ ] **Step 3: Write the implementation**

`lkv/prompts.py`:
```python
"""Role prompts for LatentMAS and the three KVComm templates.

LatentMAS interpolates the full question into every role's prompt and never
uses the `context` argument of its latent builders (stock-latentmas/prompts.py:
12,20,36,51). That duplication is preserved deliberately -- it is a property of
the system the paper measured, not a bug to fix here.
"""
from __future__ import annotations

ROLES = ("planner", "critic", "refiner", "judger")

_SYSTEM = (
    "You are one agent in a multi-agent system. Agents share their reasoning "
    "through a latent cache rather than text. Play only your own role."
)

_ROLE_BODY = {
    "planner": (
        "You are the Planner. Devise a step-by-step plan to solve the problem. "
        "Do not produce the final answer.\n\nProblem:\n{question}"
    ),
    "critic": (
        "You are the Critic. Examine the plan for errors, missing cases, and "
        "wrong assumptions. Do not produce the final answer.\n\nProblem:\n{question}"
    ),
    "refiner": (
        "You are the Refiner. Repair the plan in light of the critique and "
        "tighten it. Do not produce the final answer.\n\nProblem:\n{question}"
    ),
}

_JUDGER_TAIL = {
    "gsm8k": "Give the final numeric answer inside \\boxed{}.",
    "hotpotqa": "Answer with the shortest exact span. No explanation.",
    "countries": "Answer with the country name only.",
    "tipsheets": "Answer with the requested value only.",
    "mbppplus": (
        "Provide a self-contained Python script in a markdown code block:\n"
        "```python\nYOUR_PYTHON_CODE\n```"
    ),
}


def latent_agent_message(role: str, question: str, task: str) -> list[dict]:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    if role == "judger":
        try:
            tail = _JUDGER_TAIL[task]
        except KeyError:
            raise ValueError(f"no judger instruction for task {task!r}") from None
        body = (
            "You are the Judger. Using the shared latent context, produce the "
            f"final answer.\n\nProblem:\n{question}\n\n{tail}"
        )
    else:
        body = _ROLE_BODY[role].format(question=question)
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": body}]


def truncate_context(context: str, limit: int | None) -> str:
    """D6: upstream passes -1 to mean 'unlimited', which drops the last char."""
    if limit is None:
        return context
    if limit < 0:
        raise ValueError("limit must be None for unlimited, or a non-negative int")
    return context[:limit]


# KVComm eval.py:25-32 splits context (sender) from question (receiver).
def kvcomm_sender_prompt(instruction: str, context: str) -> str:
    return f"Instruction: {instruction} Context: {context}"


def kvcomm_receiver_prompt(instruction: str, question: str) -> str:
    return f"Instruction: {instruction} Question: {question}"


def kvcomm_skyline_prompt(instruction: str, context: str, question: str) -> str:
    return f"Instruction: {instruction} Context: {context} Question: {question}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: PASS, 9 tests (4 parametrized + 5).

- [ ] **Step 5: Commit**

```bash
git add lkv/prompts.py tests/test_prompts.py
git commit -m "feat: role and KVComm prompts; None means unlimited context (D6)"
```

---

### Task 7: Datasets and scoring — surfaces D10

The metric ambiguity must be implemented as *two* functions, not resolved by guesswork. KVComm's code computes `f1_match(...) -> bool` returning `max(f1) > 0.5`, so its reported score is the **fraction of examples above threshold** (`stock-kvcomm/utils/f1.py:65`, `base_evaluator.py:36-48`). The paper §4.1 says plain "F1 score". Both are computed and both are written to the output JSON; Task 15's gate decides which reproduces the published tables. Evidence favouring the thresholded reading: Table 6 Tipsheets shows KVComm(0.3)=0.93 exceeding Skyline=0.92, which is easier under a threshold.

**Files:**
- Create: `lkv/scoring.py`, `lkv/lemma.py`, `lkv/data/__init__.py` (replace stub), `lkv/data/gsm8k.py`, `lkv/data/mbppplus.py`, `lkv/data/hotpotqa.py`, `lkv/data/countries.py`
- Modify: `requirements.txt` (add `nltk>=3.8`)
- Test: `tests/test_scoring.py`

`lkv/lemma.py` isolates the only nltk dependency in the codebase, so the corpora
download happens in one place and the failure mode is one clear exception:

```python
"""WordNet lemmatization, matching stock-kvcomm/utils/metric_tools.py:188-210.

Isolated in its own module because it is the codebase's only nltk dependency and
it needs corpora that must be present before the first call. It raises rather
than degrading to unlemmatized text: a silent fallback would change the metric
while still reporting it as KVComm's metric.
"""
from __future__ import annotations

import functools

_REQUIRED = ("punkt", "punkt_tab", "wordnet", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng")


@functools.lru_cache(maxsize=1)
def _setup():
    import nltk
    from nltk.stem import WordNetLemmatizer
    for pkg in _REQUIRED:
        nltk.download(pkg, quiet=True)
    return nltk, WordNetLemmatizer()


@functools.lru_cache(maxsize=100_000)
def _lemma_one(word: str) -> str:
    nltk, lemmatizer = _setup()
    from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    pos = {"J": wordnet.ADJ, "N": wordnet.NOUN,
           "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)
    return lemmatizer.lemmatize(word, pos)


def lemmatize_text(text: str) -> str:
    nltk, _ = _setup()
    from nltk.tokenize import word_tokenize
    return " ".join(_lemma_one(w) for w in word_tokenize(text))
```

The per-word `lru_cache` matters: upstream calls `nltk.pos_tag` once per token,
which dominates scoring time on a 500-item run.

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_task(name: str, max_samples: int = -1, seed: int = 42) -> list[dict]`. Every item has keys `question: str`, `gold: str`, and for KVComm tasks additionally `context: str` and `instruction: str`. MBPP+ items carry `gold` as executable test code.
  - `extract_gsm8k_answer(text: str) -> str | None`, `extract_python_block(text: str) -> str | None`, `normalize(text: str | None) -> str | None`.
  - `run_python(code: str, timeout: int = 10) -> tuple[bool, str | None]`.
  - `f1_score(reference: str, candidate: str) -> float` and `f1_match(reference: str, candidate: str, threshold: float = 0.5) -> bool`.
  - `score_item(task: str, prediction: str, item: dict) -> tuple[bool, float]` returning `(correct, continuous_score)`.
  - Tasks 8, 9, 14 consume `load_task` and `score_item`.

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:
```python
"""D10: KVComm's code metric (fraction with F1 > 0.5) differs from the paper's
stated 'F1 score'. Both are implemented; Task 15 decides which reproduces."""
from lkv.scoring import (
    extract_gsm8k_answer,
    extract_python_block,
    f1_match,
    f1_score,
    normalize,
    run_python,
    score_item,
)


def test_boxed_answer_wins_over_trailing_numbers():
    assert extract_gsm8k_answer("blah \\boxed{42} then 99 more") == "42"


def test_falls_back_to_last_number():
    assert extract_gsm8k_answer("first 7 then 12") == "12"


def test_returns_none_when_no_number():
    assert extract_gsm8k_answer("no digits here") is None


def test_comma_formatted_gold_is_normalized():
    """Upstream's regex cannot span '1,000'; ours must."""
    assert normalize(extract_gsm8k_answer("\\boxed{1,000}")) == "1000"


def test_last_python_block_is_taken():
    text = "```python\nx=1\n```\ntext\n```python\ny=2\n```"
    assert extract_python_block(text) == "y=2"


def test_run_python_success_and_failure():
    ok, err = run_python("assert 1 == 1")
    assert ok is True and err is None
    ok, err = run_python("assert 1 == 2")
    assert ok is False and "AssertionError" in err


def test_run_python_times_out():
    ok, err = run_python("while True: pass", timeout=2)
    assert ok is False and "Timeout" in err


def test_f1_is_continuous_and_match_is_boolean():
    assert f1_score("the red car", "the red car") == 1.0
    partial = f1_score("the red car", "the red truck")
    assert 0.0 < partial < 1.0
    assert f1_match("the red car", "the red car") is True
    assert f1_match("the red car", "completely different words") is False


def test_f1_match_threshold_is_strictly_greater():
    """utils/f1.py:65 uses `> threshold`, not `>=`."""
    assert f1_match("a b", "a b", threshold=1.0) is False


def test_score_item_gsm8k():
    correct, score = score_item("gsm8k", "so \\boxed{18}", {"gold": "18"})
    assert correct is True and score == 1.0
    correct, score = score_item("gsm8k", "so \\boxed{17}", {"gold": "18"})
    assert correct is False and score == 0.0


def test_score_item_mbppplus_executes_gold_tests():
    item = {"gold": "assert add(1, 2) == 3"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert score_item("mbppplus", good, item)[0] is True
    assert score_item("mbppplus", bad, item)[0] is False
    assert score_item("mbppplus", "no code block", item)[0] is False


def test_score_item_hotpotqa_returns_both_metrics():
    correct, score = score_item("hotpotqa", "the red car", {"gold": "red car"})
    assert isinstance(correct, bool) and 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.scoring'`.

- [ ] **Step 3: Write scoring**

`lkv/scoring.py`:
```python
"""Answer extraction, sandboxed execution, and both F1 variants.

score_item returns (correct, score): `correct` is the binary outcome that goes
into per_item and McNemar; `score` is the continuous value so a mean-F1 metric
can be computed from the same run. For GSM8K and MBPP+ they coincide.
"""
from __future__ import annotations

import re
import traceback
from multiprocessing import Manager, Process

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_PY_BLOCK = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_gsm8k_answer(text: str) -> str | None:
    boxes = _BOXED.findall(text)
    if boxes:
        m = _NUMBER.search(boxes[-1])
        return m.group(0) if m else boxes[-1].strip()
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


def extract_gold_gsm8k(solution: str) -> str | None:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", solution)
    return m.group(1) if m else None


def normalize(text: str | None) -> str | None:
    if text is None:
        return None
    out = text.strip().lower().replace(",", "")
    return out.rstrip(".") or None


def extract_python_block(text: str) -> str | None:
    blocks = _PY_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else None


def _worker(ns, code):
    try:
        exec(code, {})
        ns["ok"], ns["error"] = True, None
    except BaseException:
        ns["ok"], ns["error"] = False, traceback.format_exc()


def run_python(code: str, timeout: int = 10):
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=_worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False, f"TimeoutError: exceeded {timeout} seconds"
        return bool(ns.get("ok", False)), ns.get("error", "process died")


_PUNCT = re.compile(r"[^a-z0-9 ]")
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace.

    Mirrors stock-kvcomm/utils/metric_tools.py:43-59 (`normalize_answer`, which
    then calls `fix_answer` at :27-41).
    """
    cleaned = _PUNCT.sub(" ", str(text).lower())
    return " ".join(t for t in cleaned.split() if t not in _ARTICLES)


def _lemmatize(text: str) -> str:
    """WordNet lemmatization with POS tagging, as upstream does.

    stock-kvcomm/utils/f1.py:12-13 applies `lemmatize_text(normalize_answer(x))`
    to BOTH sides before comparing, using nltk's WordNetLemmatizer with a POS tag
    per token (metric_tools.py:188-210). Skipping it lowers F1 systematically for
    every condition, which would make the K1/K2 gates miss for a reason that has
    nothing to do with KV transfer.

    If the nltk corpora are unavailable this raises rather than silently
    returning unlemmatized text -- a quiet fallback would change the metric
    without changing the reported metric name.
    """
    from lkv import lemma
    return lemma.lemmatize_text(text)


def _tokens(text: str) -> list[str]:
    return _lemmatize(normalize_answer(text)).split()


def f1_score(reference: str, candidate: str) -> float:
    """Word-set F1. Upstream's `precision`/`recall` are transposed relative to
    the usual convention (stock-kvcomm/utils/f1.py:18-27 divides by the
    reference and candidate sizes respectively), but F1 is symmetric in the two,
    so the value is identical either way."""
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return float(ref == cand)
    common = set(ref) & set(cand)
    if not common:
        return 0.0
    precision = len(common) / len(set(cand))
    recall = len(common) / len(set(ref))
    return 2 * precision * recall / (precision + recall)


def f1_match(reference: str, candidate: str, threshold: float = 0.5) -> bool:
    """KVComm's implemented metric: utils/f1.py:65 uses strict `>`."""
    return f1_score(reference, candidate) > threshold


def score_item(task: str, prediction: str, item: dict) -> tuple[bool, float]:
    gold = item.get("gold", "")
    if task == "gsm8k":
        pred = normalize(extract_gsm8k_answer(prediction))
        ok = bool(pred) and bool(normalize(gold)) and pred == normalize(gold)
        return ok, float(ok)
    if task == "mbppplus":
        code = extract_python_block(prediction)
        if code is None:
            return False, 0.0
        ok, _ = run_python(code + "\n" + gold)
        return ok, float(ok)
    if task in ("hotpotqa", "countries", "tipsheets"):
        return f1_match(gold, prediction), f1_score(gold, prediction)
    raise ValueError(f"no scorer for task {task!r}")
```

- [ ] **Step 4: Write the loaders, then run the test**

`lkv/data/gsm8k.py`:
```python
from datasets import load_dataset

from lkv.scoring import extract_gold_gsm8k, normalize


def load(split: str = "test"):
    ds = load_dataset("gsm8k", "main", split=split)
    return [
        {"question": row["question"].strip(),
         "gold": normalize(extract_gold_gsm8k(row["answer"]))}
        for row in ds
    ]
```

`lkv/data/mbppplus.py`:
```python
from datasets import load_dataset

_TEMPLATE = """Please provide a self-contained Python script that solves the following problem in a markdown code block:
```python
YOUR_PYTHON_CODE
```:
{prompt}
Your answer will be tested on test cases like:
{t0}
{t1}
{t2}
"""


def load(split: str = "test"):
    ds = load_dataset("evalplus/mbppplus", split=split)
    items = []
    for row in ds:
        tests = list(row["test_list"])[:3]
        while len(tests) < 3:
            tests.append("")
        items.append({
            "question": _TEMPLATE.format(prompt=row["prompt"], t0=tests[0],
                                         t1=tests[1], t2=tests[2]),
            "gold": str(row["test"]),
        })
    return items
```

`lkv/data/hotpotqa.py`:
```python
"""HotpotQA, matching KVComm's own context construction.

CRITICAL: the sender's context is ONLY the supporting-fact sentences, not the
whole distractor paragraph set. Upstream builds it with `construct_support`
(stock-kvcomm/dataloader/hotpotqa.py:4-14): for each (title, sent_id) in
`supporting_facts`, pull that one sentence out of the matching context
paragraph, then join with newlines (`:34-35`, `prompt_A`). Handing the model all
ten paragraphs instead would make the context roughly an order of magnitude
larger and mostly distractors, which would depress every KVComm score in a way
that looks like the method failing rather than the harness being wrong.
"""
from datasets import load_dataset

INSTRUCTION = "Answer the question using the given context."


def construct_support(row) -> list[str]:
    """The supporting-fact sentences only, in supporting_facts order."""
    facts = row["supporting_facts"]
    ctx = row["context"]
    out = []
    for title, sent_id in zip(facts["title"], facts["sent_id"]):
        for ctx_title, sentences in zip(ctx["title"], ctx["sentences"]):
            if ctx_title == title:
                if 0 <= sent_id < len(sentences):
                    out.append(sentences[sent_id])
                # Upstream prints a warning and drops it; a missing sentence is
                # a dataset artifact, not something to fail the run over.
    return out


def load(split: str = "validation"):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    items = []
    for row in ds:
        items.append({
            "instruction": INSTRUCTION,
            "context": "\n".join(construct_support(row)),
            "question": row["question"],
            "gold": row["answer"],
        })
    return items
```

`lkv/data/countries.py`:
```python
"""KVComm bundles countries.jsonl (200 items) in-repo rather than on the Hub.

Place a copy at lkv/data/files/countries.jsonl with one object per line:
    {"context": "...", "question": "...", "answer": "..."}
Fail loudly if it is missing -- silently returning [] would report accuracy 0.0
as if it were a measurement.
"""
import json
from pathlib import Path

INSTRUCTION = "Answer the question using the given context."
_PATH = Path(__file__).parent / "files" / "countries.jsonl"


def load(split: str = "test"):
    if not _PATH.exists():
        raise FileNotFoundError(
            f"{_PATH} missing. Copy it from the KVComm repo "
            "(dataloader/data/countries.jsonl)."
        )
    items = []
    with _PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            items.append({
                "instruction": INSTRUCTION,
                "context": row["context"],
                "question": row["question"],
                "gold": row["answer"],
            })
    return items
```

`lkv/data/__init__.py`:
```python
"""Task loading. max_samples slices deterministically after a seeded shuffle so
that two runs with the same seed compare item-for-item."""
from __future__ import annotations

import random

from lkv.data import countries, gsm8k, hotpotqa, mbppplus

_LOADERS = {
    "gsm8k": gsm8k.load,
    "mbppplus": mbppplus.load,
    "hotpotqa": hotpotqa.load,
    "countries": countries.load,
}

# Paper Table 4 sample counts, used when max_samples is -1.
PAPER_N = {"countries": 200, "tipsheets": 500, "hotpotqa": 500}


def load_task(name: str, max_samples: int = -1, seed: int = 42) -> list[dict]:
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; have {sorted(_LOADERS)}") from None
    items = loader()
    target = max_samples if max_samples > 0 else PAPER_N.get(name, -1)
    if target > 0 and target < len(items):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(items)), target))
        items = [items[i] for i in idx]
    return items
```

Run: `./.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: PASS, 12 tests. The loaders are not covered by CPU tests (they need network); they are exercised by the Phase 2 and Phase 4 gates.

- [ ] **Step 5: Commit**

```bash
git add lkv/scoring.py lkv/data/ tests/test_scoring.py
git commit -m "feat: scoring with both F1 variants and task loaders (D10)"
```

---

### Task 8: The LatentMAS relay loop — fixes D2, D5, D7

**Files:**
- Create: `lkv/latentmas.py`
- Test: `tests/test_latentmas_loop.py`

**Interfaces:**
- Consumes: `load_model`/`pad_batch` (Task 3), `generate_text_batch` (Task 3), `LatentProjector` (Task 5), `ROLES`/`latent_agent_message` (Task 6), `RunConfig` (Task 4).
- Produces:
  - `past_length(past_kv) -> int` — reads **every** layer and returns the max, so a collapsed layer cannot hide (upstream reads layer 0 only).
  - `generate_latent(model, projector, input_ids, attention_mask, latent_steps, past_key_values, past_mask=None) -> tuple[cache, Tensor, int]` returning the extended cache, the running attention mask covering the WHOLE cache, and the number of positions added. The mask must be threaded, never synthesized: the cache is built from left-padded ragged prompts, so its pad positions have to stay masked on every later hop.
  - `run_latentmas(model, tokenizer, cfg, items, trim_hook=None) -> tuple[list[str], list[dict]]` returning final texts and per-hop records. `trim_hook` has signature `(cache, hop_index: int, inherited_len: int) -> cache`; when `None`, nothing is trimmed. **This hook is the entire integration surface for Phase 5** — no KVComm code appears in this module.
  - Task 9 consumes `run_latentmas`; Task 17 supplies `trim_hook`.

- [ ] **Step 1: Write the failing test**

`tests/test_latentmas_loop.py`:
```python
"""The relay loop. Assertions are structural, so random weights are fine."""
import torch

from lkv.config import RunConfig
from lkv.latentmas import generate_latent, past_length, run_latentmas
from lkv.realign import LatentProjector


class _StubTok:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=None):
        seqs = [[7, 8, 9] for _ in texts]
        return {"input_ids": torch.tensor(seqs),
                "attention_mask": torch.ones(len(seqs), 3, dtype=torch.long)}

    def decode(self, t, skip_special_tokens=True):
        return "answer \\boxed{4}"


def _cfg(**kw):
    base = dict(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                latent_steps=5, n_agents=4, greedy=True, max_new_tokens=4,
                device="cpu", generate_bs=2)
    base.update(kw)
    return RunConfig(**base)


def test_past_length_reads_every_layer(tiny_qwen3):
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        cache = tiny_qwen3(ids, use_cache=True).past_key_values
    assert past_length(cache) == 4
    # collapse a non-zero layer; upstream reads layer 0 only and would miss this
    cache.key_cache[2] = cache.key_cache[2][:, :, :1, :]
    cache.value_cache[2] = cache.value_cache[2][:, :, :1, :]
    assert past_length(cache) == 4
    assert min(k.shape[-2] for k in cache.key_cache) == 1


def test_generate_latent_adds_prompt_plus_steps(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    ids = torch.tensor([[3, 4, 5]])
    mask = torch.ones_like(ids)
    cache, added = generate_latent(tiny_qwen3, proj, ids, mask, latent_steps=5,
                                   past_key_values=None)
    assert added == 3 + 5
    assert past_length(cache) == 3 + 5


def test_generate_latent_zero_steps_adds_only_the_prompt(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    ids = torch.tensor([[3, 4, 5]])
    cache, added = generate_latent(tiny_qwen3, proj, ids, torch.ones_like(ids),
                                   latent_steps=0, past_key_values=None)
    assert added == 3


def test_relay_runs_three_latent_agents_then_judges(tiny_qwen3):
    items = [{"question": "2+2?", "gold": "4"}]
    texts, hops = run_latentmas(tiny_qwen3, _StubTok(), _cfg(), items)
    assert len(texts) == 1
    # 4 agents: 3 latent hops + 1 judger hop
    assert [h["role"] for h in hops] == ["planner", "critic", "refiner", "judger"]
    assert sum(h["latent_steps"] for h in hops) == 3 * 5


def test_cache_grows_monotonically_without_a_hook(tiny_qwen3):
    items = [{"question": "2+2?", "gold": "4"}]
    _, hops = run_latentmas(tiny_qwen3, _StubTok(), _cfg(), items)
    lens = [h["positions_after"] for h in hops]
    assert lens == sorted(lens), lens
    assert all(h["positions_before"] == h["positions_after"] for h in hops) is False


def test_trim_hook_is_called_once_per_boundary(tiny_qwen3):
    calls = []

    def hook(cache, hop_index, inherited_len):
        calls.append((hop_index, inherited_len))
        return cache

    items = [{"question": "2+2?", "gold": "4"}]
    run_latentmas(tiny_qwen3, _StubTok(), _cfg(), items, trim_hook=hook)
    # a boundary exists after every agent except the last
    assert [c[0] for c in calls] == [0, 1, 2]


def test_n_agents_one_still_reaches_the_judger(tiny_qwen3):
    """D2 guard: with one agent there is no latent hop, and the judger must
    still run rather than silently receiving a discarded cache."""
    items = [{"question": "2+2?", "gold": "4"}]
    texts, hops = run_latentmas(tiny_qwen3, _StubTok(), _cfg(n_agents=1), items)
    assert [h["role"] for h in hops] == ["judger"]
    assert texts and texts[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_latentmas_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.latentmas'`.

- [ ] **Step 3: Write the implementation**

`lkv/latentmas.py`:
```python
"""LatentMAS: one model plays N roles over a single growing KV cache.

Differences from stock-latentmas/methods/latent_mas.py, all deliberate:
  D2  The relay cache is ALWAYS passed to the judger. Upstream nulls it when
      latent_steps == 0 (`past_for_decoding = past_kv if self.latent_steps > 0
      else None`, :158), which turns the configuration into a single-agent run
      that still reports itself as LatentMAS.
  D5  Truncation is not a dead flag pair. It is a `trim_hook` argument, which
      is also the whole integration surface for KVComm -- no kvcomm import here.
  D7  No vLLM import. Upstream imports SamplingParams unguarded at module scope
      (:9), so the pure-HF path cannot run without vLLM installed.
  Also: past_length() takes the max over ALL layers. Upstream reads layer 0,
  which KVComm always keeps full -- so a layer-trimmed cache would look intact.
"""
from __future__ import annotations

import torch

from lkv.decode import generate_text_batch
from lkv.modeling import pad_batch
from lkv.prompts import ROLES, latent_agent_message


def past_length(past_kv) -> int:
    if past_kv is None:
        return 0
    return max(int(k.shape[-2]) for k in past_kv.key_cache)


def generate_latent(model, projector, input_ids, attention_mask, latent_steps,
                    past_key_values, past_mask=None):
    """Prefill the prompt, then roll `latent_steps` last-hidden-states back in.

    Returns (cache, running_mask, positions_added). `running_mask` spans the whole
    cache and MUST be threaded into the next hop: the prompts are left-padded and
    ragged, so those pad positions have to stay masked for the rest of the relay.
    Synthesizing ones later re-enables attention to them and silently changes the
    output -- measured max logit delta ~0.53 on the tiny fixture.
    """
    before = past_length(past_key_values)
    if before and past_mask is None:
        raise ValueError(
            f"past_mask is required when the cache already holds {before} "
            "positions; the relay's earlier prompts were left-padded."
        )
    # The mask handed to the model must cover cache + this prompt.
    mask = attention_mask if not before else torch.cat(
        [past_mask.to(dtype=attention_mask.dtype, device=attention_mask.device),
         attention_mask], dim=1)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        cache = out.past_key_values
        for _ in range(latent_steps):
            hidden = out.hidden_states[-1][:, -1, :]
            embed = projector(hidden).unsqueeze(1)
            mask = torch.cat(
                [mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)],
                dim=1,
            )
            out = model(
                inputs_embeds=embed,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = out.past_key_values
    return cache, mask, past_length(cache) - before


def _render(tokenizer, cfg, items, role):
    texts = [
        tokenizer.apply_chat_template(
            latent_agent_message(role, item["question"], cfg.task),
            tokenize=False, add_generation_prompt=True,
        )
        for item in items
    ]
    return pad_batch(tokenizer, texts)


def run_latentmas(model, tokenizer, cfg, items, trim_hook=None, projector=None):
    """Returns (final_texts, hop_records).

    trim_hook(cache, hop_index, inherited_len) -> cache is called after each
    non-final agent. It is the only place KVComm enters the relay.
    """
    if projector is None:
        from lkv.realign import LatentProjector
        projector = LatentProjector(model, cfg.realign, cfg.renorm_latent, cfg.device)

    roles = list(ROLES[: cfg.n_agents - 1]) + ["judger"] if cfg.n_agents > 1 else ["judger"]
    cache = None
    run_mask = None          # spans the whole cache; threaded, never synthesized
    hops: list[dict] = []
    final_texts: list[str] = []

    for hop_index, role in enumerate(roles):
        ids, mask = _render(tokenizer, cfg, items, role)
        ids = ids.to(model.device)
        mask = mask.to(model.device)
        before = past_length(cache)

        if role == "judger":
            # Pass the judger prompt ALONE. generate_text_batch owns widening for
            # the cached span -- see its contract. Hand-rolling the mask here is
            # what the plan originally specified, and it raised IndexError,
            # because generate also needs input_ids to span the cache.
            final_texts, _ = generate_text_batch(
                model, tokenizer, ids, mask,
                max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy,
                temperature=cfg.temperature, top_p=cfg.top_p,
                past_key_values=cache,   # D2: never nulled
                past_mask=run_mask,      # the relay's real mask, not ones
            )
            hops.append({"hop": hop_index, "role": role, "latent_steps": 0,
                         "positions_before": before, "positions_after": before})
            continue

        cache, run_mask, _ = generate_latent(
            model, projector, ids, mask, cfg.latent_steps, cache, past_mask=run_mask
        )
        after = past_length(cache)
        if trim_hook is not None:
            cache = trim_hook(cache, hop_index, before)
            # A trim shortens the cache, so the running mask must be re-cut to
            # match or the next hop's widths disagree. Trimming keeps a PREFIX of
            # each layer, so the surviving mask is the matching prefix.
            run_mask = run_mask[:, : past_length(cache)]
        hops.append({"hop": hop_index, "role": role, "latent_steps": cfg.latent_steps,
                     "positions_before": before, "positions_after": after,
                     "positions_after_trim": past_length(cache)})

    return final_texts, hops
```

Note on the judger mask: `generate_text_batch` prepends the cached span to both
tensors itself, so the loop passes the judger prompt unchanged. The earlier
hand-rolled concatenation here was removed after it was shown to raise
IndexError -- `generate` needs `input_ids` to span the cache, not just the mask.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_latentmas_loop.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add lkv/latentmas.py tests/test_latentmas_loop.py
git commit -m "feat: LatentMAS relay with trim hook; cache never nulled (D2, D5, D7)"
```

---

### Task 9: Entry point, result records, and paired statistics

**Files:**
- Create: `lkv/stats.py`, `lkv/run.py`
- Test: `tests/test_stats.py`, `tests/test_run_record.py`

**Interfaces:**
- Consumes: everything from Tasks 3–8.
- Produces:
  - `mcnemar(a: list[bool], b: list[bool]) -> dict` with keys `n01`, `n10`, `chi2`, `p_exact`.
  - `bootstrap_ci(scores: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]`.
  - `build_record(cfg, items, predictions, per_item, hops, timings) -> dict` producing the JSON described in the plan header, with `per_item` always present.
  - `main(argv=None) -> int` CLI. Task 10 onward invokes `python -m lkv.run`.

- [ ] **Step 1: Write the failing test**

`tests/test_stats.py`:
```python
import math

from lkv.stats import bootstrap_ci, mcnemar


def test_mcnemar_counts_discordant_pairs_only():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    r = mcnemar(a, b)
    assert r["n10"] == 2   # a right, b wrong (indices 1, 4)
    assert r["n01"] == 1   # b right, a wrong (index 2)


def test_mcnemar_matches_hand_computed_chi2():
    """With continuity correction: (|n01-n10|-1)^2 / (n01+n10).
    n01=5, n10=20 -> (15-1)^2/25 = 7.84"""
    a = [True] * 20 + [False] * 5 + [True] * 10
    b = [False] * 20 + [True] * 5 + [True] * 10
    r = mcnemar(a, b)
    assert r["n10"] == 20 and r["n01"] == 5
    assert math.isclose(r["chi2"], 7.84, abs_tol=1e-6)
    assert r["p_exact"] < 0.01


def test_mcnemar_no_discordance_is_p_one():
    r = mcnemar([True, False], [True, False])
    assert r["n01"] == 0 and r["n10"] == 0
    assert r["chi2"] == 0.0 and r["p_exact"] == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    scores = [0.0] * 50 + [1.0] * 50
    lo, hi = bootstrap_ci(scores, n_boot=2000, seed=7)
    assert lo < 0.5 < hi
    assert bootstrap_ci(scores, n_boot=2000, seed=7) == (lo, hi)
```

`tests/test_run_record.py`:
```python
from lkv.config import RunConfig
from lkv.run import build_record


def _cfg():
    return RunConfig(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                     latent_steps=40, greedy=True, device="cpu")


def test_record_has_per_item_and_both_metrics():
    per_item = [{"i": 0, "correct": True, "score": 1.0},
                {"i": 1, "correct": False, "score": 0.25}]
    rec = build_record(_cfg(), items=[{}, {}], predictions=["a", "b"],
                       per_item=per_item, hops=[], timings={"total_sec": 2.0})
    assert rec["n"] == 2
    assert rec["accuracy"] == 0.5
    assert rec["mean_score"] == 0.625
    assert rec["per_item"] == per_item
    assert rec["sec_per_item"] == 1.0
    assert rec["latent_steps"] == 40 and rec["greedy"] is True


def test_record_refuses_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        build_record(_cfg(), items=[{}, {}], predictions=["a"],
                     per_item=[{"i": 0, "correct": True, "score": 1.0}],
                     hops=[], timings={"total_sec": 1.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_stats.py tests/test_run_record.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.stats'`.

- [ ] **Step 3: Write the implementation**

`lkv/stats.py`:
```python
"""Paired tests. Two runs on the same seeded slice are paired, so McNemar is
the correct instrument -- an unpaired z-test on the same data needs a much
larger difference to reach the same significance."""
from __future__ import annotations

import random

from scipy.stats import binomtest


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    total = n01 + n10
    if total == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p_exact": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / total
    p = binomtest(n10, total, 0.5).pvalue
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_exact": float(p)}


def bootstrap_ci(scores: list[float], n_boot: int = 10000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    if not scores:
        raise ValueError("empty scores")
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        means.append(sum(scores[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1.0 - level) / 2.0
    return means[int(tail * n_boot)], means[min(n_boot - 1, int((1 - tail) * n_boot))]
```

`lkv/run.py`:
```python
"""Entry point. One JSON per run, always including per_item."""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import torch

from lkv.config import RunConfig, parse_args
from lkv.data import load_task
from lkv.latentmas import run_latentmas
from lkv.modeling import load_model
from lkv.scoring import score_item


def build_record(cfg: RunConfig, items, predictions, per_item, hops, timings) -> dict:
    n = len(items)
    if not (len(predictions) == len(per_item) == n):
        raise ValueError(
            f"length mismatch: items={n} predictions={len(predictions)} "
            f"per_item={len(per_item)}"
        )
    correct = sum(1 for p in per_item if p["correct"])
    return {
        **{k: v for k, v in asdict(cfg).items() if k != "extra"},
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "mean_score": (sum(p["score"] for p in per_item) / n) if n else 0.0,
        "per_item": per_item,
        "hop_records": hops,
        "total_sec": timings["total_sec"],
        "sec_per_item": (timings["total_sec"] / n) if n else 0.0,
    }


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def main(argv=None) -> int:
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    items = load_task(cfg.task, cfg.max_samples, cfg.seed)
    model, tokenizer = load_model(cfg.model_name, cfg.device)

    predictions: list[str] = [""] * len(items)
    per_item: list[dict] = [None] * len(items)
    hops: list[dict] = []
    started = time.time()

    for offset, batch in _batches(items, cfg.generate_bs):
        if cfg.system in ("latentmas", "single"):
            local_cfg = cfg if cfg.system == "latentmas" else RunConfig(
                **{**{k: v for k, v in asdict(cfg).items()}, "n_agents": 1}
            )
            texts, batch_hops = run_latentmas(model, tokenizer, local_cfg, batch)
        else:
            raise NotImplementedError(f"system {cfg.system!r} arrives in a later task")
        if offset == 0:
            hops = batch_hops
        for j, text in enumerate(texts):
            i = offset + j
            ok, score = score_item(cfg.task, text, batch[j])
            predictions[i] = text
            per_item[i] = {"i": i, "correct": bool(ok), "score": float(score)}

    record = build_record(cfg, items, predictions, per_item, hops,
                          {"total_sec": time.time() - started})
    payload = json.dumps(record, indent=2)
    if cfg.out_json:
        with open(cfg.out_json, "w") as fh:
            fh.write(payload)
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("per_item", "hop_records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_stats.py tests/test_run_record.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the whole CPU suite and commit**

Run: `./.venv/bin/python -m pytest -v`
Expected: PASS, all tests from Tasks 1–9.

```bash
git add lkv/stats.py lkv/run.py tests/test_stats.py tests/test_run_record.py
git commit -m "feat: run entry point with per-item records and paired stats"
```

---

## Phase 2 — LatentMAS reproduction (GPU)

The gates here decide whether the LatentMAS side is trustworthy. **Do not proceed to Phase 3 until Task 10's gate passes**, because a broken relay makes every trimming measurement meaningless.

### Task 10: GSM8K on Qwen3-8B — the latent-step sweep

Target: **93.8** for LatentMAS, **81.1** for single-agent. `latent_steps` is not recorded anywhere upstream (the argparse default of 0 is degenerate, and the two example logs show 0 and 10, neither matching their table cells), so it must be searched. The paper says tune over `[0, 80]`.

**Files:**
- Create: `scripts/gate_latentmas.sh`
- Create: `docs/FINDINGS.md`
- Modify: none

**Interfaces:**
- Consumes: `python -m lkv.run` from Task 9.
- Produces: `runs/lm_gsm8k_8b_s{0,10,20,40,80}.json` and `runs/lm_gsm8k_8b_single.json`; a `docs/FINDINGS.md` table of the sweep.

- [ ] **Step 1: Write the gate script**

`scripts/gate_latentmas.sh`:
```bash
#!/usr/bin/env bash
# Phase 2 gate. Reproduction uses the paper's SAMPLED decoding (temperature 0.6,
# top_p 0.95) because that is what the published numbers were produced with.
# Internal A/B comparisons elsewhere use --greedy; never mix the two.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-Qwen/Qwen3-8B}
N=${N:-200}

run() {  # run <tag> <extra args...>
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task gsm8k --model_name "$M" --max_samples "$N" \
      --generate_bs 16 --seed 42 --out_json "$f" "$@" 2>&1 | tail -20
}

for S in 0 10 20 40 80; do
  run "lm_gsm8k_8b_s${S}" --system latentmas --latent_steps "$S"
done
run lm_gsm8k_8b_single --system single --latent_steps 0

$PY - <<'EOF'
import glob, json
rows = []
for path in sorted(glob.glob("runs/lm_gsm8k_8b_*.json")):
    d = json.load(open(path))
    rows.append((path.split("/")[-1], d["system"], d["latent_steps"],
                 d["accuracy"], d["sec_per_item"]))
print(f"{'run':28} {'system':10} {'steps':>5} {'acc':>7} {'s/item':>7}")
for r in rows:
    print(f"{r[0]:28} {r[1]:10} {r[2]:5} {r[3]:7.4f} {r[4]:7.2f}")
best = max(r for r in rows if r[1] == "latentmas")
print(f"\nTARGET latentmas 0.938 | single 0.811")
EOF
```

- [ ] **Step 2: Run the sweep and record it**

Run: `chmod +x scripts/gate_latentmas.sh && N=200 ./scripts/gate_latentmas.sh`
Expected: six JSON files. Write the resulting table into `docs/FINDINGS.md` under a `## Phase 2 — GSM8K 8B latent-step sweep` heading, including the exact command used.

- [ ] **Step 3: Evaluate the gate**

**GATE L1 passes** if some `latent_steps` value reaches accuracy ≥ 0.91 at n=200 (within roughly one standard error of 0.938; SE at n=200 and p≈0.93 is 0.018, so 0.938 − 1.5·SE ≈ 0.911) **and** single-agent lands in `[0.76, 0.86]`.

If L1 fails, work through these in order and record each outcome in `docs/FINDINGS.md`:
1. Re-run the best `latent_steps` with `--realign` — the paper treats it as a per-task hyperparameter and reports +2.3–5.3%.
2. Re-run with `--no_renorm_latent`. Because upstream renormalizes unconditionally (D4), the untested cell is realign-off/renorm-off, which no upstream configuration can produce.
3. Confirm the judger actually sees the cache: run with `--n_agents 1` and check that accuracy differs from the `n_agents 4` run. If they are identical item-for-item, the cache is not reaching the judger — re-read the mask concatenation in `run_latentmas`.
4. Print one full judger prompt and one full completion at `latent_steps=40`. Degenerate output (repeated characters, empty strings) means the latent embeddings are out of distribution — test `--realign` first, then inspect the norm of `projector(hidden)` against the mean input-embedding norm.

- [ ] **Step 4: Confirm no regression in the CPU suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS. Any test that had to be weakened to make the gate pass is a defect, not a fix.

- [ ] **Step 5: Commit**

```bash
git add scripts/gate_latentmas.sh docs/FINDINGS.md runs/lm_gsm8k_8b_*.json
git commit -m "test: GSM8K 8B latent-step sweep (gate L1)"
```

---

### Task 11: MBPP+ on Qwen3-8B, plus 4B and 14B GSM8K

Targets: MBPP+ 8B LatentMAS **74.6**, single **64.8**; GSM8K 4B **88.2**/**82.4**; GSM8K 14B **95.2**/**83.7**. MBPP+ matters most — it is the task where the paper reports the largest LatentMAS gain over TextMAS (+5.1), so it has the most headroom for a relay to demonstrate value. GSM8K 8B single-agent at 81.1 versus LatentMAS at 93.8 is a 12.7-point gap, but if the reproduced single-agent number lands much higher than 81.1 the task is near ceiling and cannot discriminate.

**Files:**
- Modify: `scripts/gate_latentmas.sh` (append stages)
- Modify: `docs/FINDINGS.md`

**Interfaces:**
- Consumes: the winning `latent_steps` from Task 10, referred to below as `$S_BEST`.
- Produces: `runs/lm_mbpp_8b_{latent,single}.json`, `runs/lm_gsm8k_{4b,14b}_{latent,single}.json`.

- [ ] **Step 1: Append the stages**

Append to `scripts/gate_latentmas.sh`:
```bash
# ---- Stage 2: MBPP+ and the other two model sizes ----
# S_BEST comes from the Task 10 sweep; pass it in explicitly so this stage is
# reproducible without re-reading the sweep.
S_BEST=${S_BEST:?set S_BEST to the winning latent_steps from stage 1}

mbpp() {
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task mbppplus --model_name Qwen/Qwen3-8B --max_samples 378 \
      --generate_bs 8 --seed 42 --out_json "$f" "$@" 2>&1 | tail -20
}
mbpp lm_mbpp_8b_latent --system latentmas --latent_steps "$S_BEST"
mbpp lm_mbpp_8b_single --system single    --latent_steps 0

for MODEL in Qwen/Qwen3-4B Qwen/Qwen3-14B; do
  short=$(echo "$MODEL" | sed 's#.*Qwen3-##' | tr 'A-Z' 'a-z')
  for spec in "latent latentmas $S_BEST" "single single 0"; do
    set -- $spec
    f="$OUT/lm_gsm8k_${short}_$1.json"
    [ -f "$f" ] && { echo "skip $f"; continue; }
    $PY -m lkv.run --system "$2" --latent_steps "$3" --task gsm8k \
        --model_name "$MODEL" --max_samples 200 --generate_bs 16 --seed 42 \
        --out_json "$f" 2>&1 | tail -20
  done
done
```

- [ ] **Step 2: Run stage 2**

Run: `S_BEST=<winner> ./scripts/gate_latentmas.sh`
Expected: six more JSON files. MBPP+ at 378 items with `max_new_tokens=4096` is the expensive one; budget roughly 1.5–2 GPU hours.

- [ ] **Step 3: Evaluate the gate and the paired comparison**

**GATE L2 passes** if MBPP+ LatentMAS ≥ 0.70 (74.6 − 1.5·SE, SE ≈ 0.022 at n=378) **and** MBPP+ LatentMAS > MBPP+ single.

The direction matters more than the absolute value. Run the paired test:
```bash
./.venv/bin/python - <<'EOF'
import json
from lkv.stats import mcnemar
a = json.load(open("runs/lm_mbpp_8b_latent.json"))["per_item"]
b = json.load(open("runs/lm_mbpp_8b_single.json"))["per_item"]
assert [x["i"] for x in a] == [x["i"] for x in b], "runs are not on the same slice"
print(mcnemar([x["correct"] for x in a], [x["correct"] for x in b]))
EOF
```
If single-agent beats the relay with `p_exact < 0.05`, the relay is actively harmful and Phase 5 has nothing to trim — stop and diagnose. Record the verdict in `docs/FINDINGS.md`.

- [ ] **Step 4: Record the reproduction table**

Add to `docs/FINDINGS.md` a table with columns `task | model | ours latent | paper latent | ours single | paper single | delta`. State plainly which cells reproduce and which do not. A cell that misses by more than 3 points is a finding to write up, not a number to quietly adjust.

- [ ] **Step 5: Commit**

```bash
git add scripts/gate_latentmas.sh docs/FINDINGS.md runs/lm_mbpp_8b_*.json runs/lm_gsm8k_4b_*.json runs/lm_gsm8k_14b_*.json
git commit -m "test: MBPP+ and 4B/14B GSM8K reproduction (gate L2)"
```

---

## Phase 3 — KVComm correctness (all CPU-testable)

### Task 12: Attention tracer, cache trimming, and the ragged-mask question — D8, D9

Two mechanisms and one open empirical question.

**The tracer** stashes prefill query/key tensors so layer importance can be recomputed later. KVComm subclasses `LlamaAttention` and swaps `block.self_attn` in place while leaving `config._attn_implementation == "sdpa"` (`stock-kvcomm/model_attn.py:17`, `models.py:71-93`). We copy that design exactly, because it is what keeps the stock mask builder in play. It stashes only when `hidden_states.shape[-2] > 1`, i.e. prefill and never decode (`model_attn.py:49-59`).

**The trim** keeps selected layers whole and truncates every other layer to position 0 only, with layer 0 always fully retained (`stock-kvcomm/models.py:95-108`, `if i in self.layers_list or i == 0`).

**The open question (D8).** After trimming, layer 0 is full while others hold one position. `DynamicCache.get_seq_length()` reads layer 0, so the causal mask is built at the full length; sdpa then slices it to each layer's actual key length. If the slice takes the *first* `1+q` columns of a `N+q`-wide causal mask, every one of those columns is visible to every query, and receiver tokens attend to each other bidirectionally in every trimmed layer. **This must be measured, not assumed.** The test below measures it. If confirmed, we fix it — that fix is a real contribution, and it changes what "KVComm reproduction" means, so it must be recorded before Phase 4 runs.

**Files:**
- Create: `lkv/kvcomm/tracer.py`, `lkv/kvcomm/trim.py`
- Test: `tests/test_trim.py`, `tests/test_ragged_cache.py`

**Interfaces:**
- Consumes: `tiny_qwen3`, `tiny_llama` from Task 2.
- Produces:
  - `install_tracer(model) -> list` and `remove_tracer(model, saved) -> None`; while installed, each `model.model.layers[i].self_attn.attn_inputs` is a dict with keys `query`, `key`, `scaling` after a prefill, and `config._attn_implementation` is unchanged.
  - `layer_map(n_layers_a: int, n_layers_b: int) -> list[int]` implementing `round((l + 0.5) * L_B / L_A - 0.5)`.
  - `prepare_key_cache(cache, layers_list: list[int], layer_map: list[int] | None = None, sink: int = 1) -> DynamicCache`.
  - `retention(cache) -> float` — kept positions divided by full-cache positions.
  - `build_per_layer_masks(cache, q_len: int, device, dtype) -> list[Tensor]` — the D8 fix; one correctly-sized additive mask per layer.
  - Task 13 consumes `install_tracer`; Tasks 14 and 17 consume `prepare_key_cache` and `retention`.

- [ ] **Step 1: Write the failing tests**

`tests/test_trim.py`:
```python
import torch

from lkv.kvcomm.tracer import install_tracer, remove_tracer
from lkv.kvcomm.trim import layer_map, prepare_key_cache, retention


def _cache(model, n_pos=12):
    ids = torch.randint(1, 200, (1, n_pos))
    with torch.no_grad():
        return model(ids, use_cache=True).past_key_values


def test_selected_layers_keep_everything_others_keep_one(tiny_qwen3):
    cache = _cache(tiny_qwen3, 12)
    out = prepare_key_cache(cache, layers_list=[2])
    lengths = [int(k.shape[-2]) for k in out.key_cache]
    # layer 0 is always retained in full; layer 2 is selected; 1 and 3 collapse
    assert lengths == [12, 1, 12, 1], lengths


def test_layer_zero_is_retained_even_when_not_selected(tiny_qwen3):
    out = prepare_key_cache(_cache(tiny_qwen3, 9), layers_list=[])
    assert int(out.key_cache[0].shape[-2]) == 9
    assert all(int(k.shape[-2]) == 1 for k in out.key_cache[1:])


def test_full_selection_is_a_faithful_copy(tiny_qwen3):
    cache = _cache(tiny_qwen3, 7)
    out = prepare_key_cache(cache, layers_list=[0, 1, 2, 3])
    for a, b in zip(cache.key_cache, out.key_cache):
        assert torch.equal(a, b)
    assert retention(out) == 1.0


def test_retention_is_positions_kept_over_positions_total(tiny_qwen3):
    out = prepare_key_cache(_cache(tiny_qwen3, 10), layers_list=[1])
    # layers 0 and 1 keep 10 each, layers 2 and 3 keep 1 each -> 22 / 40
    assert abs(retention(out) - 22 / 40) < 1e-9


def test_sink_size_is_configurable(tiny_qwen3):
    out = prepare_key_cache(_cache(tiny_qwen3, 10), layers_list=[], sink=3)
    assert [int(k.shape[-2]) for k in out.key_cache] == [10, 3, 3, 3]


def test_layer_map_is_the_upstream_formula():
    assert layer_map(4, 4) == [0, 1, 2, 3]
    # L_A=4 -> L_B=2: round((l+0.5)*2/4 - 0.5)
    assert layer_map(4, 2) == [0, 0, 1, 1]
    assert layer_map(2, 4) == [0, 2]


def test_tracer_preserves_the_attention_implementation(tiny_llama):
    before = tiny_llama.config._attn_implementation
    saved = install_tracer(tiny_llama)
    try:
        assert tiny_llama.config._attn_implementation == before == "sdpa"
        ids = torch.randint(1, 200, (1, 6))
        with torch.no_grad():
            tiny_llama(ids, use_cache=True)
        for layer in tiny_llama.model.layers:
            stash = layer.self_attn.attn_inputs
            assert set(("query", "key", "scaling")) <= set(stash)
            assert stash["query"].shape[-2] == 6
    finally:
        remove_tracer(tiny_llama, saved)
    assert not hasattr(tiny_llama.model.layers[0].self_attn, "attn_inputs")


def test_tracer_does_not_stash_on_single_token_decode(tiny_llama):
    saved = install_tracer(tiny_llama)
    try:
        with torch.no_grad():
            out = tiny_llama(torch.randint(1, 200, (1, 6)), use_cache=True)
            first = tiny_llama.model.layers[0].self_attn.attn_inputs["query"].shape[-2]
            tiny_llama(torch.randint(1, 200, (1, 1)), use_cache=True,
                       past_key_values=out.past_key_values)
            after = tiny_llama.model.layers[0].self_attn.attn_inputs["query"].shape[-2]
        assert first == after == 6, "decode step overwrote the prefill stash"
    finally:
        remove_tracer(tiny_llama, saved)
```

`tests/test_ragged_cache.py`:
```python
"""D8/D9: what does stock sdpa actually do with a ragged DynamicCache?

This test MEASURES the behaviour rather than asserting a belief about it. The
first test records the answer; the second asserts our own fix is causal.
"""
import torch

from lkv.kvcomm.trim import build_per_layer_masks, prepare_key_cache


def _prefill(model, n_pos):
    ids = torch.randint(1, 200, (1, n_pos))
    with torch.no_grad():
        return model(ids, use_cache=True).past_key_values


def test_ragged_cache_forward_does_not_crash(tiny_qwen3):
    """Establishes that the ragged path runs at all -- KVComm relies on it."""
    cache = prepare_key_cache(_prefill(tiny_qwen3, 12), layers_list=[2])
    q = torch.randint(1, 200, (1, 4))
    mask = torch.ones(1, 12 + 4, dtype=torch.long)
    with torch.no_grad():
        out = tiny_qwen3(q, attention_mask=mask, past_key_values=cache, use_cache=True)
    assert out.logits.shape == (1, 4, tiny_qwen3.config.vocab_size)


def _causality_drift(model, cache_builder):
    """Run 4 receiver tokens twice, changing ONLY the last one. If the FIRST
    position's logits move, information flowed backwards."""
    outs = []
    for q in (torch.tensor([[11, 12, 13, 14]]), torch.tensor([[11, 12, 13, 99]])):
        cache = cache_builder()
        lens = [int(k.shape[-2]) for k in cache.key_cache]
        mask = torch.ones(1, max(lens) + 4, dtype=torch.long)
        with torch.no_grad():
            outs.append(model(q, attention_mask=mask, past_key_values=cache,
                              use_cache=True).logits)
    return (outs[0][0, 0] - outs[1][0, 0]).abs().max().item()


def test_untrimmed_cache_is_causal(tiny_qwen3):
    """Control. Establishes the measurement itself reads zero when it should."""
    base = _prefill(tiny_qwen3, 12)
    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    drift = _causality_drift(tiny_qwen3, lambda: prepare_key_cache(base, all_layers))
    assert drift == 0.0, f"control drifted: {drift:.3e}"


def test_stock_path_is_NOT_causal_when_trimmed(tiny_qwen3):
    """D8, already confirmed by measurement before implementation began:
    cache lengths [12, 1, 12, 1] gave a first-position drift of 1.940e-01.

    This test pins the DEFECT so that a future transformers upgrade silently
    fixing it is visible rather than invisible. If it starts failing, stock
    sdpa began handling ragged caches causally -- update docs/FINDINGS.md and
    re-evaluate whether forward_with_ragged_cache is still needed.
    """
    base = _prefill(tiny_qwen3, 12)
    drift = _causality_drift(tiny_qwen3, lambda: prepare_key_cache(base, [2]))
    assert drift > 1e-3, (
        f"stock ragged path now appears causal (drift={drift:.3e}); D8 may be "
        "fixed upstream -- see docs/FINDINGS.md"
    )


def test_per_layer_masks_are_causal_and_correctly_sized(tiny_qwen3):
    cache = prepare_key_cache(_prefill(tiny_qwen3, 12), layers_list=[2])
    masks = build_per_layer_masks(cache, q_len=4, device="cpu", dtype=torch.float32)
    assert len(masks) == 4
    for i, m in enumerate(masks):
        kv_len = int(cache.key_cache[i].shape[-2]) + 4
        assert m.shape[-1] == kv_len, (i, m.shape, kv_len)
        assert m.shape[-2] == 4
        # the receiver block is the last 4 columns; it must be lower-triangular
        block = m[0, 0, :, -4:]
        for r in range(4):
            for c in range(r + 1, 4):
                assert torch.isinf(block[r, c]) and block[r, c] < 0, (i, r, c)
            assert block[r, r] == 0.0


def test_fixed_forward_is_causal(tiny_qwen3):
    """With per-layer masks, changing the LAST receiver token must not move the
    FIRST receiver position's logits at all."""
    from lkv.kvcomm.trim import forward_with_ragged_cache

    base = _prefill(tiny_qwen3, 12)
    outs = []
    for q in (torch.tensor([[11, 12, 13, 14]]), torch.tensor([[11, 12, 13, 99]])):
        cache = prepare_key_cache(base, layers_list=[2])
        with torch.no_grad():
            outs.append(forward_with_ragged_cache(tiny_qwen3, q, cache))
    drift = (outs[0][0, 0] - outs[1][0, 0]).abs().max().item()
    assert drift == 0.0, f"future token leaked into position 0: drift={drift:.3e}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_trim.py tests/test_ragged_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.kvcomm.tracer'`.

- [ ] **Step 3: Write the implementation**

`lkv/kvcomm/tracer.py`:
```python
"""Stash prefill query/key so layer importance can be recomputed.

Follows stock-kvcomm/model_attn.py exactly in the one respect that matters:
the module is subclassed and swapped in place, and config._attn_implementation
is left as "sdpa". Changing that string would send transformers down the
custom-backend path in masking_utils, which returns attention_mask=None and
makes attention bidirectional.
"""
from __future__ import annotations

import torch


def _patch(attn_module):
    original_forward = attn_module.forward

    def traced_forward(hidden_states, position_embeddings, attention_mask=None,
                       past_key_value=None, cache_position=None, **kwargs):
        # Prefill only (upstream model_attn.py:49-59). A decode step must not
        # clobber the stash -- importance is defined over the prefill attention.
        if hidden_states.shape[-2] > 1:
            attn_module.attn_inputs = {
                "hidden_states": hidden_states.detach(),
                "position_embeddings": tuple(p.detach() for p in position_embeddings),
                "scaling": attn_module.scaling,
            }
        out = original_forward(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            past_key_value=past_key_value, cache_position=cache_position, **kwargs
        )
        if hidden_states.shape[-2] > 1 and past_key_value is not None:
            idx = attn_module.layer_idx
            attn_module.attn_inputs["query"] = _project_q(attn_module, hidden_states)
            attn_module.attn_inputs["key"] = past_key_value.key_cache[idx].detach()
        return out

    return original_forward, traced_forward


def _project_q(attn_module, hidden_states):
    q = attn_module.q_proj(hidden_states)
    b, t, _ = hidden_states.shape
    n_heads = attn_module.config.num_attention_heads
    head_dim = getattr(attn_module, "head_dim", q.shape[-1] // n_heads)
    q = q.view(b, t, n_heads, head_dim).transpose(1, 2)
    if hasattr(attn_module, "q_norm"):
        q = attn_module.q_norm(q)
    return q.detach()


def install_tracer(model):
    """Returns a list of (module, original_forward) for remove_tracer."""
    saved = []
    for layer in model.model.layers:
        attn = layer.self_attn
        original, traced = _patch(attn)
        saved.append((attn, original))
        attn.forward = traced
        attn.attn_inputs = {}
    return saved


def remove_tracer(model, saved) -> None:
    for attn, original in saved:
        attn.forward = original
        if hasattr(attn, "attn_inputs"):
            del attn.attn_inputs
```

`lkv/kvcomm/trim.py`:
```python
"""Layer-selective KV trimming, plus a correct mask for the resulting cache.

prepare_key_cache mirrors stock-kvcomm/models.py:95-108: selected layers are
copied whole, every other layer keeps only `sink` leading positions, and layer 0
is always retained in full (`if i in self.layers_list or i == 0`).

build_per_layer_masks / forward_with_ragged_cache are OURS. Upstream's default
path hands the ragged cache to the stock forward, where one full-length causal
mask is sliced down to each layer's key length -- see tests/test_ragged_cache.py
for the measurement. Their opt-in `shift_back` path (models.py:253-359) exists
for the same reason but asserts at most two distinct cache lengths.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache


def layer_map(n_layers_a: int, n_layers_b: int) -> list[int]:
    return [round((i + 0.5) * n_layers_b / n_layers_a - 0.5) for i in range(n_layers_a)]


def prepare_key_cache(cache, layers_list, layer_map=None, sink: int = 1):
    keep = set(int(i) for i in layers_list)
    n = len(cache.key_cache)
    mapping = layer_map if layer_map is not None else list(range(n))
    out = DynamicCache()
    for i in range(n):
        k, v = cache.key_cache[i], cache.value_cache[i]
        if i in keep or i == 0:
            out.update(k, v, mapping[i])
        else:
            out.update(k[:, :, :sink, :], v[:, :, :sink, :], mapping[i])
    return out


def retention(cache) -> float:
    lengths = [int(k.shape[-2]) for k in cache.key_cache]
    full = max(lengths) * len(lengths)
    return (sum(lengths) / full) if full else 1.0


def build_per_layer_masks(cache, q_len: int, device, dtype):
    """One additive causal mask per layer, each sized to that layer's kv length.

    A trimmed layer's `sink` cached positions are all visible (they precede every
    receiver token); the receiver block is strictly lower-triangular.
    """
    neg = torch.finfo(dtype).min
    masks = []
    for k in cache.key_cache:
        past = int(k.shape[-2])
        m = torch.zeros(1, 1, q_len, past + q_len, device=device, dtype=dtype)
        future = torch.triu(
            torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1
        )
        m[0, 0, :, past:] = torch.where(
            future, torch.tensor(neg, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
        )
        masks.append(m)
    return masks


def forward_with_ragged_cache(model, input_ids, cache, sink: int = 1):
    """Run the decoder with a per-layer mask so causality holds in every layer.

    Positions: a trimmed layer places receiver tokens right after its sink, a
    retained layer places them after the full context. Upstream's shift_back
    does the same thing but only for two distinct lengths (models.py:240-249).
    """
    q_len = int(input_ids.shape[1])
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    masks = build_per_layer_masks(cache, q_len, device, dtype)

    hidden = model.model.embed_tokens(input_ids)
    for i, layer in enumerate(model.model.layers):
        past = int(cache.key_cache[i].shape[-2])
        cache_position = torch.arange(past, past + q_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        rotary = model.model.rotary_emb(hidden, position_ids)
        hidden = layer(
            hidden,
            attention_mask=masks[i],
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=rotary,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)
```

- [ ] **Step 4: Run the tests and record the D8 measurement**

Run: `./.venv/bin/python -m pytest tests/test_trim.py tests/test_ragged_cache.py -v -s`
Expected: PASS, 12 tests. The `-s` flag is required so the `D8 MEASUREMENT` line is printed.

Copy the printed drift value into `docs/FINDINGS.md` under `## D8 — ragged-cache causality`. Interpretation: a drift of exactly `0.000e+00` means stock sdpa handles the ragged cache causally and D8 is **not** a defect — say so plainly and note that `forward_with_ragged_cache` is then a redundant safety net, not a fix. Any non-zero drift confirms D8, and Phase 4 must then run through `forward_with_ragged_cache`, with the deviation from upstream recorded.

If `forward_with_ragged_cache` will not run against this transformers version, do not delete the test. The decoder-layer call signature is version-specific; read `transformers/models/qwen3/modeling_qwen3.py` at 4.53.3 and match it exactly.

- [ ] **Step 5: Commit**

```bash
git add lkv/kvcomm/tracer.py lkv/kvcomm/trim.py tests/test_trim.py tests/test_ragged_cache.py docs/FINDINGS.md
git commit -m "feat: KVComm tracer and layer trimming with per-layer causal masks (D8, D9)"
```

---

### Task 13: Layer importance and ranking — bit-exact against upstream

The ranking is the whole method, so it is tested against a **verbatim copy of upstream's functions** embedded in the test file. If our version and theirs disagree on random input, ours is wrong.

Semantics to preserve exactly (`stock-kvcomm/layer_importance.py`):
- Importance is receiver-side post-softmax attention probability mass on sender-context keys, sliced `attn[..., 1:-q_len]` — all context keys except position 0, excluding the receiver's own positions (`:21`).
- Aggregate `.sum(dim=-1).mean()` over batch × **query heads** (GQA repeated) × query positions; NaN maps to 0 (`:22-23`).
- Min-max normalize importance and the Gaussian separately, then `alpha * importance + (1 - alpha) * gaussian` — a convex sum, not a product (`:41-45`).
- `mu_abs = mu * (n_layers - 1)`; the Gaussian is unnormalized `exp(-0.5 * ((x - mu_abs) / sigma) ** 2)` (`:12-15`, `:42`).
- Retained count is `int(top_layers * n_layers)` — a floor, so `0.3 × 32 = 9` (`:29`).
- At the Llama default `alpha = 1.0` the Gaussian contributes nothing. This is not a bug; it is opt-in via `alpha < 1`, and paper §B.2 sets `alpha = 0.8` for Qwen and Falcon.

**Files:**
- Create: `lkv/kvcomm/importance.py`
- Test: `tests/test_importance.py`

**Interfaces:**
- Consumes: `install_tracer` (Task 12), `tiny_llama` (Task 2).
- Produces:
  - `attention_from_stash(attn_module, stash, q_len_receiver) -> Tensor` of shape `(batch, heads, q_len, kv_len)`.
  - `layer_importance(attn_weights_by_layer: dict[int, Tensor], q_len: int) -> dict[int, float]`.
  - `layer_ranking(importance_totals: dict[int, list[float]], alpha: float, mu: float, sigma: float) -> list[int]` descending.
  - `select_layers(importance_totals, top_layers: float, alpha, mu, sigma, random_selection=False, seed=42) -> list[int]`.
  - Task 14 and Task 18 consume `select_layers`.

- [ ] **Step 1: Write the failing test**

`tests/test_importance.py`:
```python
"""Bit-exactness against upstream. The reference block below is copied verbatim
from stock-kvcomm/layer_importance.py -- do not tidy it."""
import numpy as np
import torch

from lkv.kvcomm.importance import layer_importance, layer_ranking, select_layers


# ---- verbatim upstream reference ----
def _ref_normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32)
    mn, mx = v.min(), v.max()
    if mx - mn < eps:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn + eps)


def _ref_gaussian_prior(L, mu, sigma=10.0):
    x = np.arange(L, dtype=np.float32)
    return np.exp(-0.5 * ((x - mu) / float(sigma)) ** 2)


def _ref_calc(B_attn_weights: dict):
    out = {}
    for i in B_attn_weights:
        aw = B_attn_weights[i]
        attn_matrix = aw[..., 1:-aw.shape[-2]]
        importance = attn_matrix.sum(dim=-1).mean()
        out[i] = 0 if torch.isnan(importance) else importance.item()
    return out


def _ref_ranking(totals: dict, alpha, mu_frac, sigma):
    importance = []
    n = len(totals)
    for i in range(n):
        importance.append(np.mean(totals[i]))
    importance = _ref_normalize(importance)
    mu = mu_frac * (n - 1)
    g = _ref_normalize(_ref_gaussian_prior(n, mu=mu, sigma=sigma))
    importance = alpha * importance + (1.0 - alpha) * g
    return np.argsort(importance)[::-1]
# ---- end reference ----


def _fake_attn(n_layers=6, batch=2, heads=4, q_len=3, ctx=10, seed=0):
    torch.manual_seed(seed)
    out = {}
    for i in range(n_layers):
        logits = torch.randn(batch, heads, q_len, ctx + q_len)
        out[i] = torch.softmax(logits, dim=-1)
    return out


def test_layer_importance_matches_upstream_exactly():
    attn = _fake_attn()
    ours = layer_importance(attn, q_len=3)
    theirs = _ref_calc(attn)
    for i in theirs:
        assert abs(ours[i] - theirs[i]) < 1e-12, (i, ours[i], theirs[i])


def test_importance_excludes_position_zero_and_receiver_block():
    """Sliced range is 1 .. ctx-1, so ctx-1 columns out of ctx+q."""
    attn = _fake_attn(n_layers=1, batch=1, heads=1, q_len=3, ctx=10)
    manual = attn[0][..., 1:-3].sum(dim=-1).mean().item()
    assert abs(layer_importance(attn, q_len=3)[0] - manual) < 1e-12


def test_nan_importance_becomes_zero():
    attn = {0: torch.full((1, 1, 2, 6), float("nan"))}
    assert layer_importance(attn, q_len=2)[0] == 0


def test_ranking_matches_upstream_for_several_alphas():
    totals = {i: [float(v)] for i, v in enumerate([0.1, 0.9, 0.4, 0.7, 0.2, 0.55])}
    for alpha in (1.0, 0.8, 0.5, 0.0):
        ours = layer_ranking(totals, alpha=alpha, mu=0.5, sigma=10.0)
        theirs = list(_ref_ranking(totals, alpha, 0.5, 10.0))
        assert ours == theirs, (alpha, ours, theirs)


def test_alpha_one_ignores_the_gaussian():
    totals = {i: [float(v)] for i, v in enumerate([0.1, 0.9, 0.4, 0.7])}
    a = layer_ranking(totals, alpha=1.0, mu=0.5, sigma=10.0)
    b = layer_ranking(totals, alpha=1.0, mu=0.1, sigma=1.0)
    assert a == b, "mu/sigma changed the ranking at alpha=1.0"


def test_alpha_zero_is_pure_gaussian_peaking_mid_depth():
    totals = {i: [0.0] for i in range(11)}   # flat importance
    r = layer_ranking(totals, alpha=0.0, mu=0.5, sigma=2.0)
    assert r[0] == 5, r


def test_select_layers_count_is_floored():
    totals = {i: [float(i)] for i in range(32)}
    assert len(select_layers(totals, 0.3, 1.0, 0.5, 10.0)) == 9    # int(9.6)
    assert len(select_layers(totals, 0.5, 1.0, 0.5, 10.0)) == 16
    assert len(select_layers(totals, 0.7, 1.0, 0.5, 10.0)) == 22   # int(22.4)


def test_select_layers_takes_the_highest_scoring():
    totals = {i: [float(i)] for i in range(10)}
    assert sorted(select_layers(totals, 0.3, 1.0, 0.5, 10.0)) == [7, 8, 9]


def test_random_selection_is_seeded_and_same_size():
    totals = {i: [float(i)] for i in range(32)}
    a = select_layers(totals, 0.5, 1.0, 0.5, 10.0, random_selection=True, seed=7)
    b = select_layers(totals, 0.5, 1.0, 0.5, 10.0, random_selection=True, seed=7)
    c = select_layers(totals, 0.5, 1.0, 0.5, 10.0, random_selection=True, seed=8)
    assert a == b and len(a) == 16 and a != c


def test_end_to_end_ranking_from_a_traced_forward(tiny_llama):
    """The tracer must produce stashes that importance can consume."""
    from lkv.kvcomm.importance import attention_from_stash
    from lkv.kvcomm.tracer import install_tracer, remove_tracer

    saved = install_tracer(tiny_llama)
    try:
        with torch.no_grad():
            tiny_llama(torch.randint(1, 200, (1, 8)), use_cache=True)
        attn = {}
        for i, layer in enumerate(tiny_llama.model.layers):
            attn[i] = attention_from_stash(layer.self_attn,
                                           layer.self_attn.attn_inputs, q_len_receiver=8)
        assert set(attn) == {0, 1, 2, 3}
        for w in attn.values():
            assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4)
    finally:
        remove_tracer(tiny_llama, saved)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_importance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.kvcomm.importance'`.

- [ ] **Step 3: Write the implementation**

`lkv/kvcomm/importance.py`:
```python
"""Layer importance and ranking, semantically identical to
stock-kvcomm/layer_importance.py. tests/test_importance.py holds a verbatim copy
of the upstream functions and asserts agreement on random input."""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from transformers.models.llama.modeling_llama import repeat_kv


def normalize(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo + 1e-9)


def gaussian_prior(n_layers: int, mu_abs: float, sigma: float) -> np.ndarray:
    x = np.arange(n_layers, dtype=np.float32)
    return np.exp(-0.5 * ((x - mu_abs) / float(sigma)) ** 2)


def attention_from_stash(attn_module, stash: dict, q_len_receiver: int) -> torch.Tensor:
    """Recompute post-softmax attention from the stashed query and cached keys.

    Mirrors stock-kvcomm/models.py:167-186: softmax(Q K^T * scaling + causal),
    with GQA key heads repeated so every QUERY head is represented.
    """
    q = stash["query"]
    k = stash["key"]
    n_rep = q.shape[1] // k.shape[1]
    k = repeat_kv(k, n_rep)
    scores = torch.matmul(q, k.transpose(2, 3)) * stash["scaling"]

    kv_len, q_len = scores.shape[-1], scores.shape[-2]
    offset = kv_len - q_len
    causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).tril(offset)
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1, dtype=torch.float32)


def layer_importance(attn_by_layer: dict, q_len: int | None = None) -> dict:
    """Mass each receiver query puts on context keys 1..len(context)-1.

    The slice is upstream's `attn[..., 1:-q_len]` (layer_importance.py:21): drop
    key position 0, and drop the receiver's own q_len positions at the tail.
    q_len is taken from the tensor itself so it cannot disagree with the data;
    the parameter is accepted only to cross-check the caller.
    """
    out = {}
    for i, weights in attn_by_layer.items():
        tail = weights.shape[-2]
        if q_len is not None and q_len != tail:
            raise ValueError(
                f"layer {i}: caller says q_len={q_len} but tensor has {tail} queries"
            )
        value = weights[..., 1:-tail].sum(dim=-1).mean()
        out[i] = 0 if torch.isnan(value) else value.item()
    return out


def layer_ranking(importance_totals: dict, alpha: float, mu: float,
                  sigma: float) -> list[int]:
    n = len(importance_totals)
    if n != max(importance_totals) + 1:
        raise ValueError("importance_totals must be keyed 0..n-1 with no gaps")
    means = [float(np.mean(importance_totals[i])) for i in range(n)]
    score = alpha * normalize(means) + (1.0 - alpha) * normalize(
        gaussian_prior(n, mu * (n - 1), sigma)
    )
    return [int(i) for i in np.argsort(score)[::-1]]


def select_layers(importance_totals: dict, top_layers: float, alpha: float,
                  mu: float, sigma: float, random_selection: bool = False,
                  seed: int = 42) -> list[int]:
    n = len(importance_totals)
    count = int(top_layers * n)          # floor, as upstream :29
    if random_selection:
        return sorted(random.Random(seed).sample(range(n), count))
    return layer_ranking(importance_totals, alpha, mu, sigma)[:count]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_importance.py -v`
Expected: PASS, 10 tests. If `test_layer_importance_matches_upstream_exactly` fails, fix `lkv/kvcomm/importance.py` — never the reference block.

- [ ] **Step 5: Commit**

```bash
git add lkv/kvcomm/importance.py tests/test_importance.py
git commit -m "feat: KVComm layer importance and ranking, bit-exact vs upstream"
```

---

### Task 14: The KVComm single-hop pair harness

One sender prefills `Instruction + Context`; the receiver gets `Instruction + Question` only, plus the trimmed cache. Sender and receiver are the same checkpoint by default. Exactly one hop.

**Files:**
- Create: `lkv/kvcomm/pair.py`
- Modify: `lkv/run.py` (dispatch `--system kvcomm`, `--system skyline`, `--system baseline`)
- Test: `tests/test_pair.py`

**Interfaces:**
- Consumes: `prepare_key_cache`/`retention`/`forward_with_ragged_cache` (Task 12), `select_layers`/`attention_from_stash`/`layer_importance` (Task 13), prompts (Task 6), `generate_text_batch` (Task 3).
- Produces:
  - `calibrate(model, tokenizer, items, cfg) -> dict[int, list[float]]` — runs `cfg.calib_size` items with **all layers transmitted** and returns importance totals (upstream calibrates at full retention, `stock-kvcomm/models.py:58-59`).
  - `run_pair(model_a, model_b, tokenizer, cfg, items, layers_list) -> list[str]`.
  - `run_skyline(model, tokenizer, cfg, items) -> list[str]`, `run_baseline(model, tokenizer, cfg, items) -> list[str]`.
  - Task 15 invokes these through `python -m lkv.run`.

- [ ] **Step 1: Write the failing test**

`tests/test_pair.py`:
```python
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


def test_full_retention_pair_differs_from_baseline(tiny_llama):
    """Sanity: transmitting all layers must change the receiver's logits versus
    transmitting nothing. If not, the cache is not reaching the receiver."""
    from lkv.kvcomm.pair import receiver_logits
    all_layers = list(range(tiny_llama.config.num_hidden_layers))
    with_cache = receiver_logits(tiny_llama, tiny_llama, _Tok(), _cfg(),
                                 _ITEMS[:1], layers_list=all_layers)
    no_cache = receiver_logits(tiny_llama, tiny_llama, _Tok(), _cfg(),
                               _ITEMS[:1], layers_list=None)
    assert not torch.allclose(with_cache, no_cache, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_pair.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.kvcomm.pair'`.

- [ ] **Step 3: Write the implementation**

`lkv/kvcomm/pair.py`:
```python
"""KVComm: one sender prefill, one receiver generation, one hop.

The receiver never sees the context as text -- the cache is its only access,
which is what makes the Baseline (query only) the correct floor and the Skyline
(full text to one model) the correct ceiling.

Calibration runs at FULL retention (stock-kvcomm/models.py:58-59): importance is
defined over the attention a receiver pays to a complete context.
"""
from __future__ import annotations

import torch

from lkv.decode import generate_text_batch
from lkv.kvcomm.importance import attention_from_stash, layer_importance
from lkv.kvcomm.tracer import install_tracer, remove_tracer
from lkv.kvcomm.trim import prepare_key_cache
from lkv.modeling import pad_batch
from lkv.prompts import (
    kvcomm_receiver_prompt,
    kvcomm_sender_prompt,
    kvcomm_skyline_prompt,
)


def _sender_cache(model_a, tokenizer, items):
    texts = [kvcomm_sender_prompt(it["instruction"], it["context"]) for it in items]
    ids, mask = pad_batch(tokenizer, texts)
    ids, mask = ids.to(model_a.device), mask.to(model_a.device)
    with torch.no_grad():
        out = model_a(input_ids=ids, attention_mask=mask, use_cache=True)
    return out.past_key_values, int(ids.shape[1])


def _receiver_inputs(tokenizer, model_b, items):
    texts = [kvcomm_receiver_prompt(it["instruction"], it["question"]) for it in items]
    ids, mask = pad_batch(tokenizer, texts)
    return ids.to(model_b.device), mask.to(model_b.device)


def calibrate(model, tokenizer, items, cfg) -> dict:
    """Importance totals over cfg.calib_size items, at full retention."""
    totals = {i: [] for i in range(model.config.num_hidden_layers)}
    saved = install_tracer(model)
    try:
        for item in items[: cfg.calib_size]:
            cache, _ = _sender_cache(model, tokenizer, [item])
            ids, mask = _receiver_inputs(tokenizer, model, [item])
            full_mask = torch.cat(
                [torch.ones(1, cache.key_cache[0].shape[-2], dtype=mask.dtype,
                            device=mask.device), mask], dim=1)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=full_mask,
                      past_key_values=cache, use_cache=True)
            q_len = int(ids.shape[1])
            attn = {
                i: attention_from_stash(layer.self_attn, layer.self_attn.attn_inputs,
                                        q_len_receiver=q_len)
                for i, layer in enumerate(model.model.layers)
            }
            for i, value in layer_importance(attn, q_len=q_len).items():
                totals[i].append(float(value))
    finally:
        remove_tracer(model, saved)
    return totals


def receiver_logits(model_a, model_b, tokenizer, cfg, items, layers_list):
    """One forward pass of the receiver. layers_list=None means no cache."""
    ids, mask = _receiver_inputs(tokenizer, model_b, items)
    if layers_list is None:
        with torch.no_grad():
            return model_b(input_ids=ids, attention_mask=mask, use_cache=False).logits
    cache, ctx_len = _sender_cache(model_a, tokenizer, items)
    trimmed = prepare_key_cache(cache, layers_list)
    pad = torch.ones(ids.shape[0], ctx_len, dtype=mask.dtype, device=mask.device)
    with torch.no_grad():
        return model_b(input_ids=ids, attention_mask=torch.cat([pad, mask], dim=1),
                       past_key_values=trimmed, use_cache=True).logits


def run_pair(model_a, model_b, tokenizer, cfg, items, layers_list) -> list[str]:
    cache, ctx_len = _sender_cache(model_a, tokenizer, items)
    trimmed = prepare_key_cache(cache, layers_list)
    ids, mask = _receiver_inputs(tokenizer, model_b, items)
    pad = torch.ones(ids.shape[0], ctx_len, dtype=mask.dtype, device=mask.device)
    texts, _ = generate_text_batch(
        model_b, tokenizer, ids, torch.cat([pad, mask], dim=1),
        max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy,
        temperature=cfg.temperature, top_p=cfg.top_p, past_key_values=trimmed,
    )
    return texts


def _plain(model, tokenizer, cfg, texts) -> list[str]:
    ids, mask = pad_batch(tokenizer, texts)
    out, _ = generate_text_batch(
        model, tokenizer, ids.to(model.device), mask.to(model.device),
        max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy,
        temperature=cfg.temperature, top_p=cfg.top_p,
    )
    return out


def run_skyline(model, tokenizer, cfg, items) -> list[str]:
    return _plain(model, tokenizer, cfg, [
        kvcomm_skyline_prompt(it["instruction"], it["context"], it["question"])
        for it in items
    ])


def run_baseline(model, tokenizer, cfg, items) -> list[str]:
    return _plain(model, tokenizer, cfg, [
        kvcomm_receiver_prompt(it["instruction"], it["question"]) for it in items
    ])
```

Then extend `main()` in `lkv/run.py` so `--system` accepts `kvcomm`, `skyline`, and `baseline`: add the three choices to `parse_args`, load `model_b` when `cfg.model_b != cfg.model_name`, call `calibrate` once before the batch loop for `kvcomm`, pass `select_layers(...)` into `run_pair`, and add `layer_retention` plus `kept_layers` to the record.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_pair.py -v && ./.venv/bin/python -m pytest -q`
Expected: PASS, 6 new tests and the whole suite green.

- [ ] **Step 5: Commit**

```bash
git add lkv/kvcomm/pair.py lkv/run.py lkv/config.py tests/test_pair.py
git commit -m "feat: KVComm single-hop pair harness with calibration"
```

---

## Phase 4 — KVComm reproduction (GPU)

### Task 15: Countries gate and the metric decision — resolves D10

Countries first: 200 items, `max_new_tokens=5`, and the widest dynamic range in the paper (Baseline 0.00, Skyline 0.62, KVComm(0.3) 0.51). It is both the cheapest run and the most decisive, because a Baseline of 0.00 means any non-zero KVComm score is unambiguously information arriving through the cache.

**Files:**
- Create: `scripts/gate_kvcomm.sh`
- Modify: `docs/FINDINGS.md`

**Interfaces:**
- Consumes: `python -m lkv.run --system {kvcomm,skyline,baseline}` from Task 14.
- Produces: `runs/kv_countries_{baseline,skyline,r30,r50,r70,rand50}.json`.

- [ ] **Step 1: Write the gate script**

`scripts/gate_kvcomm.sh`:
```bash
#!/usr/bin/env bash
# Phase 4 gate. KVComm decodes greedily (stock-kvcomm/eval.py:80-88 sets
# do_sample=False), so --greedy here is reproduction, not an internal control.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-meta-llama/Llama-3.1-8B-Instruct}
TASK=${TASK:-countries}

run() {
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task "$TASK" --model_name "$M" --greedy --seed 42 \
      --calib_size 1 --mu 0.5 --sigma 10.0 --out_json "$f" "$@" 2>&1 | tail -20
}

run "kv_${TASK}_baseline" --system baseline
run "kv_${TASK}_skyline"  --system skyline
for R in 0.3 0.5 0.7; do
  run "kv_${TASK}_r${R/0./}0" --system kvcomm --top_layers "$R"
done
run "kv_${TASK}_rand50" --system kvcomm --top_layers 0.5 --random_selection

$PY - <<'EOF'
import glob, json, os
print(f"{'run':30} {'acc(F1>0.5)':>12} {'meanF1':>8} {'layer_ret':>10}")
for path in sorted(glob.glob("runs/kv_*.json")):
    d = json.load(open(path))
    print(f"{os.path.basename(path):30} {d['accuracy']:12.4f} "
          f"{d['mean_score']:8.4f} {d.get('layer_retention', 1.0):10.4f}")
EOF
```

- [ ] **Step 2: Run the Countries gate**

Run: `chmod +x scripts/gate_kvcomm.sh && TASK=countries ./scripts/gate_kvcomm.sh`
Expected: six JSON files, each carrying both `accuracy` (fraction with F1 > 0.5) and `mean_score` (mean F1).

- [ ] **Step 3: Decide the metric, then evaluate the gate**

Compare **both** columns against the published Countries column (Baseline 0.00, Skyline 0.62, 0.3 → 0.51, 0.5 → 0.62, 0.7 → 0.62). Whichever column tracks the published row across all five conditions is the metric KVComm reported. Write the decision and the two columns into `docs/FINDINGS.md` under `## D10 — which metric reproduces`, and state it once and for all; every later table uses that metric and names it explicitly.

**GATE K1 passes** if, under the chosen metric, all four hold:
1. `baseline ≤ 0.05` — the receiver genuinely cannot answer without the context.
2. `skyline ∈ [0.55, 0.69]` — within roughly 1.5 SE of 0.62 at n=200 (SE ≈ 0.034).
3. `r30 ≥ 0.42` and `r50 ≥ 0.55` — within 1.5·SE below 0.51 and 0.62.
4. Monotone non-decreasing across `r30 ≤ r50 ≤ r70`, allowing a 0.02 tolerance for the 0.5/0.7 tie the paper itself reports.

If K1 fails, diagnose in this order, recording each result:
1. Check `layer_retention` in the r30 record equals `int(0.3 × n_layers) / n_layers` — for Llama-3.1-8B that is `9/32 = 0.28125`. A value of 1.0 means the selection never reached `prepare_key_cache`.
2. Compare against `rand50`. If random selection matches or beats KVComm(0.5), the ranking is not working; re-check `calibrate` is tracing the *receiver's* forward, not the sender's. Note the paper itself reports random beating KVComm on 5 of 8 datasets at ratio 0.7 for this pair, so a tie at 0.7 is expected and is not evidence of a bug.
3. Re-run r30 through `forward_with_ragged_cache` (Task 12). If the score jumps, D8 was real and materially affects results — this is a finding to write up prominently.
4. Verify the receiver prompt contains no context substring, by printing one rendered prompt in full.

- [ ] **Step 4: Confirm the CPU suite still passes**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/gate_kvcomm.sh docs/FINDINGS.md runs/kv_countries_*.json
git commit -m "test: KVComm Countries reproduction and metric decision (gate K1, D10)"
```

---

### Task 16: HotpotQA and Tipsheets — the harder reproduction

HotpotQA is the discriminating benchmark: published 0.19 / 0.74 / 0.33 / 0.60 / 0.69 spans the widest range of any dataset and its three retention points are well separated, so a trimming curve measured here has real resolution. Tipsheets is included because it is the one dataset where KVComm reportedly *exceeds* Skyline (0.93 versus 0.92), which is a useful sanity anchor.

**Files:**
- Modify: `docs/FINDINGS.md`
- Create: `lkv/data/tipsheets.py`, and register it in `lkv/data/__init__.py`

**Interfaces:**
- Consumes: `scripts/gate_kvcomm.sh` from Task 15, parameterized by `TASK`.
- Produces: `runs/kv_hotpotqa_*.json`, `runs/kv_tipsheets_*.json`.

- [ ] **Step 1: Add the tipsheets loader**

`lkv/data/tipsheets.py`:
```python
"""KVComm bundles tipsheets.jsonl (500 items) in-repo, not on the Hub."""
import json
from pathlib import Path

INSTRUCTION = "Answer the question using the given context."
_PATH = Path(__file__).parent / "files" / "tipsheets.jsonl"


def load(split: str = "test"):
    if not _PATH.exists():
        raise FileNotFoundError(
            f"{_PATH} missing. Copy it from the KVComm repo "
            "(dataloader/data/tipsheets.jsonl)."
        )
    items = []
    with _PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            items.append({"instruction": INSTRUCTION, "context": row["context"],
                          "question": row["question"], "gold": row["answer"]})
    return items
```

Register it by adding `"tipsheets": tipsheets.load` to `_LOADERS` in `lkv/data/__init__.py` and importing the module alongside the others.

- [ ] **Step 2: Run both gates**

Run:
```bash
TASK=hotpotqa ./scripts/gate_kvcomm.sh && TASK=tipsheets ./scripts/gate_kvcomm.sh
```
Expected: twelve more JSON files. HotpotQA at 500 items with `max_new_tokens=48` is the longer of the two.

- [ ] **Step 3: Evaluate the gate**

**GATE K2 passes** if HotpotQA satisfies, under the metric chosen in Task 15: `baseline ∈ [0.13, 0.25]`, `skyline ≥ 0.66`, `r30 ≥ 0.26`, `r50 ≥ 0.53`, `r70 ≥ 0.62`, and `r30 < r50 < r70` strictly. SE at n=500 and p≈0.6 is 0.022, so these are 1.5·SE bands around 0.19 / 0.74 / 0.33 / 0.60 / 0.69.

Record all three datasets in a single `docs/FINDINGS.md` table: `dataset | condition | ours | paper | delta`. A dataset that reproduces on Countries but not HotpotQA points at context length, not at the trimming mechanism — note the mean sender-context token count for each dataset alongside the scores so that inference is available later.

- [ ] **Step 4: Confirm the CPU suite and the paired sanity check**

Run:
```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python - <<'EOF'
import json
from lkv.stats import mcnemar
r30 = json.load(open("runs/kv_hotpotqa_r30.json"))["per_item"]
base = json.load(open("runs/kv_hotpotqa_baseline.json"))["per_item"]
assert [x["i"] for x in r30] == [x["i"] for x in base]
print("r30 vs baseline:", mcnemar([x["correct"] for x in r30],
                                  [x["correct"] for x in base]))
EOF
```
Expected: the suite green, and r30 significantly above baseline (`p_exact < 0.01`). If the trimmed cache is not significantly better than no cache at all, nothing has been transmitted and K2 must be treated as failed regardless of the point estimates.

- [ ] **Step 5: Commit**

```bash
git add lkv/data/tipsheets.py lkv/data/__init__.py docs/FINDINGS.md runs/kv_hotpotqa_*.json runs/kv_tipsheets_*.json
git commit -m "test: KVComm HotpotQA and Tipsheets reproduction (gate K2)"
```

---

## Phase 5 — Integration

Only start Phase 5 once **L1, L2, K1 and K2 have all passed**. If any gate failed, the corresponding half of the integration cannot be interpreted, and that must be stated in `docs/FINDINGS.md` before any integrated number is produced.

### Task 17: Trim at the inter-agent boundary, with two scope semantics

This is the actual contribution. LatentMAS's cache grows monotonically and carries every agent's re-supplied question plus every agent's latent thoughts. KVComm's trim was defined for a single hop where the sender's cache *is* the message. Applying it unchanged to a growing relay collapses the entire history in unselected layers, which is far more aggressive than anything KVComm measured. So scope is a first-class axis, not an assumption:

- `trim_scope="all"` — KVComm-faithful. An unselected layer keeps `sink` positions of the whole cache. Maximum saving, maximum damage.
- `trim_scope="delta"` — an unselected layer keeps everything inherited from previous hops and only `sink` positions of the segment this agent added. Trims the *message*, not the history.

Reporting both is what separates this from re-demonstrating KVComm inside a wrapper: `delta` is a genuinely new operating point that only exists because the relay has more than one hop.

**Files:**
- Create: `lkv/integrate.py`
- Modify: `lkv/run.py` (wire `--system latentmas` + `--top_layers < 1.0` to build a hook)
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: `prepare_key_cache`/`retention` (Task 12), `select_layers` (Task 13), `run_latentmas` (Task 8).
- Produces:
  - `trim_delta(cache, layers_list, inherited_len, sink=1)` — unselected layers keep `[:inherited_len]` plus `sink` positions of the new segment.
  - `make_trim_hook(layers_list, scope: str, sink: int = 1, every_hop: bool = False, recalibrate=None) -> callable` matching the `trim_hook(cache, hop_index, inherited_len)` signature from Task 8.
  - `hook.records: list[dict]` accumulating `{"hop", "kept_layers", "positions_before", "positions_after", "retention"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_integration.py`:
```python
import torch

from lkv.config import RunConfig
from lkv.integrate import make_trim_hook, trim_delta
from lkv.kvcomm.trim import retention
from lkv.latentmas import past_length, run_latentmas
from tests.test_latentmas_loop import _StubTok


def _cache(model, n_pos):
    with torch.no_grad():
        return model(torch.randint(1, 200, (1, n_pos)), use_cache=True).past_key_values


def test_trim_delta_keeps_the_inherited_prefix(tiny_qwen3):
    cache = _cache(tiny_qwen3, 20)
    out = trim_delta(cache, layers_list=[2], inherited_len=12, sink=1)
    lengths = [int(k.shape[-2]) for k in out.key_cache]
    # layer 0 always full, layer 2 selected -> 20; layers 1 and 3 -> 12 + 1
    assert lengths == [20, 13, 20, 13], lengths


def test_trim_delta_at_first_hop_equals_scope_all(tiny_qwen3):
    from lkv.kvcomm.trim import prepare_key_cache
    cache = _cache(tiny_qwen3, 10)
    delta = trim_delta(cache, [2], inherited_len=0, sink=1)
    allsc = prepare_key_cache(cache, [2], sink=1)
    assert [int(k.shape[-2]) for k in delta.key_cache] == \
           [int(k.shape[-2]) for k in allsc.key_cache]


def test_scope_all_is_more_aggressive_than_delta(tiny_qwen3):
    cache = _cache(tiny_qwen3, 20)
    h_all = make_trim_hook([2], scope="all")
    h_delta = make_trim_hook([2], scope="delta")
    r_all = retention(h_all(cache, hop_index=1, inherited_len=12))
    r_delta = retention(h_delta(cache, hop_index=1, inherited_len=12))
    assert r_all < r_delta


def test_hook_records_one_entry_per_call(tiny_qwen3):
    hook = make_trim_hook([2], scope="all")
    cache = _cache(tiny_qwen3, 12)
    hook(cache, hop_index=0, inherited_len=0)
    hook(cache, hop_index=1, inherited_len=6)
    assert [r["hop"] for r in hook.records] == [0, 1]
    assert all(0.0 < r["retention"] <= 1.0 for r in hook.records)
    assert all(r["kept_layers"] == [2] for r in hook.records)


def test_every_hop_false_trims_only_the_first_boundary(tiny_qwen3):
    hook = make_trim_hook([2], scope="all", every_hop=False)
    cache = _cache(tiny_qwen3, 12)
    first = hook(cache, hop_index=0, inherited_len=0)
    second = hook(cache, hop_index=1, inherited_len=6)
    assert retention(first) < 1.0
    assert retention(second) == 1.0, "every_hop=False must pass hop 1 through"


def test_top_layers_one_is_a_no_op(tiny_qwen3):
    """Guard against silently trimming at full retention."""
    all_layers = list(range(tiny_qwen3.config.num_hidden_layers))
    hook = make_trim_hook(all_layers, scope="all")
    cache = _cache(tiny_qwen3, 12)
    assert retention(hook(cache, hop_index=0, inherited_len=0)) == 1.0


def test_relay_with_a_hook_shortens_the_cache(tiny_qwen3):
    cfg = RunConfig(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                    latent_steps=4, n_agents=4, greedy=True, max_new_tokens=4,
                    device="cpu", generate_bs=1, top_layers=0.5,
                    trim_scope="all")
    items = [{"question": "2+2?", "gold": "4"}]
    hook = make_trim_hook([2], scope="all", every_hop=True)
    _, hops = run_latentmas(tiny_qwen3, _StubTok(), cfg, items, trim_hook=hook)
    trimmed_hops = [h for h in hops if "positions_after_trim" in h]
    assert trimmed_hops, hops
    assert len(hook.records) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.integrate'`.

- [ ] **Step 3: Write the implementation**

`lkv/integrate.py`:
```python
"""KVComm trimming at LatentMAS agent boundaries.

Two scopes, because KVComm's trim was defined for a single hop where the
sender's cache IS the message, while LatentMAS's cache is a growing history:

  all    unselected layers keep `sink` positions of the whole cache. Faithful to
         stock-kvcomm/models.py:95-108, and far more aggressive here than
         anything KVComm measured, since the relay has three boundaries.
  delta  unselected layers keep everything inherited from earlier hops plus
         `sink` positions of the segment this agent added -- trims the message,
         not the history. Only meaningful with more than one hop.
"""
from __future__ import annotations

from transformers import DynamicCache

from lkv.kvcomm.trim import prepare_key_cache, retention


def trim_delta(cache, layers_list, inherited_len: int, sink: int = 1):
    keep = set(int(i) for i in layers_list)
    out = DynamicCache()
    for i in range(len(cache.key_cache)):
        k, v = cache.key_cache[i], cache.value_cache[i]
        if i in keep or i == 0:
            out.update(k, v, i)
        else:
            cut = inherited_len + sink
            out.update(k[:, :, :cut, :], v[:, :, :cut, :], i)
    return out


def make_trim_hook(layers_list, scope: str = "all", sink: int = 1,
                   every_hop: bool = False, recalibrate=None):
    """Build a trim_hook for run_latentmas.

    every_hop=False trims only the first boundary (one-shot); True trims at all
    of them, which is what compounds. recalibrate(cache, hop_index) -> layers
    lets Task 18 re-rank per hop.
    """
    if scope not in ("all", "delta"):
        raise ValueError(f"scope must be 'all' or 'delta', got {scope!r}")

    def hook(cache, hop_index: int, inherited_len: int):
        if not every_hop and hop_index != 0:
            return cache
        layers = recalibrate(cache, hop_index) if recalibrate else layers_list
        before = max(int(k.shape[-2]) for k in cache.key_cache)
        if scope == "all":
            out = prepare_key_cache(cache, layers, sink=sink)
        else:
            out = trim_delta(cache, layers, inherited_len, sink=sink)
        hook.records.append({
            "hop": hop_index,
            "kept_layers": sorted(int(i) for i in layers),
            "positions_before": before,
            "positions_after": max(int(k.shape[-2]) for k in out.key_cache),
            "retention": retention(out),
        })
        return out

    hook.records = []
    return hook
```

Then in `lkv/run.py`: when `cfg.system == "latentmas"` and `cfg.top_layers < 1.0`, run `calibrate` on the first `cfg.calib_size` items, call `select_layers(...)`, build the hook with `make_trim_hook(layers, cfg.trim_scope, every_hop=cfg.trim_every_hop)`, pass it to `run_latentmas`, and copy `hook.records` plus a mean `layer_retention` into the record.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_integration.py -v && ./.venv/bin/python -m pytest -q`
Expected: PASS, 7 new tests and the full suite green.

- [ ] **Step 5: Commit**

```bash
git add lkv/integrate.py lkv/run.py tests/test_integration.py
git commit -m "feat: KVComm trimming at LatentMAS boundaries, scopes all and delta"
```

---

### Task 18: Relay-native calibration

KVComm calibrates on the receiver's attention over a *sender's context*. In LatentMAS the "receiver" is the same model in its next role, and the "context" is the previous agents' contribution. Calibration must therefore be measured at an agent boundary, not on a KVComm-style pair, or the ranking is imported from a different distribution.

Two variants, and the comparison between them is a result:
- `calib_source="pair"` — reuse Task 14's KVComm-style calibration. Cheap, and the honest baseline for "we just dropped KVComm in".
- `calib_source="relay"` — trace the next role's prefill over the accumulated relay cache and rank from that.

**Files:**
- Create: `lkv/relay_calib.py`
- Modify: `lkv/config.py` (add `--calib_source {pair,relay}`, default `relay`), `lkv/run.py`
- Test: `tests/test_relay_calib.py`

**Interfaces:**
- Consumes: `install_tracer`/`remove_tracer` (Task 12), `attention_from_stash`/`layer_importance`/`select_layers` (Task 13), `generate_latent`/`past_length` (Task 8), prompts (Task 6).
- Produces: `calibrate_relay(model, tokenizer, cfg, items, hop_index: int = 0) -> dict[int, list[float]]`, same shape as Task 14's `calibrate` so `select_layers` consumes either.

- [ ] **Step 1: Write the failing test**

`tests/test_relay_calib.py`:
```python
import torch

from lkv.config import RunConfig
from lkv.relay_calib import calibrate_relay
from tests.test_latentmas_loop import _StubTok


def _cfg(**kw):
    base = dict(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                latent_steps=4, n_agents=4, greedy=True, max_new_tokens=4,
                device="cpu", generate_bs=1, calib_size=1, alpha=1.0,
                mu=0.5, sigma=10.0)
    base.update(kw)
    return RunConfig(**base)


_ITEMS = [{"question": "2+2?", "gold": "4"}, {"question": "3+3?", "gold": "6"}]


def test_returns_one_list_per_layer(tiny_qwen3):
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    assert sorted(totals) == [0, 1, 2, 3]
    assert all(len(v) == 1 for v in totals.values())
    assert all(isinstance(v[0], float) for v in totals.values())


def test_honours_calib_size(tiny_qwen3):
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(calib_size=2), _ITEMS)
    assert all(len(v) == 2 for v in totals.values())


def test_importance_is_finite_and_nonnegative(tiny_qwen3):
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    for values in totals.values():
        for v in values:
            assert v >= 0.0 and v == v and abs(v) < 1e6


def test_tracer_is_removed_afterwards(tiny_qwen3):
    calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    assert not hasattr(tiny_qwen3.model.layers[0].self_attn, "attn_inputs")


def test_later_hop_index_sees_a_longer_context(tiny_qwen3):
    """Calibrating at hop 2 accumulates two agents' output, so the sliced
    context is wider than at hop 0."""
    h0 = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS, hop_index=0)
    h2 = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS, hop_index=2)
    assert sorted(h0) == sorted(h2)
    assert h0 != h2, "hop index had no effect on calibration"


def test_ranking_is_usable_by_select_layers(tiny_qwen3):
    from lkv.kvcomm.importance import select_layers
    totals = calibrate_relay(tiny_qwen3, _StubTok(), _cfg(), _ITEMS)
    layers = select_layers(totals, 0.5, alpha=1.0, mu=0.5, sigma=10.0)
    assert len(layers) == 2 and all(0 <= i < 4 for i in layers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_relay_calib.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.relay_calib'`.

- [ ] **Step 3: Write the implementation**

`lkv/relay_calib.py`:
```python
"""Calibrate layer importance at a LatentMAS agent boundary.

KVComm ranks layers by how much attention a RECEIVER pays to a SENDER's context
(stock-kvcomm/layer_importance.py:21). Here the receiver is the same model in its
next role and the context is the accumulated relay cache, so importance must be
measured there -- otherwise the ranking comes from a different distribution than
the one it will be applied to.
"""
from __future__ import annotations

import torch

from lkv.kvcomm.importance import attention_from_stash, layer_importance
from lkv.kvcomm.tracer import install_tracer, remove_tracer
from lkv.latentmas import generate_latent, past_length
from lkv.modeling import pad_batch
from lkv.prompts import ROLES, latent_agent_message
from lkv.realign import LatentProjector


def _render(tokenizer, cfg, item, role):
    text = tokenizer.apply_chat_template(
        latent_agent_message(role, item["question"], cfg.task),
        tokenize=False, add_generation_prompt=True,
    )
    return pad_batch(tokenizer, [text])


def calibrate_relay(model, tokenizer, cfg, items, hop_index: int = 0) -> dict:
    """Importance of each layer at the boundary AFTER agent `hop_index`."""
    totals = {i: [] for i in range(model.config.num_hidden_layers)}
    projector = LatentProjector(model, cfg.realign, cfg.renorm_latent, cfg.device)
    roles = list(ROLES[: cfg.n_agents - 1])
    if hop_index >= len(roles):
        raise ValueError(f"hop_index {hop_index} exceeds {len(roles)} latent agents")

    for item in items[: cfg.calib_size]:
        # Build the cache up to and including the agent at hop_index.
        cache = None
        for role in roles[: hop_index + 1]:
            ids, mask = _render(tokenizer, cfg, item, role)
            cache, _ = generate_latent(
                model, projector, ids.to(model.device), mask.to(model.device),
                cfg.latent_steps, cache,
            )

        # Trace the NEXT role's prefill over that cache.
        next_role = roles[hop_index + 1] if hop_index + 1 < len(roles) else "judger"
        ids, mask = _render(tokenizer, cfg, item, next_role)
        ids, mask = ids.to(model.device), mask.to(model.device)
        past = past_length(cache)
        full_mask = torch.cat(
            [torch.ones(1, past, dtype=mask.dtype, device=mask.device), mask], dim=1
        )
        saved = install_tracer(model)
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=full_mask,
                      past_key_values=cache, use_cache=True)
            q_len = int(ids.shape[1])
            attn = {
                i: attention_from_stash(layer.self_attn, layer.self_attn.attn_inputs,
                                        q_len_receiver=q_len)
                for i, layer in enumerate(model.model.layers)
            }
            for i, value in layer_importance(attn, q_len=q_len).items():
                totals[i].append(float(value))
        finally:
            remove_tracer(model, saved)
    return totals
```

Add `--calib_source` to `lkv/config.py` with `choices=["pair", "relay"]`, `default="relay"`, and a matching `calib_source: str = "relay"` field on `RunConfig`. In `lkv/run.py`, dispatch on it when building the hook.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_relay_calib.py -v && ./.venv/bin/python -m pytest -q`
Expected: PASS, 6 new tests and the full suite green.

- [ ] **Step 5: Commit**

```bash
git add lkv/relay_calib.py lkv/config.py lkv/run.py tests/test_relay_calib.py
git commit -m "feat: relay-native layer calibration at agent boundaries"
```

---

### Task 19: Efficiency instrumentation

Accuracy alone cannot support the claim. KVComm's own paper reports **no** measured bytes, memory, or latency — only FLOPs ratios and the asserted `1/0.3` layer ratio. So measuring bytes and memory here is new information rather than a re-derivation, and it must be measured, not computed from a formula.

**Files:**
- Create: `lkv/instrument.py`
- Modify: `lkv/run.py`
- Test: `tests/test_instrument.py`

**Interfaces:**
- Consumes: nothing beyond torch.
- Produces:
  - `cache_bytes(cache) -> int` — actual `numel * element_size` summed over both K and V across all layers.
  - `attention_positions(cache) -> int` — sum of per-layer key lengths, the quantity attention cost is linear in.
  - `PeakMemory()` context manager exposing `.peak_bytes` after exit; a no-op returning 0 on CPU.
  - `Timer()` context manager exposing `.elapsed`.
  - `theoretical_bytes_per_position(config, dtype) -> int` for cross-checking the measurement.

- [ ] **Step 1: Write the failing test**

`tests/test_instrument.py`:
```python
import torch

from lkv.instrument import (
    PeakMemory,
    Timer,
    attention_positions,
    cache_bytes,
    theoretical_bytes_per_position,
)
from lkv.kvcomm.trim import prepare_key_cache


def _cache(model, n_pos):
    with torch.no_grad():
        return model(torch.randint(1, 200, (1, n_pos)), use_cache=True).past_key_values


def test_cache_bytes_matches_the_theoretical_figure(tiny_qwen3):
    n_pos = 16
    cache = _cache(tiny_qwen3, n_pos)
    per_pos = theoretical_bytes_per_position(tiny_qwen3.config, torch.float32)
    assert cache_bytes(cache) == per_pos * n_pos


def test_trimming_reduces_measured_bytes(tiny_qwen3):
    cache = _cache(tiny_qwen3, 20)
    full = cache_bytes(cache)
    trimmed = cache_bytes(prepare_key_cache(cache, layers_list=[2]))
    assert trimmed < full
    # layers 0 and 2 keep 20, layers 1 and 3 keep 1 -> 42/80 of the positions
    assert abs(trimmed / full - 42 / 80) < 1e-9


def test_attention_positions_sums_every_layer(tiny_qwen3):
    trimmed = prepare_key_cache(_cache(tiny_qwen3, 20), layers_list=[2])
    assert attention_positions(trimmed) == 20 + 1 + 20 + 1


def test_timer_measures_something_positive():
    with Timer() as t:
        sum(range(200000))
    assert t.elapsed > 0.0


def test_peak_memory_is_zero_and_safe_on_cpu():
    with PeakMemory("cpu") as m:
        torch.ones(1000, 1000)
    assert m.peak_bytes == 0


def test_theoretical_bytes_uses_kv_heads_not_query_heads(tiny_qwen3):
    """GQA: cost scales with num_key_value_heads (2), not num_attention_heads (8)."""
    cfg = tiny_qwen3.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    expected = cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2 * 4
    assert theoretical_bytes_per_position(cfg, torch.float32) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_instrument.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lkv.instrument'`.

- [ ] **Step 3: Write the implementation**

`lkv/instrument.py`:
```python
"""Measured efficiency, not derived efficiency.

KVComm's paper reports no bytes, no memory, and no latency -- only FLOPs ratios
and the asserted 1/0.3 layer ratio. Everything here is read off real tensors so
the numbers stand on their own.
"""
from __future__ import annotations

import time

import torch


def cache_bytes(cache) -> int:
    total = 0
    for k, v in zip(cache.key_cache, cache.value_cache):
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def attention_positions(cache) -> int:
    """Sum of per-layer key lengths -- attention cost is linear in this."""
    return sum(int(k.shape[-2]) for k in cache.key_cache)


def theoretical_bytes_per_position(config, dtype: torch.dtype) -> int:
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    element = torch.empty(0, dtype=dtype).element_size()
    return (config.num_hidden_layers * config.num_key_value_heads * head_dim
            * 2 * element)


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


class PeakMemory:
    """Peak allocated bytes on CUDA; a no-op returning 0 elsewhere."""

    def __init__(self, device="cuda:0"):
        self.device = str(device)
        self.peak_bytes = 0
        self._active = self.device.startswith("cuda") and torch.cuda.is_available()

    def __enter__(self):
        if self._active:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc):
        if self._active:
            torch.cuda.synchronize(self.device)
            self.peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
        return False
```

In `lkv/run.py`, wrap the batch loop in `Timer` and `PeakMemory(cfg.device)`, and add `peak_mem_bytes`, plus `cache_bytes_at_final_hop` and `attention_positions_at_final_hop` read from the last `hop_records` entry, to the record.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_instrument.py -v && ./.venv/bin/python -m pytest -q`
Expected: PASS, 6 new tests and the full suite green.

- [ ] **Step 5: Commit**

```bash
git add lkv/instrument.py lkv/run.py tests/test_instrument.py
git commit -m "feat: measured cache bytes, attention positions, peak memory, wall clock"
```

---

### Task 20: The integration sweep and its analysis

**Files:**
- Create: `scripts/sweep_integration.sh`, `scripts/analyze.py`
- Modify: `docs/FINDINGS.md`

**Interfaces:**
- Consumes: everything.
- Produces: `runs/int_*.json` and a final table in `docs/FINDINGS.md`.

The design uses **MBPP+ 8B** as the primary benchmark and GSM8K 8B as a secondary. Reason: whichever of the two showed the larger paired relay-versus-single gap in Task 11 is the only one where trimming has something to damage, and MBPP+ is the more likely of the two on the published numbers (+5.1 over TextMAS versus +1.5). If Task 11 found GSM8K single-agent near the LatentMAS score, GSM8K cannot discriminate and must be reported as such rather than swept.

- [ ] **Step 1: Write the sweep script**

`scripts/sweep_integration.sh`:
```bash
#!/usr/bin/env bash
# Phase 5 sweep. --greedy throughout: these are internal A/B comparisons, and
# mixing sampled and greedy runs inside one comparison invalidates the pairing.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-Qwen/Qwen3-8B}
TASK=${TASK:-mbppplus}
N=${N:-378}
S_BEST=${S_BEST:?set S_BEST to the latent_steps chosen in Task 10}

run() {
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task "$TASK" --model_name "$M" --max_samples "$N" \
      --latent_steps "$S_BEST" --greedy --seed 42 --generate_bs 8 \
      --out_json "$f" "$@" 2>&1 | tail -15
}

# Reference points
run "int_${TASK}_full"   --system latentmas --top_layers 1.0
run "int_${TASK}_single"  --system single

# The two scopes x three ratios, trimming at EVERY boundary so it compounds
for SCOPE in all delta; do
  for R in 0.3 0.5 0.7; do
    run "int_${TASK}_${SCOPE}_r${R/0./}0" --system latentmas --top_layers "$R" \
        --trim_scope "$SCOPE" --trim_every_hop --calib_source relay
  done
done

# One-shot (first boundary only), to separate compounding from trimming
run "int_${TASK}_all_r50_oneshot" --system latentmas --top_layers 0.5 \
    --trim_scope all --calib_source relay

# Controls: random layers, and the imported pair-style calibration
run "int_${TASK}_all_r50_random" --system latentmas --top_layers 0.5 \
    --trim_scope all --trim_every_hop --random_selection
run "int_${TASK}_all_r50_paircal" --system latentmas --top_layers 0.5 \
    --trim_scope all --trim_every_hop --calib_source pair

$PY scripts/analyze.py "$TASK"
```

`scripts/analyze.py`:
```python
"""Turn the sweep JSONs into the table that goes in docs/FINDINGS.md.

Every trimmed condition is compared to the full relay with McNemar, because both
ran greedily on the same seeded slice and are therefore paired.
"""
import glob
import json
import os
import sys

from lkv.stats import mcnemar

task = sys.argv[1] if len(sys.argv) > 1 else "mbppplus"
runs = {}
for path in sorted(glob.glob(f"runs/int_{task}_*.json")):
    runs[os.path.basename(path)[:-5]] = json.load(open(path))

full = runs.get(f"int_{task}_full")
if full is None:
    raise SystemExit(f"missing reference run int_{task}_full")

header = (f"{'condition':34} {'acc':>7} {'d_acc':>7} {'layer_ret':>9} "
          f"{'bytes_ret':>9} {'attn_ret':>8} {'s/item':>7} {'p_vs_full':>9}")
print(header)
print("-" * len(header))

base_bytes = full.get("cache_bytes_at_final_hop") or 0
base_attn = full.get("attention_positions_at_final_hop") or 0

for name, d in runs.items():
    same_slice = [x["i"] for x in d["per_item"]] == [x["i"] for x in full["per_item"]]
    p = mcnemar([x["correct"] for x in d["per_item"]],
                [x["correct"] for x in full["per_item"]])["p_exact"] if same_slice else float("nan")
    b = d.get("cache_bytes_at_final_hop") or 0
    a = d.get("attention_positions_at_final_hop") or 0
    print(f"{name:34} {d['accuracy']:7.4f} {d['accuracy'] - full['accuracy']:+7.4f} "
          f"{d.get('layer_retention', 1.0):9.4f} "
          f"{(b / base_bytes if base_bytes else 1.0):9.4f} "
          f"{(a / base_attn if base_attn else 1.0):8.4f} "
          f"{d['sec_per_item']:7.2f} {p:9.4f}")

print("\nNote: p_vs_full is McNemar exact-binomial against int_%s_full." % task)
print("A condition with p >= 0.05 is NOT shown to be worse -- it is also not "
      "shown to be equal. State sample size when reporting.")
```

- [ ] **Step 2: Run the sweep**

Run: `chmod +x scripts/sweep_integration.sh && S_BEST=<winner> TASK=mbppplus N=378 ./scripts/sweep_integration.sh`
Expected: eleven JSON files and the printed table. Budget roughly 4–6 GPU hours at 378 items.

- [ ] **Step 3: Write up the findings honestly**

Add a `## Phase 5 — integration results` section to `docs/FINDINGS.md` containing the table verbatim, plus explicit answers to these five questions. Each answer must cite the numbers.

1. **Does trimming preserve accuracy?** For each scope and ratio, the accuracy delta versus the full relay and its McNemar `p_exact`. A non-significant delta at n=378 is not proof of equivalence — report the confidence interval as well.
2. **What is actually saved?** Measured `bytes_ret` and `attn_ret`, not the layer ratio. If `bytes_ret` for `scope=all, r=0.3` is not close to the layer ratio, explain why (layer 0 is retained in full, and the sink keeps one position per dropped layer).
3. **Does `delta` beat `all`?** This is the only genuinely new axis, since it exists only because the relay has more than one hop. Report both accuracy and savings; `delta` saving less while preserving more accuracy is a real result, and so is the opposite.
4. **Is the ranking doing work?** Compare against `random`. If random matches, the mechanism is retention, not selection. The paper itself reports random beating KVComm on 5 of 8 datasets at ratio 0.7 for the Llama-8B pair, so state this comparison plainly either way.
5. **Does relay-native calibration beat imported pair calibration?** Compare `calib_source=relay` against `paircal`. If they tie, say so — it means KVComm's ranking transfers, which is a finding, not a failure.

State explicitly whether the full relay beat single-agent on this task (from Task 11). **If it did not, every trimming number in the table is measuring the erosion of something that was not helping, and the write-up must lead with that.**

- [ ] **Step 4: Confirm the suite and the reproducibility of one run**

Run:
```bash
./.venv/bin/python -m pytest -q
# byte-identical re-run under --greedy
mv runs/int_mbppplus_all_r50.json /tmp/check.json
S_BEST=<winner> TASK=mbppplus N=378 ./scripts/sweep_integration.sh >/dev/null
./.venv/bin/python -c "
import json
a=json.load(open('/tmp/check.json')); b=json.load(open('runs/int_mbppplus_all_r50.json'))
assert a['per_item']==b['per_item'], 'greedy run is not deterministic'
print('reproducible: accuracy', a['accuracy'])"
```
Expected: the suite green and the per-item outcomes identical. Non-determinism under `--greedy` invalidates every paired test in the table and must be fixed before publishing any of it.

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_integration.sh scripts/analyze.py docs/FINDINGS.md runs/int_*.json
git commit -m "test: integration sweep across scopes, ratios, and controls"
```

---

## Appendix — decisions already made, so they are not re-litigated

- **No custom attention backend.** Follow KVComm: subclass the attention module, leave `_attn_implementation == "sdpa"`. Registering a backend without also registering a mask builder makes attention bidirectional with no error raised.
- **`padding_side="left"` everywhere**, and slice generated text at the *padded* prompt width.
- **`latent_steps` has no default.** A zero default silently converts LatentMAS into single-agent.
- **Realign and renorm are separate flags**, because upstream applies the renorm unconditionally and the realign-off/renorm-off cell is untested by any upstream configuration.
- **Both F1 variants are always computed and reported**; Task 15 decides which reproduces, and every table names the metric it uses.
- **Reproduction runs use sampled decoding** (LatentMAS: `temperature=0.6`, `top_p=0.95`; KVComm: greedy, per its own code). **Internal comparisons use `--greedy`.** Never mixed inside one comparison.
- **Paired tests only.** Every run writes `per_item` on a seeded slice, so McNemar applies.
- **Benchmarks not used, and why.** AIME is unusable as a measurement instrument at n=30 (SE ≈ 8–9 points). HumanEval+ at n=164 is a weaker version of MBPP+. ARC and MedQA are near ceiling for these models, so trimming cannot be distinguished from noise.
