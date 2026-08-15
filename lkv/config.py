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

from lkv.prompts import ROLES

MODEL_CHOICES = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # Ungated mirror of the entry above. The meta-llama repos are gated behind
    # accepting Meta's licence, which requires a Hugging Face account; the Qwen
    # repos are Apache-2.0 and download anonymously. KVComm's published numbers
    # are Llama numbers, so without this K1/K2 would be unreachable without an
    # account. Config verified against the published Llama-3.1-8B-Instruct
    # spec: 32 layers, hidden 4096, intermediate 14336, 32 heads / 8 KV heads,
    # vocab 128256, rope_theta 500000, bf16, 16.07 GB across 4 shards.
    # Recorded as a MIRROR, not asserted byte-identical (the gated original
    # cannot be fetched for comparison) -- see docs/FINDINGS.md.
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
]

TASK_CHOICES = ["gsm8k", "mbppplus", "hotpotqa", "countries", "tipsheets"]

# gsm8k/mbppplus items carry only question/gold (lkv/data/gsm8k.py,
# lkv/data/mbppplus.py) -- no instruction/context. --calib_source pair's
# kvcomm_sender_prompt(it["instruction"], it["context"]) (lkv/kvcomm/pair.py)
# needs both, so pairing it with either task is a KeyError waiting to happen
# at run time, not argument-parse time. See parse_args.
_TASKS_WITHOUT_CONTEXT = {"gsm8k", "mbppplus"}

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
    # "pair": reuse Task 14's KVComm-style sender/receiver calibration (cheap,
    # but imports a ranking from a different distribution than the relay it
    # gets applied to). "relay": trace the next role's prefill over the
    # accumulated relay cache itself -- see lkv/relay_calib.py.
    calib_source: str = "relay"
    # Decode over a layer-trimmed cache with per-layer causal masks instead of
    # the stock forward (D8). OFF by default because KVComm's published table
    # was produced on the stock, non-causal path -- reproducing their numbers
    # requires it. Turn it ON for the integration, where there is no upstream
    # to be faithful to. See docs/FINDINGS.md.
    causal_ragged: bool = False
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
    p.add_argument("--system", required=True,
                   choices=["latentmas", "kvcomm", "single", "textmas", "skyline", "baseline"])
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
    p.add_argument("--calib_source", default="relay", choices=["pair", "relay"])
    p.add_argument("--causal_ragged", action="store_true",
                   help="decode a layer-trimmed cache with per-layer causal "
                        "masks (D8). Off reproduces KVComm as published; on is "
                        "correct and is what the integration should use.")
    p.add_argument("--trim_scope", default="all", choices=["all", "delta"])
    p.add_argument("--trim_every_hop", action="store_true")
    p.add_argument("--out_json", default="")
    ns = p.parse_args(argv)

    if ns.system in ("latentmas", "textmas") and ns.latent_steps is None:
        p.error("--latent_steps is required for --system latentmas/textmas "
                "(a 0 default silently disables the relay; see D2)")
    if ns.latent_steps is None:
        ns.latent_steps = 0

    if ns.calib_source == "pair" and ns.task in _TASKS_WITHOUT_CONTEXT:
        p.error(
            f"--calib_source pair cannot run on --task {ns.task}: it calibrates "
            "via kvcomm_sender_prompt(instruction, context), and "
            f"{sorted(_TASKS_WITHOUT_CONTEXT)} items carry neither field "
            "(lkv/data/gsm8k.py, lkv/data/mbppplus.py emit only "
            "question/gold). Tasks with instruction/context are "
            "hotpotqa, countries, and tipsheets. Use --calib_source relay "
            "instead, or pick one of those tasks."
        )

    if ns.causal_ragged and ns.system not in ("kvcomm", "latentmas"):
        p.error(
            f"--causal_ragged does nothing for --system {ns.system}: only "
            "kvcomm (run_pair) and latentmas (the relay's latent hops and "
            "judger) ever decode over a trimmed cache. Accepting it here would "
            "stamp ragged_path onto a run where no ragged cache existed."
        )

    if ns.n_agents > len(ROLES):
        p.error(
            f"--n_agents {ns.n_agents} exceeds the {len(ROLES)} defined roles "
            f"{ROLES} -- run_latentmas would reuse 'judger' twice and the relay "
            "mask/cache shapes would desync (see lkv/decode.py's ValueError on "
            "past_mask shape). Pick --n_agents <= 4."
        )

    kw = vars(ns)
    if not kw["model_b"]:
        kw["model_b"] = kw["model_name"]
    if kw["alpha"] < 0:
        kw["alpha"] = default_alpha(kw["model_name"])
    if kw["max_new_tokens"] == 0:
        kw["max_new_tokens"] = task_defaults(kw["task"])
    return RunConfig(**kw)
