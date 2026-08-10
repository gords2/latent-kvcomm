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
