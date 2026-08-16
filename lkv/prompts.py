"""Role prompts for LatentMAS and the three KVComm templates.

LatentMAS interpolates the full question into every role's prompt and never
uses the `context` argument of its latent builders (stock-latentmas/prompts.py:
12,20,36,51). That duplication is preserved deliberately -- it is a property of
the system the paper measured, not a bug to fix here.
"""
from __future__ import annotations

ROLES = ("planner", "critic", "refiner", "judger")

# Upstream's QA_INSTRUCTION, verbatim (stock-kvcomm/eval.py:10). Every QA task
# shares it: countries, hotpotqa and tipsheets all fall through to eval.py's
# `else` branch, which is the QA one.
#
# The exact wording is load-bearing, and "no explanation is needed" carries the
# weight. Countries generates at max_tokens=5 (their CountriesEvaluator sets
# it), which suffices only if the model answers with the bare country and
# nothing else. This project previously invented a milder paraphrase --
# "Answer the question using the given context." -- under which
# Llama-3.1-8B-Instruct opens with a sentence instead: measured, it emitted
# 'The context is the Australian' for gold 'Australia' and was cut off at token
# 5 with the answer never reaching the output. That produced accuracy 0.0000
# across all six Countries conditions INCLUDING skyline, plus a 0.0020 HotpotQA
# baseline. See D18 in docs/FINDINGS.md. Do not paraphrase this string.
QA_INSTRUCTION = (
    "Directly answer the question based on the context passage, "
    "no explanation is needed."
)

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
