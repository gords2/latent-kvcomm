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


def test_qa_instruction_is_upstreams_verbatim_and_shared_by_every_qa_task():
    """D18 regression -- the defect that made K1 unmeasurable.

    A paraphrase of this string ("Answer the question using the given
    context.") is what produced accuracy 0.0000 on all six Countries
    conditions, skyline included. Countries generates at max_new_tokens=5, so
    the instruction has to suppress a preamble outright or the answer never
    reaches the output: measured, the model emitted 'The context is the
    Australian' for gold 'Australia' and was truncated mid-sentence.

    Pinned as a literal, not compared against an import of the same constant --
    that would pass no matter what the constant said. Every QA task must share
    it, since upstream routes all three through eval.py's QA branch.
    """
    from lkv.data import countries, hotpotqa
    from lkv.prompts import QA_INSTRUCTION

    assert QA_INSTRUCTION == (
        "Directly answer the question based on the context passage, "
        "no explanation is needed."
    )
    assert countries.INSTRUCTION == QA_INSTRUCTION
    assert hotpotqa.INSTRUCTION == QA_INSTRUCTION


def test_kvcomm_templates_separate_context_from_question():
    s = kvcomm_sender_prompt("INSTR", "CTX")
    r = kvcomm_receiver_prompt("INSTR", "Q")
    k = kvcomm_skyline_prompt("INSTR", "CTX", "Q")
    assert "CTX" in s and "Q" not in s
    assert "Q" in r and "CTX" not in r
    assert "CTX" in k and "Q" in k
