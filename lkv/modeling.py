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


def chat_pad_batch(tokenizer, messages: list[str], enable_thinking: bool = True):
    """Render each message as a one-turn user chat, then pad the batch.

    Every KVComm path must go through here, because upstream does: eval.py's
    apply_chat_template (:47) wraps sender, receiver, skyline and baseline
    messages alike in `[{"role": "user", ...}]` with add_generation_prompt=True.

    Feeding the bare "Instruction: ... Context: ... Question: ..." string
    instead -- which this project did until D19 -- is not a milder version of
    the same thing. It turns the task into document continuation: an Instruct
    model handed an unterminated transcript continues the FORMAT rather than
    answering it, and the measured output was 'Answer: Australia\\nInstruction'
    where the bare answer was wanted. On Countries that costs everything,
    because max_new_tokens is 5 and the F1 threshold is a strict `> 0.5`: one
    echoed word drags a correct answer to exactly 0.5, which scores as wrong.
    77 of 200 items failed precisely that way. See D19 in docs/FINDINGS.md.

    add_special_tokens=False inside pad_batch stays correct: the rendered
    template already carries BOS, so tokenizing it again would double it.
    """
    # enable_thinking is forwarded ONLY when False -- Qwen3's template accepts
    # it, other families' templates raise on an unexpected kwarg, and True is
    # the template default anyway. Same rule as lkv/latentmas.py's _render.
    extra = {} if enable_thinking else {"enable_thinking": False}
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": m}],
            tokenize=False, add_generation_prompt=True, **extra,
        )
        for m in messages
    ]
    return pad_batch(tokenizer, texts)


def pad_batch(tokenizer, texts: list[str]):
    """Tokenize a batch, forcing left padding regardless of tokenizer state."""
    tokenizer.padding_side = "left"
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False,
    )
    return enc["input_ids"], enc["attention_mask"]
