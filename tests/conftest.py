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
