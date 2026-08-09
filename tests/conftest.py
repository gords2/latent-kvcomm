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


@pytest.fixture(scope="session")
def _capture_model_state(tiny_qwen3, tiny_llama):
    """Capture initial state of shared session models for leak detection.

    Stores the original forward method function objects and training state
    so we can detect if tests leave models modified (replaced forward methods,
    attached attributes, or left in train mode).
    """
    qwen3_state = {}
    llama_state = {}

    # Store the underlying function objects of forward methods
    for i, layer in enumerate(tiny_qwen3.model.layers):
        qwen3_state[i] = layer.self_attn.forward.__func__

    for i, layer in enumerate(tiny_llama.model.layers):
        llama_state[i] = layer.self_attn.forward.__func__

    return {
        "tiny_qwen3": {"model": tiny_qwen3, "forward_funcs": qwen3_state},
        "tiny_llama": {"model": tiny_llama, "forward_funcs": llama_state},
    }


@pytest.fixture(autouse=True)
def _guard_model_state(_capture_model_state):
    """Guard against state leakage from session-scoped models.

    Runs after each test and verifies that shared models remain unmodified:
    - training mode is still False (model.eval())
    - no leftover attn_inputs attributes on layers
    - forward methods are still the original bound methods

    Fails loudly if a test leaves a model dirty, catching bugs that would
    otherwise silently corrupt subsequent tests.
    """
    yield  # Let test run

    # Verify after test
    for model_name, data in _capture_model_state.items():
        model = data["model"]
        forward_funcs = data["forward_funcs"]

        # Check training mode: should be False (eval)
        if model.training:
            raise AssertionError(
                f"{model_name} was left in train mode. "
                f"Tests must call model.eval() after any modifications."
            )

        # Get model layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            raise AssertionError(f"Cannot find model.layers in {model_name}")

        for i, layer in enumerate(layers):
            # Check for leftover attn_inputs or other attached attributes
            if hasattr(layer, "attn_inputs"):
                raise AssertionError(
                    f"{model_name} layer {i} has leftover attn_inputs attribute. "
                    f"Tests must remove attached attributes in finally blocks."
                )

            # Check forward method wasn't replaced
            current_func = layer.self_attn.forward.__func__
            original_func = forward_funcs[i]
            if current_func is not original_func:
                raise AssertionError(
                    f"{model_name} layer {i}.self_attn.forward was replaced. "
                    f"Tests must restore original forward in finally blocks."
                )
