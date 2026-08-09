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
