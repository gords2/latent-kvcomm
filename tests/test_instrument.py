import torch

from lkv.instrument import (
    PeakMemory,
    Timer,
    attention_positions,
    cache_bytes,
    theoretical_bytes_per_position,
)
from lkv.kvcomm.trim import prepare_key_cache, retention


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


def test_measured_byte_ratio_matches_reported_retention(tiny_qwen3):
    """Cross-check: bytes come from tensor sizes, retention from position
    counts -- two different code paths computing what should be the same
    saving. Agreement here is evidence neither one is misreporting it;
    disagreement would mean the paper's headline ratio is wrong somewhere.

    This holds exactly (not just approximately) for THIS model because every
    layer shares one num_key_value_heads/head_dim, so a layer's byte cost is
    a constant multiple of its position count -- the byte ratio and the
    position ratio (what `retention` computes) collapse to the same fraction.
    """
    full = _cache(tiny_qwen3, 20)
    trimmed = prepare_key_cache(full, layers_list=[2])
    byte_ratio = cache_bytes(trimmed) / cache_bytes(full)
    assert abs(byte_ratio - retention(trimmed)) < 1e-9
