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
