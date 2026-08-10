"""Layer importance and ranking, semantically identical to
stock-kvcomm/layer_importance.py. tests/test_importance.py holds a verbatim copy
of the upstream functions and asserts agreement on random input."""
from __future__ import annotations

import random

import numpy as np
import torch
from transformers.models.llama.modeling_llama import repeat_kv


def normalize(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo + 1e-9)


def gaussian_prior(n_layers: int, mu_abs: float, sigma: float) -> np.ndarray:
    x = np.arange(n_layers, dtype=np.float32)
    return np.exp(-0.5 * ((x - mu_abs) / float(sigma)) ** 2)


def attention_from_stash(attn_module, stash: dict, q_len_receiver: int) -> torch.Tensor:
    """Recompute post-softmax attention from the stashed query and cached keys.

    Mirrors stock-kvcomm/models.py:167-186: softmax(Q K^T * scaling + causal),
    with GQA key heads repeated so every QUERY head is represented. `query` and
    `key` share a rotary basis (tracer.py's _project_q applies RoPE to the
    stashed query to match the cached key) so no RoPE is re-applied here.

    n_rep is read off attn_module.num_key_value_groups rather than inferred
    from q.shape[1] // k.shape[1] -- the shapes would agree even if repeat_kv
    had already run (a stale stash, for instance), so the module's own
    configured group count is the more trustworthy source. q_len_receiver is
    cross-checked against the tensor for the same reason layer_importance
    cross-checks its own q_len: it cannot disagree with the data.
    """
    q = stash["query"]
    k = stash["key"]
    if q_len_receiver != q.shape[-2]:
        raise ValueError(
            f"q_len_receiver={q_len_receiver} but stashed query has "
            f"{q.shape[-2]} positions"
        )
    n_rep = attn_module.num_key_value_groups
    k = repeat_kv(k, n_rep)
    scores = torch.matmul(q, k.transpose(2, 3)) * stash["scaling"]

    kv_len, q_len = scores.shape[-1], scores.shape[-2]
    # The cached prefix (sender context, already in the KV cache before this
    # receiver prefill ran) is `kv_len - q_len` positions. Query row i sits at
    # absolute position offset+i, so it must see columns 0..offset+i inclusive
    # -- tril(diagonal=offset) keeps exactly col <= row + offset.
    offset = kv_len - q_len
    causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).tril(offset)
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1, dtype=torch.float32)


def layer_importance(attn_by_layer: dict, q_len: int | None = None) -> dict:
    """Mass each receiver query puts on context keys 1..len(context)-1.

    The slice is upstream's `attn[..., 1:-q_len]` (layer_importance.py:21): drop
    key position 0, and drop the receiver's own q_len positions at the tail.
    q_len is taken from the tensor itself so it cannot disagree with the data;
    the parameter is accepted only to cross-check the caller.
    """
    out = {}
    for i, weights in attn_by_layer.items():
        tail = weights.shape[-2]
        if q_len is not None and q_len != tail:
            raise ValueError(
                f"layer {i}: caller says q_len={q_len} but tensor has {tail} queries"
            )
        value = weights[..., 1:-tail].sum(dim=-1).mean()
        out[i] = 0 if torch.isnan(value) else value.item()
    return out


def layer_ranking(importance_totals: dict, alpha: float, mu: float,
                  sigma: float) -> list[int]:
    n = len(importance_totals)
    if n != max(importance_totals) + 1:
        raise ValueError("importance_totals must be keyed 0..n-1 with no gaps")
    means = [float(np.mean(importance_totals[i])) for i in range(n)]
    score = alpha * normalize(means) + (1.0 - alpha) * normalize(
        gaussian_prior(n, mu * (n - 1), sigma)
    )
    return [int(i) for i in np.argsort(score)[::-1]]


def select_layers(importance_totals: dict, top_layers: float, alpha: float,
                  mu: float, sigma: float, random_selection: bool = False,
                  seed: int = 42) -> list[int]:
    n = len(importance_totals)
    count = int(top_layers * n)          # floor, as upstream :29
    if random_selection:
        return sorted(random.Random(seed).sample(range(n), count))
    return layer_ranking(importance_totals, alpha, mu, sigma)[:count]
