"""Stash prefill query/key so layer importance can be recomputed.

Follows stock-kvcomm/model_attn.py exactly in the one respect that matters:
the attention module keeps computing through its own real forward, and
config._attn_implementation is left as "sdpa". Changing that string would
send transformers down the custom-backend path in masking_utils, which
returns attention_mask=None and makes attention bidirectional. Upstream
achieves this by subclassing LlamaAttention/Qwen2Attention/etc. per
architecture; we patch attn.forward in place instead so the same tracer
works across architectures without a subclass per model family.

Verified against transformers 4.53.3
(transformers/models/qwen3/modeling_qwen3.py and
transformers/models/llama/modeling_llama.py): both Qwen3Attention.forward and
LlamaAttention.forward have the identical signature
(hidden_states, position_embeddings, attention_mask, past_key_value=None,
cache_position=None, **kwargs), and both decoder layers call self.self_attn
with every argument passed by keyword -- so the closure below binds correctly
regardless of argument order.
"""
from __future__ import annotations


def _patch(attn_module):
    original_forward = attn_module.forward

    def traced_forward(hidden_states, position_embeddings, attention_mask=None,
                       past_key_value=None, cache_position=None, **kwargs):
        # Prefill only (upstream model_attn.py:49-59). A decode step must not
        # clobber the stash -- importance is defined over the prefill attention.
        if hidden_states.shape[-2] > 1:
            attn_module.attn_inputs = {
                "hidden_states": hidden_states.detach(),
                "position_embeddings": tuple(p.detach() for p in position_embeddings),
                "scaling": attn_module.scaling,
            }
        out = original_forward(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            past_key_value=past_key_value, cache_position=cache_position, **kwargs
        )
        if hidden_states.shape[-2] > 1 and past_key_value is not None:
            idx = attn_module.layer_idx
            attn_module.attn_inputs["query"] = _project_q(attn_module, hidden_states)
            attn_module.attn_inputs["key"] = past_key_value.key_cache[idx].detach()
        return out

    return original_forward, traced_forward


def _project_q(attn_module, hidden_states):
    q = attn_module.q_proj(hidden_states)
    b, t, _ = hidden_states.shape
    n_heads = attn_module.config.num_attention_heads
    head_dim = getattr(attn_module, "head_dim", q.shape[-1] // n_heads)
    q = q.view(b, t, n_heads, head_dim).transpose(1, 2)
    if hasattr(attn_module, "q_norm"):
        q = attn_module.q_norm(q)
    return q.detach()


def install_tracer(model):
    """Returns a list of (module, original_forward) for remove_tracer."""
    saved = []
    for layer in model.model.layers:
        attn = layer.self_attn
        original, traced = _patch(attn)
        saved.append((attn, original))
        attn.forward = traced
        attn.attn_inputs = {}
    return saved


def remove_tracer(model, saved) -> None:
    for attn, original in saved:
        attn.forward = original
        if hasattr(attn, "attn_inputs"):
            del attn.attn_inputs
