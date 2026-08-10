"""KVComm trimming at LatentMAS agent boundaries.

Two scopes, because KVComm's trim was defined for a single hop where the
sender's cache IS the message, while LatentMAS's cache is a growing history
carrying every earlier agent's re-supplied question plus every earlier
agent's latent thoughts, across three boundaries:

  all    unselected layers keep `sink` positions of the WHOLE cache. Faithful
         to stock-kvcomm/models.py:95-108 (lkv/kvcomm/trim.py:prepare_key_cache),
         and far more aggressive here than anything KVComm measured: applied at
         every boundary it would collapse the relay's entire shared history, not
         just one hop's message.
  delta  unselected layers keep everything inherited from earlier hops
         (`[:inherited_len]`) plus `sink` positions of the segment THIS agent
         added -- trims the message, not the history. Only meaningful with more
         than one hop: at inherited_len=0 it degenerates to `all` by
         construction (see test_trim_delta_at_first_hop_equals_scope_all).

Layer 0 is always retained in full in both scopes, matching upstream
(stock-kvcomm/models.py:101, `if i in self.layers_list or i == 0`) and
prepare_key_cache. That is also why run_latentmas's `run_mask[:, :past_length(cache)]`
re-cut after a trim is safe: past_length is the MAX over all layers, layer 0
never shrinks, so the max -- and therefore the mask width -- never shrinks
either. Both scopes keep a PREFIX of every layer for exactly this reason: a
non-prefix subset would desynchronise the re-cut mask from cache_position.
"""
from __future__ import annotations

from transformers import DynamicCache

from lkv.kvcomm.trim import prepare_key_cache, retention


def trim_delta(cache, layers_list, inherited_len: int, sink: int = 1):
    """Unselected layers keep `[:inherited_len]` plus `sink` new positions.

    inherited_len is the cache length BEFORE this hop's agent added anything
    (run_latentmas's own `before = past_length(cache)`, taken prior to
    generate_latent) -- so the kept prefix is exactly the history this hop
    inherited from earlier boundaries, plus a `sink`-sized peek at what this
    agent just contributed. Layer 0 is always retained in full, never a
    candidate for collapse in either scope.
    """
    keep = set(int(i) for i in layers_list)
    out = DynamicCache()
    for i in range(len(cache.key_cache)):
        k, v = cache.key_cache[i], cache.value_cache[i]
        if i in keep or i == 0:
            out.update(k, v, i)
        else:
            cut = inherited_len + sink
            out.update(k[:, :, :cut, :], v[:, :, :cut, :], i)
    return out


def make_trim_hook(layers_list, scope: str = "all", sink: int = 1,
                   every_hop: bool = False, recalibrate=None):
    """Build a trim_hook for run_latentmas: `trim_hook(cache, hop_index,
    inherited_len) -> cache`.

    every_hop=False trims only the FIRST time this hook is ever CALLED, and
    passes every later call through untouched -- that isolates the effect of
    trimming once from the effect of compounding at every boundary, which
    Task 20 needs as two distinct conditions. The gate is call history
    (`hook.records` being empty), not `hop_index == 0`: run_latentmas always
    invokes hooks in increasing hop_index order starting at 0, so in real
    relay use the two coincide, but a caller probing the hook directly (as
    test_scope_all_is_more_aggressive_than_delta does, at an arbitrary
    hop_index) still gets "the first call trims."

    hook.records accumulates exactly one entry per CALL, whether or not that
    call actually trimmed -- it is the evidence trail Task 19 reads, and a
    pass-through hop (retention 1.0) is as real a fact about the run as a
    trimmed one, so it is recorded rather than silently dropped.

    recalibrate(cache, hop_index) -> layers lets Task 18 re-rank the kept
    layers per hop instead of reusing the same layers_list at every boundary.
    """
    if scope not in ("all", "delta"):
        raise ValueError(f"scope must be 'all' or 'delta', got {scope!r}")

    def hook(cache, hop_index: int, inherited_len: int):
        do_trim = every_hop or not hook.records
        layers = recalibrate(cache, hop_index) if recalibrate else layers_list
        before = max(int(k.shape[-2]) for k in cache.key_cache)
        if do_trim:
            if scope == "all":
                out = prepare_key_cache(cache, layers, sink=sink)
            else:
                out = trim_delta(cache, layers, inherited_len, sink=sink)
        else:
            out = cache
        hook.records.append({
            "hop": hop_index,
            "kept_layers": sorted(int(i) for i in layers),
            "positions_before": before,
            "positions_after": max(int(k.shape[-2]) for k in out.key_cache),
            "retention": retention(out),
        })
        return out

    hook.records = []
    return hook
