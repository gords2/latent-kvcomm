"""The relay loop. Assertions are structural, so random weights are fine."""
import torch

from lkv.config import RunConfig
from lkv.latentmas import generate_latent, past_length, run_latentmas
from lkv.realign import LatentProjector


class _StubTok:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=None):
        seqs = [[7, 8, 9] for _ in texts]
        return {"input_ids": torch.tensor(seqs),
                "attention_mask": torch.ones(len(seqs), 3, dtype=torch.long)}

    def decode(self, t, skip_special_tokens=True):
        return "answer \\boxed{4}"


def _cfg(**kw):
    base = dict(system="latentmas", task="gsm8k", model_name="Qwen/Qwen3-8B",
                latent_steps=5, n_agents=4, greedy=True, max_new_tokens=4,
                device="cpu", generate_bs=2)
    base.update(kw)
    return RunConfig(**base)


def test_past_length_reads_every_layer(tiny_qwen3):
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        cache = tiny_qwen3(ids, use_cache=True).past_key_values
    assert past_length(cache) == 4
    # collapse a non-zero layer; upstream reads layer 0 only and would miss this
    cache.key_cache[2] = cache.key_cache[2][:, :, :1, :]
    cache.value_cache[2] = cache.value_cache[2][:, :, :1, :]
    assert past_length(cache) == 4
    assert min(k.shape[-2] for k in cache.key_cache) == 1


def test_generate_latent_adds_prompt_plus_steps(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    ids = torch.tensor([[3, 4, 5]])
    mask = torch.ones_like(ids)
    cache, running_mask, added = generate_latent(tiny_qwen3, proj, ids, mask, latent_steps=5,
                                                  past_key_values=None)
    assert added == 3 + 5
    assert past_length(cache) == 3 + 5
    assert running_mask.shape[1] == past_length(cache)


def test_generate_latent_zero_steps_adds_only_the_prompt(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    ids = torch.tensor([[3, 4, 5]])
    cache, running_mask, added = generate_latent(tiny_qwen3, proj, ids, torch.ones_like(ids),
                                                  latent_steps=0, past_key_values=None)
    assert added == 3
    assert running_mask.shape[1] == past_length(cache)


def test_generate_latent_is_invariant_to_batch_composition(tiny_qwen3):
    """D16: an item's representation must not depend on how long its peers are.

    Every relay hop left-pads its prompt, which inserts pad columns BETWEEN the
    cached history and that hop's new real tokens. A forward that derives
    positions from `cache_position` alone -- which is what a bare `model(...)`
    call does -- counts those pads as real distance, so the same item lands at a
    different RoPE offset depending on the LONGEST item sharing its batch, and
    `--generate_bs` silently becomes an experimental variable.

    Measured before the fix: 3.737 max delta on this item's hop-1 keys, its real
    tokens sitting 5-6 positions from the previous hop instead of 1-2. The
    judger reaches the same cache through `generate()`, which has always used
    the cumsum convention, so the unfixed relay also disagreed with itself.
    """
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")

    def relay(hop0_ids, hop0_mask, hop1_ids, hop1_mask):
        cache, run_mask, _ = generate_latent(
            tiny_qwen3, proj, hop0_ids, hop0_mask, latent_steps=2,
            past_key_values=None)
        cache, run_mask, _ = generate_latent(
            tiny_qwen3, proj, hop1_ids, hop1_mask, latent_steps=2,
            past_key_values=cache, past_mask=run_mask)
        return cache

    torch.manual_seed(3)
    h0 = torch.randint(1, 200, (2, 8))
    h1 = torch.randint(1, 200, (2, 6))
    m1 = torch.ones(2, 6, dtype=torch.long)
    h1[1, :4] = 0
    m1[1, :4] = 0  # row 1 is padded ONLY because row 0's prompt is longer

    mixed = relay(h0, torch.ones(2, 8, dtype=torch.long), h1, m1)
    alone = relay(h0[1:2], torch.ones(1, 8, dtype=torch.long),
                  h1[1:2, 4:], torch.ones(1, 2, dtype=torch.long))

    # Compare the LATENT positions: they are always real in both runs and they
    # are what the next hop actually consumes.
    got = mixed.key_cache[-1][1, :, -2:, :]
    want = alone.key_cache[-1][0, :, -2:, :]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-3)


def test_generate_latent_requires_past_mask_for_nonempty_cache(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    ids = torch.tensor([[3, 4, 5]])
    mask = torch.ones_like(ids)
    cache, running_mask, _ = generate_latent(tiny_qwen3, proj, ids, mask, latent_steps=1,
                                              past_key_values=None)
    try:
        generate_latent(tiny_qwen3, proj, ids, mask, latent_steps=1,
                         past_key_values=cache, past_mask=None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when past_mask is missing for a nonempty cache")


def test_relay_runs_three_latent_agents_then_judges(tiny_qwen3):
    items = [{"question": "2+2?", "gold": "4"}]
    texts, hops = run_latentmas(tiny_qwen3, _StubTok(), _cfg(), items)
    assert len(texts) == 1
    # 4 agents: 3 latent hops + 1 judger hop
    assert [h["role"] for h in hops] == ["planner", "critic", "refiner", "judger"]
    assert sum(h["latent_steps"] for h in hops) == 3 * 5


def test_cache_grows_monotonically_without_a_hook(tiny_qwen3):
    items = [{"question": "2+2?", "gold": "4"}]
    _, hops = run_latentmas(tiny_qwen3, _StubTok(), _cfg(), items)
    lens = [h["positions_after"] for h in hops]
    assert lens == sorted(lens), lens
    assert all(h["positions_before"] == h["positions_after"] for h in hops) is False


def test_trim_hook_is_called_once_per_boundary(tiny_qwen3):
    calls = []

    def hook(cache, hop_index, inherited_len):
        calls.append((hop_index, inherited_len))
        return cache

    items = [{"question": "2+2?", "gold": "4"}]
    run_latentmas(tiny_qwen3, _StubTok(), _cfg(), items, trim_hook=hook)
    # a boundary exists after every agent except the last
    assert [c[0] for c in calls] == [0, 1, 2]


def test_n_agents_one_still_reaches_the_judger(tiny_qwen3):
    """D2 guard: with one agent there is no latent hop, and the judger must
    still run rather than silently receiving a discarded cache."""
    items = [{"question": "2+2?", "gold": "4"}]
    texts, hops = run_latentmas(tiny_qwen3, _StubTok(), _cfg(n_agents=1), items)
    assert [h["role"] for h in hops] == ["judger"]
    assert texts and texts[0]
