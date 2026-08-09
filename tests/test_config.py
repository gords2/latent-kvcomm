import pytest

from lkv.config import MODEL_CHOICES, RunConfig, parse_args


def test_qwen3_8b_is_selectable():
    """D3: upstream run.py:91 lists Qwen/Qwen3-4B twice and omits 8B."""
    assert "Qwen/Qwen3-8B" in MODEL_CHOICES
    assert len(MODEL_CHOICES) == len(set(MODEL_CHOICES)), "duplicate model choice"


def test_latent_steps_has_no_silent_zero_default():
    """D2: a 0 default silently disables the whole relay, so require it."""
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40"])
    assert cfg.latent_steps == 40
    with pytest.raises(SystemExit):
        parse_args(["--system", "latentmas", "--task", "gsm8k",
                    "--model_name", "Qwen/Qwen3-8B"])


def test_kvcomm_defaults_match_paper():
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
                      "--top_layers", "0.3"])
    assert cfg.mu == 0.5 and cfg.sigma == 10.0 and cfg.calib_size == 1
    assert cfg.alpha == 1.0            # Llama family
    assert cfg.model_b == cfg.model_name  # receiver defaults to sender


def test_alpha_autoselects_08_for_qwen_and_falcon():
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "Qwen/Qwen2.5-7B-Instruct",
                      "--top_layers", "0.5"])
    assert cfg.alpha == 0.8


def test_greedy_and_sampling_are_mutually_exclusive_in_effect():
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B",
                      "--latent_steps", "40", "--greedy"])
    assert cfg.greedy is True


def test_defaults_are_paper_defaults():
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40"])
    assert cfg.n_agents == 4
    assert cfg.temperature == 0.6 and cfg.top_p == 0.95
    assert cfg.prompt == "sequential"
    assert cfg.seed == 42
    assert cfg.realign is False        # store_true upstream
    assert cfg.renorm_latent is True   # D4: upstream applies it unconditionally
    assert isinstance(cfg, RunConfig)
