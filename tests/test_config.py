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


def test_ungated_llama_mirror_is_accepted_and_resolves_to_the_llama_alpha():
    """The meta-llama repos are gated behind a Hugging Face account; the
    NousResearch mirror is not. KVComm's published numbers are Llama numbers, so
    K1/K2 need a Llama -- this keeps them reachable without an account.

    alpha MUST land on 1.0. default_alpha matches a substring of the model id,
    and this id contains "llama" only inside "Meta-Llama-3.1". If that match
    ever broke, the run would fall through to the 1.0 default and still look
    correct here -- the Qwen case below is what keeps this from being vacuous.
    """
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "NousResearch/Meta-Llama-3.1-8B-Instruct",
                      "--top_layers", "0.3"])
    assert cfg.alpha == 1.0
    assert cfg.model_b == cfg.model_name


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


def test_max_new_tokens_resolves_per_task_when_unspecified():
    """GPU accuracy gates depend on per-task budgets; wrong budget -> silent accuracy drop."""
    # countries task defaults to 5
    cfg_countries = parse_args(["--system", "kvcomm", "--task", "countries",
                                 "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
                                 "--top_layers", "0.3"])
    assert cfg_countries.max_new_tokens == 5

    # gsm8k task defaults to 2048
    cfg_gsm8k = parse_args(["--system", "latentmas", "--task", "gsm8k",
                             "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40"])
    assert cfg_gsm8k.max_new_tokens == 2048


def test_explicit_max_new_tokens_passes_through_unchanged():
    """An explicit --max_new_tokens should not be overwritten by task defaults."""
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40",
                      "--max_new_tokens", "1024"])
    assert cfg.max_new_tokens == 1024


def test_no_renorm_latent_flag_disables_renormalization():
    """The flag that turns off latent renormalization must be exercisable."""
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
                      "--top_layers", "0.3", "--no_renorm_latent"])
    assert cfg.renorm_latent is False


def test_calib_source_pair_rejects_tasks_without_instruction_context():
    """Finding 3: --calib_source pair calibrates via
    kvcomm_sender_prompt(instruction, context) (lkv/kvcomm/pair.py), but
    gsm8k/mbppplus items carry only question/gold (lkv/data/gsm8k.py,
    lkv/data/mbppplus.py) -- a KeyError at run time, not argument-parse time.
    The sweep in the plan schedules exactly this combination with
    TASK=mbppplus; catching it here means the run dies loudly up front
    instead of the sweep script (set -uo pipefail, no -e) silently dropping
    the control from the results table.
    """
    for task in ("gsm8k", "mbppplus"):
        with pytest.raises(SystemExit):
            parse_args(["--system", "latentmas", "--task", task,
                        "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40",
                        "--top_layers", "0.5", "--calib_source", "pair"])

    # The tasks that DO carry instruction/context must still be accepted.
    for task in ("hotpotqa", "countries"):
        cfg = parse_args(["--system", "latentmas", "--task", task,
                          "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40",
                          "--top_layers", "0.5", "--calib_source", "pair"])
        assert cfg.calib_source == "pair"

    # --calib_source relay (the default) must remain unaffected on either task.
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40",
                      "--top_layers", "0.5"])
    assert cfg.calib_source == "relay"


def test_n_agents_over_the_defined_roles_is_rejected():
    """Fix-now item: at n_agents=5 the run used to die deep inside
    lkv/decode.py with an opaque `ValueError: past_mask has shape (1, 15),
    expected (1, 20)` -- correct behaviour (run_latentmas would reuse
    'judger' twice, per lkv/prompts.py's 4-role ROLES tuple), but a confusing
    way to learn you asked for five roles when only four are defined.
    """
    with pytest.raises(SystemExit):
        parse_args(["--system", "latentmas", "--task", "gsm8k",
                    "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40",
                    "--n_agents", "5"])

    # 4 (the default, and the max) must still be accepted.
    cfg = parse_args(["--system", "latentmas", "--task", "gsm8k",
                      "--model_name", "Qwen/Qwen3-8B", "--latent_steps", "40",
                      "--n_agents", "4"])
    assert cfg.n_agents == 4


def test_realign_and_no_renorm_latent_compose():
    """The four combinations of realign and renorm_latent are the ablation grid Task 5 sweeps."""
    cfg = parse_args(["--system", "kvcomm", "--task", "countries",
                      "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
                      "--top_layers", "0.3",
                      "--realign", "--no_renorm_latent"])
    assert cfg.realign is True
    assert cfg.renorm_latent is False
