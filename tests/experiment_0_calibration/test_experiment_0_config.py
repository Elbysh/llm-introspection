from experiment_0_calibration.protocol_config import load_config


def test_development_config_covers_every_decoder_block():
    config = load_config("configs/experiment_0_calibration/development_full.yaml")

    assert config.layers == list(range(32))
    assert config.activation_site == "decoder_block_output"
    assert config.hidden_state_offset == 1
    assert config.protocol_status == "development"


def test_development_config_makes_weighting_and_bootstrap_explicit():
    config = load_config("configs/experiment_0_calibration/development_full.yaml")

    assert config.position_policy == "all_sentence_tokens"
    assert config.point_weighting == "equal_token"
    assert config.bootstrap_unit == "sentence"
    assert config.bootstrap_resamples_sd > config.bootstrap_resamples_mad > 0


def test_stochastic_families_have_separate_seed_namespaces():
    config = load_config("configs/experiment_0_calibration/development_full.yaml")

    # The fixed-random interval must end before the renewed-noise interval
    # starts. The material-plan builder also checks every realized seed.
    fixed_random_upper_bound = (
        config.fixed_random_base_seed
        + len(config.layers) * config.fixed_random_count_per_layer
    )
    assert fixed_random_upper_bound < config.renewed_noise_base_seed
