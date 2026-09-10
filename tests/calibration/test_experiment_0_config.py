from calibration.experiment_0.config import load_config


def test_development_config_covers_every_decoder_block():
    config = load_config("configs/calibration/experiment_0/development_full.yaml")

    assert config.layers == list(range(32))
    assert config.activation_site == "decoder_block_output"
    assert config.hidden_state_offset == 1
    assert config.protocol_status == "development"


def test_development_config_makes_weighting_and_bootstrap_explicit():
    config = load_config("configs/calibration/experiment_0/development_full.yaml")

    assert config.position_policy == "all_sentence_tokens"
    assert config.point_weighting == "equal_token"
    assert config.bootstrap_unit == "sentence"
    assert config.bootstrap_resamples_sd > config.bootstrap_resamples_mad > 0

