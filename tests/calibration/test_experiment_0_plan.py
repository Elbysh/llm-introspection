import pytest

torch = pytest.importorskip("torch")

from calibration.experiment_0.plan import unit


def test_direction_normalization_preserves_original_norm():
    normalized, original_norm = unit(torch.tensor([3.0, 4.0]))

    assert original_norm == pytest.approx(5.0)
    assert normalized.tolist() == pytest.approx([0.6, 0.8])


def test_zero_direction_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        unit(torch.zeros(4))

