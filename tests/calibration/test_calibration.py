import numpy as np
import pytest

torch = pytest.importorskip("torch")

from calibration.compute_scales import robust_mad
from calibration.directions import unit


def test_unit_normalizes_without_changing_direction():
    result, original_norm = unit(torch.tensor([3.0, 4.0]))

    assert original_norm == pytest.approx(5.0)
    assert torch.linalg.vector_norm(result).item() == pytest.approx(1.0)
    assert result.tolist() == pytest.approx([0.6, 0.8])


def test_unit_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        unit(torch.zeros(3))


def test_robust_mad_ignores_a_single_large_outlier():
    clean = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    contaminated = np.array([-2.0, -1.0, 0.0, 1.0, 2000.0])

    assert robust_mad(contaminated) == pytest.approx(robust_mad(clean))
