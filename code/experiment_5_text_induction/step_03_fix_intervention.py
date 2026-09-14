"""Section 9.5, step 3: same calibrated direction and per-token dose in both texts."""

import math

from experiment_0_calibration.prepare_material import file_sha256
from experiment_0_calibration.protocol_config import repo_path


def fix_intervention(condition, scales):
    record = scales[condition["direction_id"]]
    if record["direction_family"] != "concept" or record["concept"] != condition["concept"]:
        raise ValueError("The injected direction must be the concept evoked by the variant")
    if record["decoder_block_index"] != condition["layer"]:
        raise ValueError("Intervention and calibrated direction layers differ")
    scale = float(record[condition["scale_statistic"]])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Natural scale must be finite and positive")
    source = repo_path(record["source_path"])
    if file_sha256(source) != record["source_sha256"]:
        raise ValueError("The concept vector changed since calibration")
    alpha = condition["dose"] * scale if condition["dose_axis"] == "z" else condition["dose"]
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid requested coefficient")
    # No text-dependent input is used in this conversion. Text has no z dose.
    return {**condition, "family": "concept", "natural_scale": scale, "alpha_requested": alpha}
