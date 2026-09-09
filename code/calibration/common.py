"""Shared helpers for the standalone calibration pipeline."""

import importlib.util
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str) -> Path:
    """Resolve configuration paths consistently from the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_repository_corpus() -> List[str]:
    """Read LOCALIZATION_SENTENCES without modifying or duplicating it."""
    source = REPO_ROOT / "code" / "utils" / "all_prompts.py"
    spec = importlib.util.spec_from_file_location("existing_all_prompts", str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repository sentence corpus")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.LOCALIZATION_SENTENCES)


def disable_optional_triton_native_ops() -> bool:
    """Use standard CUDA kernels if optional PyTorch Triton JIT cannot compile.

    This is needed on the current Ruche image, where the optional Triton host
    extension fails to link. Disabling it does not change the mathematical
    operation being evaluated.
    """
    # Keep torch lazy so plotting existing results does not require a model
    # environment or a GPU installation.
    import torch

    python_native = getattr(torch.backends, "python_native", None)
    triton_backend = getattr(python_native, "triton", None)
    if triton_backend is None:
        return False
    triton_backend.enabled = False
    return True
