#!/usr/bin/env python3
"""
Forward hooks applying dropout / Gaussian noise to Attention and MLP output vectors.

Follows the protocol: for each token of the target sentence and each layer, modify the
Attention and MLP output vectors by either
  (i)  dropout: zero each entry independently with rate p, rescale by 1/(1-p), or
  (ii) additive Gaussian noise with mean 0 and standard deviation sigma.

Both are stateless and resampled per token at every forward pass, so unlike the fixed
directions in save_random_vectors.py there is nothing to precompute or save -- only p or sigma
plus a seed.

    from gaussian_dropout_hooks import apply_perturbation
    with apply_perturbation(model, layer, [(sent1_range, 0.1)], kind="dropout", seed=trial):
        outputs = model(**encoding, use_cache=True)
"""

from contextlib import contextmanager

import torch

from save_random_vectors import derive_seed

SITES = ("self_attn", "mlp")
KINDS = ("dropout", "gaussian")


def _split_output(output):
    """Sublayer output is a tensor (mlp) or a tuple whose first entry is the tensor (self_attn)."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rebuild_output(tensor, rest):
    return tensor if rest is None else (tensor,) + rest


def _perturb_slice(output, target_range, fn):
    """Apply fn to the target token range of a sublayer output, leaving other tokens untouched."""
    tensor, rest = _split_output(output)
    start, end = target_range
    end = min(end, tensor.shape[1])
    if start >= end:
        return output
    modified = tensor.clone()
    modified[:, start:end, :] = fn(tensor[:, start:end, :])
    return _rebuild_output(modified, rest)


def make_dropout_hook(p, target_range, seed=None):
    """Hook zeroing each entry with probability p, then rescaling by 1/(1-p)."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout rate p must be in [0, 1), got {p}")
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    def fn(x):
        keep = (torch.rand(x.shape, generator=gen, dtype=torch.float32) >= p).to(
            device=x.device, dtype=x.dtype)
        return x * keep / (1.0 - p)

    def hook_fn(module, inputs, output):
        return _perturb_slice(output, target_range, fn)

    return hook_fn


def make_gaussian_noise_hook(sigma, target_range, seed=None, sigma_mode="absolute"):
    """
    Hook adding N(0, sigma^2) noise to each entry.

    sigma_mode="absolute" uses sigma directly (protocol default). "relative" multiplies it
    by the per-token RMS of the sublayer output, since attention and MLP output scales vary
    by more than an order of magnitude across layers.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma_mode not in ("absolute", "relative"):
        raise ValueError(f"unknown sigma_mode {sigma_mode!r}")
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    def fn(x):
        noise = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(
            device=x.device, dtype=x.dtype)
        if sigma_mode == "relative":
            rms = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True).sqrt().to(x.dtype)
            return x + sigma * rms * noise
        return x + sigma * noise

    def hook_fn(module, inputs, output):
        return _perturb_slice(output, target_range, fn)

    return hook_fn


def register_perturbation_hooks(model, layer, target_range, kind, p=0.1, sigma=1.0,
                                seed=0, sites=SITES, sigma_mode="absolute"):
    """Attach a perturbation hook to each requested sublayer of one layer; returns handles."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")

    handles = []
    for site in sites:
        submodule = getattr(model.model.layers[layer], site)
        site_seed = derive_seed(seed, kind, layer, site)
        if kind == "dropout":
            hook = make_dropout_hook(p, target_range, seed=site_seed)
        else:
            hook = make_gaussian_noise_hook(sigma, target_range, seed=site_seed,
                                            sigma_mode=sigma_mode)
        handles.append(submodule.register_forward_hook(hook))
    return handles


def make_vector_injection_hook(vector, targets):
    """Additive steering-vector hook; targets is a sequence of ((start, end), coeff)."""
    def hook_fn(module, inputs, output):
        tensor, rest = _split_output(output)
        steer = vector.to(device=tensor.device, dtype=tensor.dtype).reshape(1, 1, -1)
        modified = tensor.clone()
        for (start, end), coeff in targets:
            end = min(end, tensor.shape[1])
            if start < end:
                modified[:, start:end, :] += coeff * steer
        return _rebuild_output(modified, rest)

    return hook_fn


@contextmanager
def apply_perturbation(model, layer, targets, kind, vector=None, seed=0,
                       sites=SITES, sigma_mode="absolute"):
    """
    One entry point for every family; removes its hooks on exit.

    targets: sequence of ((start, end), level). `level` is the injection coefficient for
    concept/random, the dropout rate p for dropout, and sigma for gaussian.

    Note the site asymmetry inherited from the protocol: concept/random are added to the
    residual stream (the decoder layer output, as the existing experiments do), while
    dropout/gaussian perturb the Attention and MLP outputs.
    """
    if kind in ("concept", "random"):
        if vector is None:
            raise ValueError(f"kind={kind!r} requires a vector")
        handles = [model.model.layers[layer].register_forward_hook(
            make_vector_injection_hook(vector, targets))]
    elif kind in KINDS:
        handles = []
        for i, (token_range, level) in enumerate(targets):
            handles += register_perturbation_hooks(
                model, layer, token_range, kind,
                p=level if kind == "dropout" else 0.1,
                sigma=level if kind == "gaussian" else 1.0,
                seed=derive_seed(seed, "target", i), sites=sites, sigma_mode=sigma_mode)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()
