"""
Reproducibility utilities for NWM.

NWM's own stochasticity (exploration, random action fallbacks) is driven by a
private :class:`numpy.random.Generator` owned by each agent, so two agents
constructed with the same ``seed`` behave identically regardless of global RNG
state. :func:`set_global_seed` additionally pins the process-wide RNGs (Python,
NumPy, and PyTorch when installed) which is useful for seeding the *environment*
and any baseline that relies on global state.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["make_rng", "set_global_seed"]


def make_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create an independent NumPy random generator.

    Parameters
    ----------
    seed : int | None
        Seed for the generator. ``None`` draws fresh entropy from the OS.

    Returns
    -------
    numpy.random.Generator
        A PCG64-backed generator that does not touch global RNG state.
    """
    return np.random.default_rng(seed)


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """
    Seed all process-wide random number generators.

    Seeds the :mod:`random` module, NumPy's legacy global RNG, ``PYTHONHASHSEED``
    and, if PyTorch is installed, its CPU/CUDA generators. Use this to make an
    entire experiment (environment resets included) reproducible.

    Parameters
    ----------
    seed : int
        The seed value.
    deterministic_torch : bool
        If True and PyTorch is available, request deterministic algorithms
        (``torch.use_deterministic_algorithms``). May slow training and raise
        for ops without a deterministic implementation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Intentionally seed the legacy global RNG so baselines/environments that
    # rely on it are reproducible; NWM itself uses a private Generator.
    np.random.seed(seed)  # noqa: NPY002

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
