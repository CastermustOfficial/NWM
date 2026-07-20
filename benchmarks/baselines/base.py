"""Common agent interface used by the benchmark runner.

Every agent (NWM and baselines) exposes the same three-method contract so the
training loop is identical across methods and comparisons stay fair.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BenchmarkAgent(Protocol):
    """Minimal interface a benchmarkable agent must implement."""

    #: Human-readable short name used in tables and plots.
    name: str

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Return an action for ``state``."""
        ...

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        terminated: bool | None = None,
    ) -> None:
        """Consume one environment transition (learning happens here).

        ``terminated`` distinguishes a true terminal event from a time-limit
        truncation; value-based agents must only zero the bootstrap target on
        true termination. ``None`` preserves the legacy behavior (``done``).
        """
        ...

    def end_episode(self) -> None:
        """Hook called once per episode after the terminal transition."""
        ...
