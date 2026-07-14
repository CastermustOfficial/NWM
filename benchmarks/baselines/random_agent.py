"""Uniform random policy - the trivial lower-bound baseline."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Selects actions uniformly at random. Does not learn."""

    name = "Random"

    def __init__(self, num_actions: int, seed: int | None = None) -> None:
        self.num_actions = num_actions
        self._rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        return int(self._rng.integers(0, self.num_actions))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        return None

    def end_episode(self) -> None:
        return None
