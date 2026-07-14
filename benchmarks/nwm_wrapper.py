"""Adapter exposing :class:`nwm.NWMAgent` through the benchmark interface."""

from __future__ import annotations

import numpy as np

from nwm import NWMAgent, NWMConfig


class NWMBenchmarkAgent:
    """Wrap ``NWMAgent`` so it satisfies the common ``BenchmarkAgent`` contract.

    ``NWMAgent.step`` already performs end-of-episode learning when ``done`` is
    True, so :meth:`observe` forwards directly to it and :meth:`end_episode` is a
    no-op.
    """

    name = "NWM"

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        config: NWMConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.agent = NWMAgent(
            state_dim=state_dim, num_actions=num_actions, config=config, seed=seed
        )

    def act(self, state: np.ndarray, training: bool = True) -> int:
        return self.agent.select_action(state, training=training)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.agent.step(state, action, reward, next_state, done)

    def end_episode(self) -> None:
        return None
