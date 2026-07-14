"""Tabular Q-learning with uniform state discretization.

A classic value-based baseline. Continuous observations are mapped to a
discrete grid (state aggregation); each cell keeps an independent action-value
row updated with the standard Q-learning TD rule::

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
"""

from __future__ import annotations

import numpy as np


class TabularQLearningAgent:
    """Epsilon-greedy tabular Q-learning over a discretized observation space."""

    name = "TabularQ"

    def __init__(
        self,
        num_actions: int,
        low: np.ndarray,
        high: np.ndarray,
        bins: tuple[int, ...],
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.bins = tuple(bins)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)

        # Q-table shape: (bins[0], ..., bins[d-1], num_actions)
        self.q_table = np.zeros((*self.bins, num_actions), dtype=np.float64)
        self._span = self.high - self.low

    def _discretize(self, state: np.ndarray) -> tuple[int, ...]:
        state = np.asarray(state, dtype=np.float64)
        ratios = (state - self.low) / self._span
        ratios = np.clip(ratios, 0.0, 0.999999)
        idx = (ratios * np.asarray(self.bins)).astype(int)
        return tuple(int(i) for i in idx)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.num_actions))
        cell = self._discretize(state)
        return int(np.argmax(self.q_table[cell]))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        s = self._discretize(state)
        s_next = self._discretize(next_state)
        best_next = 0.0 if done else float(np.max(self.q_table[s_next]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.alpha * td_error

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
