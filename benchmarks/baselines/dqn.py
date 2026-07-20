"""A compact Deep Q-Network baseline (PyTorch).

Standard DQN with an MLP Q-function, an experience replay buffer, a target
network updated on a fixed interval, and epsilon-greedy exploration. Kept
deliberately small so it trains on CPU within the benchmark budget.

Requires the ``torch`` extra: ``pip install -e ".[baselines]"``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DQNAgent requires PyTorch. Install it with: pip install -e '.[baselines]'"
    ) from exc


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network with replay buffer and a periodically synced target net."""

    name = "DQN"

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        min_buffer: int = 500,
        target_update: int = 250,
        train_freq: int = 1,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.985,
        seed: int | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_update = target_update
        self.train_freq = train_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if seed is not None:
            torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.device = torch.device("cpu")
        self.q = _QNetwork(state_dim, num_actions, hidden).to(self.device)
        self.target = _QNetwork(state_dim, num_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer: deque[tuple] = deque(maxlen=buffer_size)
        self._steps = 0
        self._grad_steps = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.num_actions))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.q(s.unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        terminated: bool | None = None,
    ) -> None:
        # Only a true terminal event zeroes the bootstrap target; a time-limit
        # truncation is not the end of the underlying process (standard fix).
        terminal = done if terminated is None else terminated
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                action,
                reward,
                np.asarray(next_state, dtype=np.float32),
                float(terminal),
            )
        )
        self._steps += 1
        # Update every ``train_freq`` environment steps (standard DQN practice:
        # far cheaper than a gradient step per step, with equivalent quality).
        if self._steps % self.train_freq == 0:
            self._learn()

    def _learn(self) -> None:
        if len(self.buffer) < self.min_buffer:
            return

        idx = self._rng.integers(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.q(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
