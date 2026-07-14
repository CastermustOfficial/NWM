"""Baseline agents implementing the common :class:`BenchmarkAgent` interface."""

from benchmarks.baselines.base import BenchmarkAgent
from benchmarks.baselines.q_learning import TabularQLearningAgent
from benchmarks.baselines.random_agent import RandomAgent

__all__ = [
    "BenchmarkAgent",
    "RandomAgent",
    "TabularQLearningAgent",
]

try:  # DQN is optional (requires torch)
    from benchmarks.baselines.dqn import DQNAgent  # noqa: F401

    __all__.append("DQNAgent")
except ImportError:  # pragma: no cover
    pass
