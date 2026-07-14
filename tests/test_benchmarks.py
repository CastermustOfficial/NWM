"""Smoke tests for the benchmark harness (baselines + runner)."""

import numpy as np
import pytest
from benchmarks.baselines import RandomAgent, TabularQLearningAgent
from benchmarks.baselines.base import BenchmarkAgent
from benchmarks.config import ENVIRONMENTS
from benchmarks.metrics import summarize
from benchmarks.nwm_wrapper import NWMBenchmarkAgent
from benchmarks.runner import train_one_run


def _make_agents():
    agents = [
        RandomAgent(num_actions=2, seed=0),
        TabularQLearningAgent(num_actions=2, low=(-1, -1), high=(1, 1), bins=(4, 4), seed=0),
        NWMBenchmarkAgent(state_dim=2, num_actions=2, seed=0),
    ]
    return agents


def test_baselines_satisfy_protocol():
    for agent in _make_agents():
        assert isinstance(agent, BenchmarkAgent)
        assert isinstance(agent.name, str)


def test_baselines_act_and_observe():
    state = np.zeros(2, dtype=np.float32)
    next_state = np.ones(2, dtype=np.float32)
    for agent in _make_agents():
        action = agent.act(state)
        assert isinstance(action, int)
        agent.observe(state, action, 1.0, next_state, done=True)
        agent.end_episode()


def test_tabular_q_learns_positive_values():
    agent = TabularQLearningAgent(
        num_actions=2, low=(-1, -1), high=(1, 1), bins=(3, 3), alpha=0.5, seed=0
    )
    s = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(20):
        agent.observe(s, 1, reward=1.0, next_state=s, done=False)
    cell = agent._discretize(s)
    assert agent.q_table[cell][1] > agent.q_table[cell][0]


@pytest.mark.integration
def test_train_one_run_produces_curves():
    from benchmarks.config import build_random

    spec = ENVIRONMENTS["cartpole"]
    result = train_one_run(
        spec=spec,
        agent_label="Random",
        agent_builder=build_random,
        seed=0,
        eval_interval=5,
        eval_episodes=3,
        episodes_override=10,
    )
    assert len(result.train_rewards) == 10
    assert len(result.eval_means) == len(result.eval_points) >= 1
    summary = summarize([result])
    assert summary.agent == "Random"
    assert np.isfinite(summary.final_eval_mean)
