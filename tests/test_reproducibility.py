"""End-to-end reproducibility tests: identical seeds yield identical training."""

import os
import tempfile

import gymnasium as gym
import numpy as np
import pytest

from nwm import NWMAgent, NWMConfig, set_global_seed


def _train_and_collect(seed, episodes=40):
    """Train an NWM agent with a fully seeded environment and return rewards."""
    set_global_seed(seed)
    env = gym.make("CartPole-v1")
    agent = NWMAgent(
        state_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=NWMConfig(warmup_episodes=10),
        seed=seed,
    )
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total += reward
        rewards.append(total)
    env.close()
    return rewards


@pytest.mark.integration
def test_training_is_reproducible():
    run_a = _train_and_collect(42)
    run_b = _train_and_collect(42)
    assert run_a == run_b


@pytest.mark.integration
def test_different_seeds_diverge():
    run_a = _train_and_collect(42)
    run_b = _train_and_collect(43)
    assert run_a != run_b


def test_save_load_preserves_rng_stream():
    """After save/load, the agent must continue the exact same action stream."""
    agent = NWMAgent(state_dim=4, num_actions=3, seed=7)
    rng = np.random.default_rng(0)
    warm = [rng.standard_normal(4).astype(np.float32) for _ in range(30)]
    for s in warm:
        agent.select_action(s)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        agent.save(path)
        reloaded = NWMAgent.load(path)
        probe = [rng.standard_normal(4).astype(np.float32) for _ in range(30)]
        original = [agent.select_action(s) for s in probe]
        restored = [reloaded.select_action(s) for s in probe]
        assert original == restored
    finally:
        os.unlink(path)
