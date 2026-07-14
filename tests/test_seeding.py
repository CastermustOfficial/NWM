"""Tests for reproducibility helpers and per-agent RNG determinism."""

import numpy as np

from nwm import NWMAgent, NWMConfig, make_rng, set_global_seed


class TestSeedingHelpers:
    def test_make_rng_is_deterministic(self):
        a = make_rng(123).random(5)
        b = make_rng(123).random(5)
        assert np.array_equal(a, b)

    def test_make_rng_differs_across_seeds(self):
        a = make_rng(1).random(5)
        b = make_rng(2).random(5)
        assert not np.array_equal(a, b)

    def test_set_global_seed_makes_numpy_reproducible(self):
        set_global_seed(7)
        first = np.random.rand(4)
        set_global_seed(7)
        second = np.random.rand(4)
        assert np.array_equal(first, second)

    def test_make_rng_does_not_touch_global_state(self):
        set_global_seed(7)
        _ = make_rng(999).random(10)  # should not advance the global RNG
        after = np.random.rand(4)
        set_global_seed(7)
        expected = np.random.rand(4)
        assert np.array_equal(after, expected)


class TestAgentDeterminism:
    def _action_sequence(self, seed):
        agent = NWMAgent(state_dim=4, num_actions=2, seed=seed)
        rng = np.random.default_rng(0)
        actions = []
        for _ in range(200):
            state = rng.standard_normal(4).astype(np.float32)
            actions.append(agent.select_action(state, training=True))
        return actions

    def test_same_seed_same_actions(self):
        assert self._action_sequence(42) == self._action_sequence(42)

    def test_different_seed_different_actions(self):
        # Extremely unlikely to be identical across 200 exploratory draws.
        assert self._action_sequence(1) != self._action_sequence(2)

    def test_seed_argument_overrides_config(self):
        cfg = NWMConfig(seed=1)
        agent_arg = NWMAgent(state_dim=4, num_actions=2, config=cfg, seed=99)
        agent_ref = NWMAgent(state_dim=4, num_actions=2, seed=99)
        rng = np.random.default_rng(0)
        for _ in range(50):
            state = rng.standard_normal(4).astype(np.float32)
            assert agent_arg.select_action(state) == agent_ref.select_action(state)

    def test_config_seed_used_when_no_argument(self):
        a = NWMAgent(state_dim=4, num_actions=3, config=NWMConfig(seed=5))
        b = NWMAgent(state_dim=4, num_actions=3, config=NWMConfig(seed=5))
        rng = np.random.default_rng(3)
        for _ in range(50):
            state = rng.standard_normal(4).astype(np.float32)
            assert a.select_action(state) == b.select_action(state)
