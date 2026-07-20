"""Tests for the 2.4.0 features: EMA memory, dynamic unlock, truncation-aware
credit, eval-time sticky exploration, and truncation-aware baseline bootstrap."""

from __future__ import annotations

import numpy as np

from nwm import NWMAgent, NWMConfig
from nwm.core.centroid import PersistentCentroid


def _state(x: float) -> np.ndarray:
    return np.array([x, x], dtype=np.float32)


class TestScoreEma:
    def test_ema_tracks_recent_scores(self) -> None:
        legacy = PersistentCentroid(_state(0.0), 0.9, action=0)
        ema = PersistentCentroid(_state(0.0), 0.9, action=0, ema_alpha=0.3)
        for _ in range(5):
            legacy.merge(_state(0.0), 0.9, 0)
            ema.merge(_state(0.0), 0.9, 0)
        for _ in range(20):
            legacy.merge(_state(0.0), 0.1, 0)
            ema.merge(_state(0.0), 0.1, 0)
        s, n, _ = legacy.action_votes[0]
        legacy_mean = s / n
        ema_mean = ema.action_votes[0][0]
        # The EMA forgets the stale 0.9s much faster than the running average.
        assert ema_mean < legacy_mean
        assert ema_mean < 0.15

    def test_ema_force_uses_mean_directly(self) -> None:
        ema = PersistentCentroid(_state(0.0), 0.9, action=0, ema_alpha=0.3)
        assert ema.get_force(0) > 0  # attraction for a high score


class TestDynamicUnlock:
    def _locked_centroid(self) -> PersistentCentroid:
        c = PersistentCentroid(_state(0.0), 1.0, action=0)
        for _ in range(10):
            c.merge(_state(0.0), 1.0, 0)
        assert c.locked
        return c

    def test_lock_is_permanent_by_default(self) -> None:
        c = self._locked_centroid()
        for _ in range(50):
            c.merge(_state(0.0), 0.0, 0)
        assert c.locked

    def test_unlocks_when_recent_scores_collapse(self) -> None:
        c = self._locked_centroid()
        for _ in range(50):
            c.merge(_state(0.0), 0.0, 0, dynamic_unlock=True)
        assert not c.locked


class TestTruncationCredit:
    def _run_episode(self, config: NWMConfig, terminated: bool) -> NWMAgent:
        agent = NWMAgent(state_dim=2, num_actions=2, config=config, seed=0)
        for _ in range(agent.config.warmup_episodes):
            agent.step(_state(0.0), 0, 1.0, _state(0.0), done=True, terminated=True)
        for i in range(9):
            agent.step(_state(float(i)), 0, 1.0, _state(float(i + 1)), done=False)
        agent.step(_state(9.0), 0, 1.0, _state(10.0), done=True, terminated=terminated)
        return agent

    def test_terminated_path_matches_legacy(self) -> None:
        base = NWMConfig(warmup_episodes=1, seed=0)
        flag = NWMConfig(warmup_episodes=1, truncation_credit=True, seed=0)
        a = self._run_episode(base, terminated=True)
        b = self._run_episode(flag, terminated=True)
        scores_a = [c.score_sum for c in a.field.centroids]
        scores_b = [c.score_sum for c in b.field.centroids]
        assert scores_a == scores_b

    def test_truncated_episode_changes_credit(self) -> None:
        base = NWMConfig(warmup_episodes=1, seed=0)
        flag = NWMConfig(warmup_episodes=1, truncation_credit=True, seed=0)
        a = self._run_episode(base, terminated=False)
        b = self._run_episode(flag, terminated=False)
        scores_a = [c.score_sum for c in a.field.centroids]
        scores_b = [c.score_sum for c in b.field.centroids]
        assert scores_a != scores_b


class TestEvalSticky:
    def test_eval_fallback_repeats_action_when_enabled(self) -> None:
        config = NWMConfig(exploration_repeat=1.0, eval_sticky=True, seed=0)
        agent = NWMAgent(state_dim=2, num_actions=10, config=config, seed=0)
        actions = [agent.select_action(_state(0.0), training=False) for _ in range(10)]
        assert len(set(actions)) == 1  # empty field -> sticky fallback repeats

    def test_eval_fallback_uniform_by_default(self) -> None:
        config = NWMConfig(exploration_repeat=1.0, seed=0)
        agent = NWMAgent(state_dim=2, num_actions=10, config=config, seed=0)
        actions = [agent.select_action(_state(0.0), training=False) for _ in range(10)]
        assert len(set(actions)) > 1


class TestBaselineBootstrap:
    def test_tabular_q_bootstraps_on_truncation(self) -> None:
        from benchmarks.baselines import TabularQLearningAgent

        def make() -> TabularQLearningAgent:
            agent = TabularQLearningAgent(
                num_actions=2, low=(0.0, 0.0), high=(1.0, 1.0), bins=(2, 2), seed=0
            )
            agent.q_table[:] = 1.0  # non-zero next-state values
            return agent

        s, s2 = np.array([0.1, 0.1]), np.array([0.9, 0.9])
        terminal = make()
        terminal.observe(s, 0, 0.0, s2, done=True, terminated=True)
        truncated = make()
        truncated.observe(s, 0, 0.0, s2, done=True, terminated=False)
        cell = terminal._discretize(s)
        # Truncation must keep the bootstrap term; termination must zero it.
        assert truncated.q_table[cell][0] > terminal.q_table[cell][0]


class TestCreditPropagation:
    def test_value_flows_backwards_through_graph(self) -> None:
        from nwm.core.potential_field import PersistentPotentialField

        field = PersistentPotentialField(state_dim=2, credit_propagation=0.5)
        # Centroid A: own episodes failed (low score). Centroid B: high score.
        a_uid = field.add(np.array([0.0, 0.0], dtype=np.float32), 0, 0.2)
        b_uid = field.add(np.array([5.0, 5.0], dtype=np.float32), 0, 0.9)
        field.record_transition(a_uid, 0, b_uid)
        field.propagate()
        succ = field._successor_value(a_uid, 0)
        assert succ is not None and succ > 0.6  # B's value visible from A
        # The blended force at A turns less repulsive than A's own score says.
        force_prop = field.centroids[0].get_force(0, succ, 0.5)
        force_raw = field.centroids[0].get_force(0)
        assert force_prop > force_raw

    def test_self_loops_are_ignored(self) -> None:
        from nwm.core.potential_field import PersistentPotentialField

        field = PersistentPotentialField(state_dim=2, credit_propagation=0.5)
        uid = field.add(np.array([0.0, 0.0], dtype=np.float32), 0, 0.2)
        field.record_transition(uid, 0, uid)
        assert field.succ == {}

    def test_disabled_by_default(self) -> None:
        from nwm.core.potential_field import PersistentPotentialField

        field = PersistentPotentialField(state_dim=2)
        field.add(np.array([0.0, 0.0], dtype=np.float32), 0, 0.2)
        field.propagate()
        assert field._prop_values == {}


class TestStagnationRevival:
    def test_checkpoint_and_restore_roundtrip(self) -> None:
        agent = NWMAgent(state_dim=2, num_actions=2, config=NWMConfig(warmup_episodes=0), seed=0)
        agent.field.add(_state(0.0), 0, 0.9)
        agent.best_reward = 10.0
        agent.checkpoint_if_best(10.0)
        agent.field.add(_state(3.0), 1, 0.1)
        assert len(agent.field) == 2
        agent.restore_best_model()
        assert len(agent.field) == 1

    def test_collapse_triggers_restore_and_exploration(self) -> None:
        config = NWMConfig(stagnation_revival=True, warmup_episodes=0)
        agent = NWMAgent(state_dim=2, num_actions=2, config=config, seed=0)
        agent._recent_rewards.extend([100.0] * 20)
        agent._maybe_revive(100.0)  # new best -> snapshot
        assert agent._field_snapshot is not None
        agent.field.add(_state(0.0), 0, 0.1)  # divergence after snapshot
        agent.exploration_rate = 0.01
        for _ in range(30):
            agent._maybe_revive(10.0)  # sustained collapse far below best
        assert agent.exploration_rate >= 0.3
        assert len(agent.field) == 0  # restored to the (empty) best snapshot
