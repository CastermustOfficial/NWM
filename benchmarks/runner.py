"""Training / evaluation loop shared by every agent in the benchmark."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import gymnasium as gym
import numpy as np

from benchmarks.config import EnvSpec
from nwm import set_global_seed


@dataclass
class RunResult:
    """Outcome of training one agent on one environment with one seed."""

    env: str
    agent: str
    seed: int
    episodes: int
    train_rewards: list[float] = field(default_factory=list)
    eval_points: list[int] = field(default_factory=list)
    eval_means: list[float] = field(default_factory=list)
    eval_stds: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def final_eval(self) -> float:
        return self.eval_means[-1] if self.eval_means else float("nan")

    @property
    def best_eval(self) -> float:
        return max(self.eval_means) if self.eval_means else float("nan")

    @property
    def mean_train_reward(self) -> float:
        return float(np.mean(self.train_rewards)) if self.train_rewards else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final_eval"] = self.final_eval
        d["best_eval"] = self.best_eval
        d["mean_train_reward"] = self.mean_train_reward
        return d


def evaluate(agent, gym_id: str, episodes: int, base_seed: int) -> tuple[float, float]:
    """Run greedy (non-training) rollouts and return (mean, std) of returns."""
    env = gym.make(gym_id)
    returns = []
    for i in range(episodes):
        state, _ = env.reset(seed=base_seed + i)
        done = False
        total = 0.0
        while not done:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def train_one_run(
    spec: EnvSpec,
    agent_label: str,
    agent_builder,
    seed: int,
    eval_interval: int,
    eval_episodes: int,
    episodes_override: int | None = None,
) -> RunResult:
    """Train a single agent instance and record learning + evaluation curves."""
    set_global_seed(seed)
    episodes = episodes_override or spec.episodes

    env = gym.make(spec.gym_id)
    state_dim = int(np.asarray(env.observation_space.shape).prod())
    num_actions = int(env.action_space.n)
    agent = agent_builder(spec, state_dim, num_actions, seed)

    result = RunResult(env=spec.name, agent=agent_label, seed=seed, episodes=episodes)
    start = time.perf_counter()

    for ep in range(episodes):
        state, _ = env.reset(seed=seed * 100_000 + ep)
        done = False
        total = 0.0
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.observe(state, action, float(reward), next_state, done)
            state = next_state
            total += float(reward)
        agent.end_episode()
        result.train_rewards.append(total)

        if (ep + 1) % eval_interval == 0 or (ep + 1) == episodes:
            # Evaluation seeds are disjoint from training seeds.
            mean, std = evaluate(agent, spec.gym_id, eval_episodes, base_seed=10_000_000 + ep)
            result.eval_points.append(ep + 1)
            result.eval_means.append(mean)
            result.eval_stds.append(std)

    result.wall_time_s = time.perf_counter() - start
    env.close()
    return result
