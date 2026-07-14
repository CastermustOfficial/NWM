"""Experiment configuration: environments, discretization grids, agent factories.

Environments are chosen to span a difficulty gradient:

- **CartPole-v1**  - dense reward, easy, the reference task in the original repo.
- **Acrobot-v1**   - medium, longer horizon, shaped negative reward.
- **MountainCar-v0** - hard, sparse reward, requires momentum building.

This spread is what makes the comparison informative: it shows not just whether
NWM works, but *where* a non-parametric potential-field method holds up against
value-based baselines and where it struggles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nwm import NWMConfig


@dataclass(frozen=True)
class EnvSpec:
    """Specification for one benchmark environment."""

    name: str
    gym_id: str
    episodes: int
    q_low: tuple[float, ...]
    q_high: tuple[float, ...]
    q_bins: tuple[int, ...]
    reward_threshold: float
    nwm_overrides: dict = field(default_factory=dict)


ENVIRONMENTS: dict[str, EnvSpec] = {
    "cartpole": EnvSpec(
        name="CartPole-v1",
        gym_id="CartPole-v1",
        episodes=200,
        q_low=(-2.4, -3.0, -0.21, -3.5),
        q_high=(2.4, 3.0, 0.21, 3.5),
        q_bins=(3, 3, 8, 8),
        reward_threshold=475.0,
        nwm_overrides={"warmup_episodes": 30},
    ),
    "acrobot": EnvSpec(
        name="Acrobot-v1",
        gym_id="Acrobot-v1",
        episodes=200,
        q_low=(-1.0, -1.0, -1.0, -1.0, -12.57, -28.27),
        q_high=(1.0, 1.0, 1.0, 1.0, 12.57, 28.27),
        q_bins=(6, 6, 6, 6, 8, 8),
        reward_threshold=-100.0,
        nwm_overrides={"warmup_episodes": 20},
    ),
    "mountaincar": EnvSpec(
        name="MountainCar-v0",
        gym_id="MountainCar-v0",
        episodes=300,
        q_low=(-1.2, -0.07),
        q_high=(0.6, 0.07),
        q_bins=(20, 20),
        reward_threshold=-110.0,
        nwm_overrides={"warmup_episodes": 20, "distance_cutoff": 3.0},
    ),
}


# Default seeds for the full protocol.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Evaluation protocol: greedy rollouts every ``eval_interval`` training episodes.
EVAL_INTERVAL = 10
EVAL_EPISODES = 10


def build_nwm(spec: EnvSpec, state_dim: int, num_actions: int, seed: int):
    from benchmarks.nwm_wrapper import NWMBenchmarkAgent

    config = NWMConfig(**spec.nwm_overrides)
    return NWMBenchmarkAgent(state_dim, num_actions, config=config, seed=seed)


def build_random(spec: EnvSpec, state_dim: int, num_actions: int, seed: int):
    from benchmarks.baselines import RandomAgent

    return RandomAgent(num_actions=num_actions, seed=seed)


def build_tabular_q(spec: EnvSpec, state_dim: int, num_actions: int, seed: int):
    from benchmarks.baselines import TabularQLearningAgent

    return TabularQLearningAgent(
        num_actions=num_actions,
        low=spec.q_low,
        high=spec.q_high,
        bins=spec.q_bins,
        seed=seed,
    )


def build_dqn(spec: EnvSpec, state_dim: int, num_actions: int, seed: int):
    from benchmarks.baselines.dqn import DQNAgent

    return DQNAgent(state_dim=state_dim, num_actions=num_actions, seed=seed)


#: Ordered mapping of agent label -> builder. DQN is optional (needs torch).
AGENT_BUILDERS = {
    "Random": build_random,
    "TabularQ": build_tabular_q,
    "DQN": build_dqn,
    "NWM": build_nwm,
}
