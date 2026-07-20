"""
Configuration management for NWM agents.

This module provides dataclasses for configuring NWM agent behavior,
including exploration parameters, memory settings, and force field tuning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NWMConfig:
    """
    Configuration for NWM Agent.

    This dataclass holds all tunable parameters for the NWM reinforcement
    learning agent. Parameters can be adjusted to optimize performance
    for different environments.

    Attributes
    ----------
    max_centroids : int
        Maximum number of memory centroids to maintain. Higher values
        provide more detailed spatial memory but use more computation.
        Default: 500

    warmup_episodes : int
        Number of episodes for pure exploration before using the
        potential field for action selection. Default: 50

    exploration_rate : float
        Initial exploration rate (epsilon). Range [0, 1].
        Default: 1.0 (full exploration at start)

    exploration_decay : float
        Decay factor applied to exploration rate after each episode.
        Default: 0.99

    min_exploration : float
        Minimum exploration rate floor. Default: 0.05

    merge_threshold : float
        Distance threshold for merging nearby experiences into
        existing centroids. Lower values = more centroids. Default: 0.3

    distance_cutoff : float
        Maximum distance for a centroid to influence action selection.
        Default: 2.5

    lock_min_visits : int
        Minimum visits required before a centroid can be locked.
        Default: 8

    lock_min_score : float
        Minimum average score required to lock a centroid. Default: 0.80

    exploration_repeat : float
        Probability of repeating the previous action instead of sampling a fresh
        uniform one during exploratory steps ("sticky" exploration). Values in
        ``(0, 1)`` build momentum for sparse-reward tasks (e.g. MountainCar);
        ``0.0`` (default) is classic uniform exploration.

    adaptive_repeat : bool
        When True, ``exploration_repeat`` auto-ramps (+0.05 per episode, up to
        0.95) while recent returns are perfectly flat -- the signature of a
        sparse-reward task whose goal has never been reached -- and stops as
        soon as returns vary. Solves MountainCar with no hand-tuning. Keep it
        off (default) for tasks that give a learning signal from the start.

    seed : int | None
        Seed for the agent's private random generator. ``None`` (default) draws
        fresh OS entropy. Set an integer for reproducible runs. An explicit
        ``seed`` argument to :class:`~nwm.NWMAgent` overrides this value.

    Examples
    --------
    >>> config = NWMConfig(max_centroids=1000, warmup_episodes=100)
    >>> agent = NWMAgent(state_dim=4, num_actions=2, config=config)

    >>> # Using defaults
    >>> config = NWMConfig()
    >>> print(config.max_centroids)
    500
    """

    # Memory settings
    max_centroids: int = 500
    merge_threshold: float = 0.3
    distance_cutoff: float = 2.5

    # Exploration settings
    warmup_episodes: int = 50
    exploration_rate: float = 1.0
    exploration_decay: float = 0.99
    min_exploration: float = 0.05

    # Locking criteria (Dynamic Smart Lock)
    lock_min_visits: int = 8
    lock_min_score: float = 0.80

    # Force field tuning
    attraction_weight: float = 1.0
    repulsion_weight: float = 1.5
    lock_boost: float = 2.5

    # --- Temporally-correlated ("sticky") exploration ---
    # Probability of repeating the previous action instead of sampling a fresh
    # uniform action during exploratory steps. Values in (0, 1) produce
    # temporally-correlated exploration that builds momentum in environments
    # requiring sustained torque (e.g. MountainCar). 0.0 = pure uniform.
    exploration_repeat: float = 0.0

    # When True, exploration_repeat auto-ramps (+0.05 per episode, up to 0.95)
    # while recent returns are constant -- i.e. the environment is giving no
    # learning signal at all, the signature of a sparse-reward task whose goal
    # has never been reached. It stops ramping as soon as returns vary.
    adaptive_repeat: bool = False

    # --- Credit assignment ---
    # When True, the temporal weight blends the episode score toward the
    # neutral value 0.5 (final = 0.5 + (score - 0.5) * t_weight) instead of
    # scaling it toward 0 (final = score * t_weight). With blending, early
    # steps of a good episode stay mildly attractive rather than becoming
    # repulsive. Helps dense-reward tasks (CartPole, Acrobot); keep it off
    # for MountainCar, where the repulsion-dominant legacy scheme is stronger.
    credit_blend: bool = False

    # When True, the dynamic quality gate is sign-aware: with negative average
    # returns the threshold becomes avg + 0.25*|avg| ("25% better than the
    # recent average"), so above-average episodes can earn attraction scores.
    # The legacy gate max(40, avg*1.25) always caps scores at 0.6 when returns
    # are negative, silently disabling attraction and the Smart Lock there.
    relative_gate: bool = False

    # --- Memory recency ---
    # If > 0, per-action centroid scores are exponential moving averages with
    # this smoothing factor instead of all-time running sums. Percentile
    # episode scores are non-stationary (a 0.9 from episode 10 is not a 0.9
    # from episode 190), so without forgetting, stale judgments accumulate.
    score_ema: float = 0.0

    # If True, a locked centroid whose recent merged scores collapse (EMA
    # below 0.45) loses its lock. The legacy Smart Lock is permanent, which
    # protects memories even after they stop being right.
    dynamic_unlock: bool = False

    # --- Truncation-aware credit ---
    # If True, temporal credit uses the terminated/truncated distinction:
    # a bad episode ending in a true terminal event keeps the late-step
    # emphasis (the end caused the failure), while a bad truncated episode
    # (aimless until the time limit) gets flat mild credit; a good truncated
    # episode (survived the full limit) gets full flat credit instead of
    # under-crediting its early steps. Requires passing ``terminated`` to
    # ``NWMAgent.step``; without it the legacy ramp is used.
    truncation_credit: bool = False

    # --- Environment-agnostic exploration collapse ---
    # If True, the adaptive exploration schedule uses percentiles of the
    # agent's own return history instead of the absolute thresholds 450/300,
    # which only make sense for CartPole-scale rewards.
    relative_explore: bool = False

    # If True, sticky exploration is also applied to the random fallback used
    # during greedy evaluation on unknown states, preserving momentum there.
    eval_sticky: bool = False

    # Reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_centroids < 10:
            raise ValueError("max_centroids must be at least 10")
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("exploration_rate must be between 0 and 1")
        if not 0 <= self.min_exploration <= 1:
            raise ValueError("min_exploration must be between 0 and 1")
        if self.merge_threshold <= 0:
            raise ValueError("merge_threshold must be positive")
        if self.distance_cutoff <= 0:
            raise ValueError("distance_cutoff must be positive")
        if self.warmup_episodes < 0:
            raise ValueError("warmup_episodes must be non-negative")
        if not 0 < self.exploration_decay <= 1:
            raise ValueError("exploration_decay must be in (0, 1]")
        if not 0 <= self.exploration_repeat <= 1:
            raise ValueError("exploration_repeat must be between 0 and 1")
