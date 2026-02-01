"""
Configuration management for NWM agents.

This module provides dataclasses for configuring NWM agent behavior,
including exploration parameters, memory settings, and force field tuning.
"""

from dataclasses import dataclass, field
from typing import Optional


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
    
    def __post_init__(self):
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
