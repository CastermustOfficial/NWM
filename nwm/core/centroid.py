"""
Persistent Centroid implementation for NWM framework.

A centroid represents a region in state space with accumulated experience
about which actions lead to success or failure in that region.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


class PersistentCentroid:
    """
    A memory unit that stores aggregated experience for a region of state space.
    
    Each centroid maintains:
    - A representative state (running average of nearby observed states)
    - Action-specific scores tracking success/failure for each action
    - A "locked" status for high-confidence memories
    
    The centroid uses Welford's algorithm for numerically stable running
    statistics and implements the Dynamic Smart Lock mechanism.
    
    Attributes
    ----------
    state : np.ndarray
        The representative state vector (running mean of merged states).
    count : int
        Number of experiences merged into this centroid.
    locked : bool
        Whether this centroid is locked (high-confidence memory).
    
    Examples
    --------
    >>> centroid = PersistentCentroid(
    ...     state=np.array([0.1, 0.2, 0.3, 0.4]),
    ...     score=0.8,
    ...     action=1
    ... )
    >>> centroid.merge(np.array([0.15, 0.25, 0.35, 0.45]), score=0.9, action=1)
    >>> print(f"Avg score: {centroid.avg_score:.2f}")
    Avg score: 0.85
    """
    
    __slots__ = ['state', 'state_sq_sum', 'score_sum', 'count', 'action_votes', 'locked']
    
    def __init__(self, state: np.ndarray, score: float, action: int):
        """
        Create a new centroid from an initial observation.
        
        Parameters
        ----------
        state : np.ndarray
            Initial state observation.
        score : float
            Score for this state-action pair (0-1 scale).
        action : int
            Action taken in this state.
        """
        self.state = np.array(state, dtype=np.float32)
        self.state_sq_sum = self.state * self.state
        self.score_sum = score
        self.count = 1
        self.action_votes: Dict[int, List[float]] = {action: [score, 1, score * score]}
        self.locked = False
    
    @property
    def avg_score(self) -> float:
        """Average score across all merged experiences."""
        return self.score_sum / self.count
    
    def merge(
        self, 
        state: np.ndarray, 
        score: float, 
        action: int,
        lock_min_visits: int = 8,
        lock_min_score: float = 0.80
    ) -> None:
        """
        Merge a new experience into this centroid.
        
        Uses exponential moving average with increasing stiffness for
        well-established centroids (count > 50).
        
        Parameters
        ----------
        state : np.ndarray
            Observed state to merge.
        score : float
            Score for this experience (0-1 scale).
        action : int
            Action taken.
        lock_min_visits : int
            Minimum visits to allow locking.
        lock_min_score : float
            Minimum avg score to allow locking.
        """
        # Increase stiffness for well-established centroids
        denominator = self.count + 1
        if self.count > 50:
            denominator = self.count * 2
        
        alpha = 1.0 / denominator
        
        # Update running state mean and variance
        self.state = self.state * (1 - alpha) + state * alpha
        self.state_sq_sum = self.state_sq_sum * (1 - alpha) + (state * state) * alpha
        
        self.score_sum += score
        self.count += 1
        
        # Dynamic Smart Lock: lock if statistically valid and high-performing
        if not self.locked:
            if self.count > lock_min_visits and self.avg_score > lock_min_score:
                self.locked = True
        
        # Update action-specific statistics
        if action not in self.action_votes:
            self.action_votes[action] = [0.0, 0, 0.0]
        
        self.action_votes[action][0] += score
        self.action_votes[action][1] += 1
        self.action_votes[action][2] += score * score
    
    def get_force(self, action: int) -> float:
        """
        Calculate the force (attraction/repulsion) for a specific action.
        
        Forces are computed based on the average score for the action:
        - Positive force (attraction) for high scores (> 0.6)
        - Negative force (repulsion) for low scores (< 0.4)
        - Zero force for neutral scores
        
        Parameters
        ----------
        action : int
            Action to query.
            
        Returns
        -------
        float
            Force value. Positive = attraction, negative = repulsion.
        """
        if action not in self.action_votes:
            return 0.0
        
        s, n, _ = self.action_votes[action]
        avg_score = s / n
        
        if avg_score > 0.6:
            return (avg_score - 0.5) * 1.5  # Attraction
        elif avg_score < 0.4:
            return (avg_score - 0.5) * 1.5  # Repulsion
        return 0.0
    
    def get_action_variance(self, action: int) -> float:
        """
        Get variance of scores for a specific action.
        
        Higher variance means less confidence in the force estimate.
        
        Parameters
        ----------
        action : int
            Action to query.
            
        Returns
        -------
        float
            Variance of scores, or 1.0 if action not observed.
        """
        if action not in self.action_votes:
            return 1.0
        
        s, n, s_sq = self.action_votes[action]
        if n < 2:
            return 0.0
        
        mean = s / n
        return max(0.0, (s_sq / n) - (mean * mean))
    
    def get_state_variance(self) -> float:
        """
        Get variance of merged states.
        
        Low variance means the centroid is tightly clustered.
        
        Returns
        -------
        float
            Average variance across state dimensions.
        """
        if self.count < 2:
            return 0.0
        
        mean_sq = self.state_sq_sum
        sq_mean = self.state * self.state
        return float(np.mean(np.maximum(0, mean_sq - sq_mean)))
    
    def __repr__(self) -> str:
        """String representation of the centroid."""
        return (
            f"PersistentCentroid(count={self.count}, "
            f"avg_score={self.avg_score:.3f}, "
            f"locked={self.locked}, "
            f"actions={list(self.action_votes.keys())})"
        )
