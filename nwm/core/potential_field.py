"""
Persistent Potential Field implementation for NWM framework.

The potential field stores a collection of centroids and provides
methods for querying attractive/repulsive forces based on spatial proximity.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional

from nwm.core.centroid import PersistentCentroid


class PersistentPotentialField:
    """
    Spatial memory structure that stores centroids and computes forces.
    
    The potential field maintains a collection of centroids representing
    learned experiences. When queried, it returns forces based on:
    - Spatial proximity to nearby centroids
    - Historical success/failure of actions in those regions
    - Confidence (variance) and lock status of centroids
    
    Attributes
    ----------
    state_dim : int
        Dimensionality of the state space.
    max_centroids : int
        Maximum number of centroids to maintain.
    centroids : List[PersistentCentroid]
        Collection of memory centroids.
    
    Examples
    --------
    >>> field = PersistentPotentialField(state_dim=4, max_centroids=100)
    >>> field.add(state=np.array([0.1, 0.2, 0.3, 0.4]), action=1, score=0.9)
    >>> forces, min_dist = field.query_forces(np.array([0.1, 0.2, 0.3, 0.4]))
    >>> print(forces)
    {1: 0.6}
    """
    
    def __init__(
        self, 
        state_dim: int, 
        max_centroids: int = 500,
        merge_threshold: float = 0.3, 
        distance_cutoff: float = 2.5,
        lock_min_visits: int = 8,
        lock_min_score: float = 0.80,
        lock_boost: float = 2.5
    ):
        """
        Initialize a new potential field.
        
        Parameters
        ----------
        state_dim : int
            Dimensionality of the state space.
        max_centroids : int
            Maximum centroids to maintain (older low-value ones are pruned).
        merge_threshold : float
            Distance threshold for merging experiences into existing centroids.
        distance_cutoff : float
            Maximum distance for a centroid to influence force calculations.
        lock_min_visits : int
            Minimum visits before a centroid can be locked.
        lock_min_score : float
            Minimum average score to allow locking.
        lock_boost : float
            Multiplier for locked centroid influence.
        """
        self.state_dim = state_dim
        self.max_centroids = max_centroids
        self.merge_threshold = merge_threshold
        self.distance_cutoff = distance_cutoff
        self.lock_min_visits = lock_min_visits
        self.lock_min_score = lock_min_score
        self.lock_boost = lock_boost
        
        self.centroids: List[PersistentCentroid] = []
        
        # Running state normalization statistics
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_var = np.ones(state_dim, dtype=np.float32)
        self.state_count = 0
        
        # Global statistics
        self.total_experiences = 0
        self.score_sum = 0.0
    
    def _update_state_stats(self, state: np.ndarray) -> None:
        """Update running mean and variance for state normalization."""
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state - self.state_mean
        self.state_var += (delta * delta2 - self.state_var) / self.state_count
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        if self.state_count < 10:
            return state
        std = np.sqrt(self.state_var + 1e-8)
        return (state - self.state_mean) / std
    
    def add(self, state: np.ndarray, action: int, score: float) -> None:
        """
        Add a new experience to the potential field.
        
        If a nearby centroid exists (within merge_threshold), the experience
        is merged. Otherwise, a new centroid is created.
        
        Parameters
        ----------
        state : np.ndarray
            Observed state.
        action : int
            Action taken.
        score : float
            Resulting score (0-1 scale, higher is better).
        """
        norm_state = self._normalize_state(state)
        
        self.total_experiences += 1
        self.score_sum += score
        
        if not self.centroids:
            self.centroids.append(PersistentCentroid(norm_state, score, action))
            return
        
        # Find nearest centroid
        centroid_states = np.array([c.state for c in self.centroids])
        dists = np.linalg.norm(centroid_states - norm_state, axis=1)
        nearest_idx = int(np.argmin(dists))
        min_dist = dists[nearest_idx]
        
        if min_dist < self.merge_threshold:
            self.centroids[nearest_idx].merge(
                norm_state, score, action,
                self.lock_min_visits, self.lock_min_score
            )
        else:
            self.centroids.append(PersistentCentroid(norm_state, score, action))
            if len(self.centroids) > self.max_centroids + 50:
                self._prune()
    
    def _prune(self) -> None:
        """Remove low-value centroids when capacity is exceeded."""
        if len(self.centroids) <= self.max_centroids:
            return
        
        # Preserve locked centroids
        locked = [c for c in self.centroids if c.locked]
        unlocked = [c for c in self.centroids if not c.locked]
        
        remaining_slots = self.max_centroids - len(locked)
        if remaining_slots < 0:
            # Too many locked: keep most valuable
            self.centroids.sort(
                key=lambda c: c.count * ((c.score_sum / c.count - 0.5) ** 2 + 0.1),
                reverse=True
            )
            self.centroids = self.centroids[:self.max_centroids]
            return
        
        # Keep most valuable unlocked centroids
        unlocked.sort(
            key=lambda c: c.count * ((c.score_sum / c.count - 0.5) ** 2 + 0.1),
            reverse=True
        )
        kept_unlocked = unlocked[:remaining_slots]
        self.centroids = locked + kept_unlocked
    
    def query_forces(
        self, 
        state: np.ndarray, 
        k: int = 20
    ) -> Tuple[Dict[int, float], float]:
        """
        Query forces for all actions from nearby centroids.
        
        Returns a dictionary mapping action IDs to force values:
        - Positive values indicate attraction (good outcomes)
        - Negative values indicate repulsion (bad outcomes)
        
        Parameters
        ----------
        state : np.ndarray
            Current state to query from.
        k : int
            Number of nearest centroids to consider.
            
        Returns
        -------
        Tuple[Dict[int, float], float]
            (forces_dict, min_distance) where forces_dict maps actions
            to their net force and min_distance is distance to nearest centroid.
        """
        if not self.centroids:
            return {}, float('inf')
        
        norm_state = self._normalize_state(state)
        
        centroid_states = np.array([c.state for c in self.centroids])
        all_dists = np.linalg.norm(centroid_states - norm_state, axis=1)
        nearest_indices = np.argsort(all_dists)[:k]
        
        forces: Dict[int, float] = {}
        total_weights: Dict[int, float] = {}
        min_dist = float(all_dists[nearest_indices[0]])
        
        for idx in nearest_indices:
            dist = float(all_dists[idx])
            if dist > self.distance_cutoff:
                continue
            
            centroid = self.centroids[idx]
            
            for action in centroid.action_votes:
                force = centroid.get_force(action)
                var = centroid.get_action_variance(action)
                confidence = 1.0 / (1.0 + var * 2.0)
                
                # Repulsion uses stronger spatial decay (inverse square)
                if force < 0:
                    spatial_weight = 1.0 / (dist ** 2 + 0.1)
                else:
                    spatial_weight = 1.0 / (dist + 0.1)
                
                # Boost for locked centroids
                lock_boost = self.lock_boost if centroid.locked else 1.0
                
                impact = (
                    force * spatial_weight * confidence * 
                    np.log(1 + centroid.count) * lock_boost
                )
                
                if action not in forces:
                    forces[action] = 0.0
                    total_weights[action] = 0.0
                
                forces[action] += impact
                total_weights[action] += spatial_weight
        
        # Normalize by total weights
        final_forces = {}
        for a in forces:
            if total_weights[a] > 0:
                final_forces[a] = forces[a] / total_weights[a]
        
        return final_forces, min_dist
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics about the potential field.
        
        Returns
        -------
        Dict[str, float]
            Statistics including centroid count, locked count, and scores.
        """
        locked_count = sum(1 for c in self.centroids if c.locked)
        return {
            'num_centroids': len(self.centroids),
            'locked_centroids': locked_count,
            'avg_score': self.score_sum / max(1, self.total_experiences),
            'avg_state_variance': float(np.mean([
                c.get_state_variance() for c in self.centroids
            ])) if self.centroids else 0.0
        }
    
    def clear(self) -> None:
        """Clear all centroids and reset statistics."""
        self.centroids = []
        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_var = np.ones(self.state_dim, dtype=np.float32)
        self.state_count = 0
        self.total_experiences = 0
        self.score_sum = 0.0
    
    def __len__(self) -> int:
        """Return number of centroids."""
        return len(self.centroids)
    
    def __repr__(self) -> str:
        """String representation."""
        locked = sum(1 for c in self.centroids if c.locked)
        return (
            f"PersistentPotentialField(centroids={len(self.centroids)}, "
            f"locked={locked}, experiences={self.total_experiences})"
        )
