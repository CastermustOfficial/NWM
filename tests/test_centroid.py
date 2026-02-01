"""Tests for PersistentCentroid class."""

import numpy as np
import pytest
from nwm.core.centroid import PersistentCentroid


class TestPersistentCentroid:
    """Test suite for PersistentCentroid."""
    
    def test_init(self):
        """Test centroid initialization."""
        state = np.array([0.1, 0.2, 0.3, 0.4])
        centroid = PersistentCentroid(state=state, score=0.8, action=1)
        
        assert centroid.count == 1
        assert centroid.avg_score == 0.8
        assert not centroid.locked
        assert 1 in centroid.action_votes
    
    def test_merge(self):
        """Test merging experiences."""
        state1 = np.array([0.1, 0.2, 0.3, 0.4])
        state2 = np.array([0.15, 0.25, 0.35, 0.45])
        
        centroid = PersistentCentroid(state=state1, score=0.8, action=1)
        centroid.merge(state2, score=0.9, action=1)
        
        assert centroid.count == 2
        assert abs(centroid.avg_score - 0.85) < 0.01
    
    def test_locking(self):
        """Test that centroids lock after enough high-score visits."""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        centroid = PersistentCentroid(state=state, score=0.85, action=0)
        
        # Merge 9 more times with high scores
        for _ in range(9):
            centroid.merge(state, score=0.85, action=0)
        
        assert centroid.locked  # Should be locked after 10 visits with avg > 0.8
    
    def test_get_force_attraction(self):
        """Test that high scores produce attraction."""
        centroid = PersistentCentroid(
            state=np.zeros(4), score=0.9, action=0
        )
        force = centroid.get_force(0)
        assert force > 0  # Attraction
    
    def test_get_force_repulsion(self):
        """Test that low scores produce repulsion."""
        centroid = PersistentCentroid(
            state=np.zeros(4), score=0.1, action=0
        )
        force = centroid.get_force(0)
        assert force < 0  # Repulsion
    
    def test_get_force_unknown_action(self):
        """Test that unknown actions return zero force."""
        centroid = PersistentCentroid(
            state=np.zeros(4), score=0.8, action=0
        )
        assert centroid.get_force(99) == 0.0
    
    def test_action_variance(self):
        """Test variance calculation."""
        centroid = PersistentCentroid(
            state=np.zeros(4), score=0.5, action=0
        )
        # Add varied scores
        centroid.merge(np.zeros(4), score=0.3, action=0)
        centroid.merge(np.zeros(4), score=0.7, action=0)
        
        var = centroid.get_action_variance(0)
        assert var > 0  # Should have some variance
    
    def test_repr(self):
        """Test string representation."""
        centroid = PersistentCentroid(
            state=np.zeros(4), score=0.8, action=1
        )
        repr_str = repr(centroid)
        assert "PersistentCentroid" in repr_str
        assert "count=1" in repr_str
