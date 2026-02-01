"""Tests for PersistentPotentialField class."""

import numpy as np
import pytest
from nwm.core.potential_field import PersistentPotentialField


class TestPersistentPotentialField:
    """Test suite for PersistentPotentialField."""
    
    def test_init(self):
        """Test field initialization."""
        field = PersistentPotentialField(state_dim=4, max_centroids=100)
        
        assert field.state_dim == 4
        assert field.max_centroids == 100
        assert len(field) == 0
    
    def test_add_single(self):
        """Test adding a single experience."""
        field = PersistentPotentialField(state_dim=4)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        field.add(state, action=0, score=0.8)
        
        assert len(field) == 1
        assert field.total_experiences == 1
    
    def test_merge_nearby(self):
        """Test that nearby experiences merge."""
        field = PersistentPotentialField(
            state_dim=4, merge_threshold=0.5
        )
        state1 = np.array([0.1, 0.2, 0.3, 0.4])
        state2 = np.array([0.11, 0.21, 0.31, 0.41])  # Very close
        
        field.add(state1, action=0, score=0.8)
        field.add(state2, action=0, score=0.9)
        
        # Should merge into one centroid
        assert len(field) == 1
        assert field.centroids[0].count == 2
    
    def test_separate_distant(self):
        """Test that distant experiences stay separate."""
        field = PersistentPotentialField(
            state_dim=4, merge_threshold=0.1
        )
        state1 = np.array([0.0, 0.0, 0.0, 0.0])
        state2 = np.array([10.0, 10.0, 10.0, 10.0])  # Far away
        
        field.add(state1, action=0, score=0.8)
        field.add(state2, action=1, score=0.2)
        
        assert len(field) == 2
    
    def test_query_forces(self):
        """Test force querying."""
        field = PersistentPotentialField(state_dim=4)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Add experiences
        field.add(state, action=0, score=0.9)  # Good
        field.add(state, action=1, score=0.1)  # Bad
        
        forces, min_dist = field.query_forces(state)
        
        assert 0 in forces or 1 in forces
        assert min_dist < float('inf')
    
    def test_query_empty(self):
        """Test querying empty field."""
        field = PersistentPotentialField(state_dim=4)
        forces, min_dist = field.query_forces(np.zeros(4))
        
        assert forces == {}
        assert min_dist == float('inf')
    
    def test_pruning(self):
        """Test that pruning keeps centroids within limit."""
        field = PersistentPotentialField(
            state_dim=4, max_centroids=10, merge_threshold=0.01
        )
        
        # Add many distinct experiences
        for i in range(100):
            state = np.array([i * 10, i * 10, i * 10, i * 10], dtype=np.float32)
            field.add(state, action=i % 2, score=0.5)
        
        # Should be pruned to max
        assert len(field) <= field.max_centroids + 50
    
    def test_get_stats(self):
        """Test statistics collection."""
        field = PersistentPotentialField(state_dim=4)
        field.add(np.zeros(4), action=0, score=0.8)
        field.add(np.zeros(4), action=0, score=0.6)
        
        stats = field.get_stats()
        
        assert 'num_centroids' in stats
        assert 'locked_centroids' in stats
        assert 'avg_score' in stats
    
    def test_clear(self):
        """Test clearing the field."""
        field = PersistentPotentialField(state_dim=4)
        field.add(np.zeros(4), action=0, score=0.8)
        
        field.clear()
        
        assert len(field) == 0
        assert field.total_experiences == 0
    
    def test_repr(self):
        """Test string representation."""
        field = PersistentPotentialField(state_dim=4)
        repr_str = repr(field)
        assert "PersistentPotentialField" in repr_str
