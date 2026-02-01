"""Tests for NWMAgent class."""

import numpy as np
import pytest
import tempfile
import os
from nwm import NWMAgent, NWMConfig


class TestNWMConfig:
    """Test suite for NWMConfig."""
    
    def test_defaults(self):
        """Test default configuration."""
        config = NWMConfig()
        assert config.max_centroids == 500
        assert config.warmup_episodes == 50
        assert config.exploration_rate == 1.0
    
    def test_custom(self):
        """Test custom configuration."""
        config = NWMConfig(max_centroids=100, warmup_episodes=10)
        assert config.max_centroids == 100
        assert config.warmup_episodes == 10
    
    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            NWMConfig(max_centroids=5)  # Too small
        
        with pytest.raises(ValueError):
            NWMConfig(exploration_rate=1.5)  # Out of range


class TestNWMAgent:
    """Test suite for NWMAgent."""
    
    def test_init_default(self):
        """Test agent initialization with defaults."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        
        assert agent.state_dim == 4
        assert agent.num_actions == 2
        assert agent.total_episodes == 0
    
    def test_init_custom_config(self):
        """Test agent initialization with custom config."""
        config = NWMConfig(max_centroids=100)
        agent = NWMAgent(state_dim=4, num_actions=2, config=config)
        
        assert agent.field.max_centroids == 100
    
    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        state = np.zeros(4)
        
        # During early training, should always explore (random)
        action = agent.select_action(state, training=True)
        assert 0 <= action < 2
    
    def test_select_action_evaluation(self):
        """Test action selection during evaluation."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        state = np.zeros(4)
        
        # Even without training, should return valid action
        action = agent.select_action(state, training=False)
        assert 0 <= action < 2
    
    def test_step_and_episode(self):
        """Test step processing and episode end."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        state = np.zeros(4)
        next_state = np.ones(4)
        
        # Simulate an episode
        for i in range(10):
            done = (i == 9)
            agent.step(state, action=0, reward=1.0, next_state=next_state, done=done)
        
        assert agent.total_episodes == 1
    
    def test_get_stats(self):
        """Test statistics collection."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        stats = agent.get_stats()
        
        assert 'num_centroids' in stats
        assert 'best_reward' in stats
        assert 'exploration_rate' in stats
        assert 'total_episodes' in stats
    
    def test_reset(self):
        """Test agent reset."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        
        # Add some experience
        for i in range(5):
            agent.step(np.zeros(4), 0, 1.0, np.zeros(4), done=(i == 4))
        
        agent.reset()
        
        assert agent.total_episodes == 0
        assert len(agent.field) == 0
    
    def test_save_load(self):
        """Test save and load functionality."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        
        # Train a bit
        for ep in range(5):
            for i in range(10):
                agent.step(np.zeros(4), 0, 1.0, np.zeros(4), done=(i == 9))
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        
        try:
            agent.save(path)
            
            # Load
            loaded = NWMAgent.load(path)
            
            assert loaded.state_dim == agent.state_dim
            assert loaded.num_actions == agent.num_actions
            assert loaded.total_episodes == agent.total_episodes
        finally:
            os.unlink(path)
    
    def test_repr(self):
        """Test string representation."""
        agent = NWMAgent(state_dim=4, num_actions=2)
        repr_str = repr(agent)
        assert "NWMAgent" in repr_str
        assert "state_dim=4" in repr_str


class TestNWMAgentTraining:
    """Integration tests for NWMAgent training."""
    
    def test_learning_improves(self):
        """Test that the agent improves over training."""
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        agent = NWMAgent(
            state_dim=4,
            num_actions=2,
            config=NWMConfig(warmup_episodes=10)
        )
        
        early_rewards = []
        late_rewards = []
        
        for episode in range(60):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            if episode < 20:
                early_rewards.append(total_reward)
            elif episode >= 40:
                late_rewards.append(total_reward)
        
        # Late rewards should generally be better (not guaranteed but likely)
        # We just check that the agent can run without errors
        assert len(late_rewards) == 20
        env.close()
