"""
NWM Agent implementation.

This module provides the main NWMAgent class - a reinforcement learning
agent that uses persistent potential fields for action selection.
"""

from __future__ import annotations

import bisect
import random
from collections import deque
from typing import Dict, List, Optional, Any
import pickle
from pathlib import Path

import numpy as np

from nwm.core.potential_field import PersistentPotentialField
from nwm.utils.config import NWMConfig


class NWMAgent:
    """
    Negative Weight Mapping Agent for Reinforcement Learning.
    
    This agent uses a persistent potential field to guide action selection
    through attractive (success) and repulsive (failure) forces. It implements
    the Dynamic Smart Lock mechanism for stable long-term memory.
    
    The agent is compatible with Gymnasium environments and provides a simple
    interface for training and evaluation.
    
    Parameters
    ----------
    state_dim : int
        Dimensionality of the state space.
    num_actions : int
        Number of available actions.
    config : Optional[NWMConfig]
        Configuration object. If None, uses default settings.
    
    Attributes
    ----------
    field : PersistentPotentialField
        The potential field storing experiences.
    exploration_rate : float
        Current exploration rate (epsilon).
    total_episodes : int
        Number of episodes completed.
    best_reward : float
        Best episode reward achieved.
    
    Examples
    --------
    >>> import gymnasium as gym
    >>> from nwm import NWMAgent
    >>>
    >>> env = gym.make("CartPole-v1")
    >>> agent = NWMAgent(
    ...     state_dim=env.observation_space.shape[0],
    ...     num_actions=env.action_space.n
    ... )
    >>>
    >>> for episode in range(100):
    ...     state, _ = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = agent.select_action(state)
    ...         next_state, reward, terminated, truncated, _ = env.step(action)
    ...         done = terminated or truncated
    ...         agent.step(state, action, reward, next_state, done)
    ...         state = next_state
    
    Notes
    -----
    The agent uses several key mechanisms:
    
    1. **Potential Field**: Maps states to forces based on past experiences
    2. **Dynamic Smart Lock**: Protects high-value memories from being overwritten
    3. **Adaptive Exploration**: Automatically reduces exploration as performance improves
    4. **Fear & Greed**: Avoids dangerous actions before seeking optimal ones
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        config: Optional[NWMConfig] = None
    ):
        """Initialize the NWM agent."""
        self.config = config or NWMConfig()
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Exploration settings
        self.exploration_rate = self.config.exploration_rate
        self.base_exploration = self.config.exploration_rate
        self.min_exploration = self.config.min_exploration
        self.exploration_decay = self.config.exploration_decay
        
        # Initialize potential field
        self.field = PersistentPotentialField(
            state_dim=state_dim,
            max_centroids=self.config.max_centroids,
            merge_threshold=self.config.merge_threshold,
            distance_cutoff=self.config.distance_cutoff,
            lock_min_visits=self.config.lock_min_visits,
            lock_min_score=self.config.lock_min_score,
            lock_boost=self.config.lock_boost
        )
        
        # Episode buffers
        self._current_states: List[np.ndarray] = []
        self._current_actions: List[int] = []
        self._current_rewards: List[float] = []
        
        # History and statistics
        self._reward_history: List[float] = []
        self._recent_rewards: deque = deque(maxlen=20)
        self._sorted_rewards: List[float] = []
        self.best_reward = float('-inf')
        self.total_episodes = 0
        self._adaptive_mode = "Normal"
        self._last_dynamic_threshold = 0.0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action for the given state.
        
        During training, uses epsilon-greedy exploration combined with
        the potential field. During evaluation, uses the potential field
        deterministically.
        
        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        training : bool
            If True, includes exploration. If False, exploitation only.
            
        Returns
        -------
        int
            Selected action index.
        """
        state = np.asarray(state, dtype=np.float32)
        
        # Exploration during warmup or with probability epsilon
        if training and (
            len(self._reward_history) < 10 or 
            random.random() < self.exploration_rate
        ):
            return random.randint(0, self.num_actions - 1)
        
        # Query potential field for forces
        forces, min_dist = self.field.query_forces(state)
        
        # If too far from known regions, explore
        if min_dist > self.field.distance_cutoff:
            return random.randint(0, self.num_actions - 1)
        
        if not forces:
            return random.randint(0, self.num_actions - 1)
        
        # Fear & Greed: Avoid dangerous actions first
        best_attraction = max(forces.values()) if forces else 0
        risk_tolerance = -0.5
        if best_attraction > 0.8:
            risk_tolerance = -1.5  # More cautious when we have good options
        
        safe_actions = [a for a in forces if forces[a] > risk_tolerance]
        
        if not safe_actions:
            return max(forces, key=forces.get)
        
        return max(safe_actions, key=lambda a: forces[a])
    
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        terminated: Optional[bool] = None
    ) -> None:
        """
        Process a single environment step.
        
        Accumulates experience and triggers learning at episode end.
        
        Parameters
        ----------
        state : np.ndarray
            State before action.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            State after action.
        done : bool
            Whether episode ended.
        terminated : Optional[bool]
            Deprecated. Use `done` instead.
        """
        state = np.asarray(state, dtype=np.float32)
        
        # Update state normalization statistics
        self.field._update_state_stats(state)
        
        # Buffer experience
        self._current_states.append(state)
        self._current_actions.append(action)
        self._current_rewards.append(reward)
        
        if done:
            self._end_episode()
    
    def _get_percentile_score(self, total_reward: float) -> float:
        """Calculate percentile rank of reward."""
        self._reward_history.append(total_reward)
        bisect.insort(self._sorted_rewards, total_reward)
        rank = bisect.bisect_left(self._sorted_rewards, total_reward)
        return rank / len(self._sorted_rewards)
    
    def _update_adaptive_exploration(self) -> None:
        """Adjust exploration rate based on performance."""
        if len(self._recent_rewards) < 10:
            return
        
        avg_recent = sum(self._recent_rewards) / len(self._recent_rewards)
        
        if avg_recent > 450:
            self.exploration_rate = 0.0
            self._adaptive_mode = "Perfect (Exp=0)"
        elif avg_recent > 300:
            self.exploration_rate = 0.02
            self._adaptive_mode = "Refining (Exp=0.02)"
        else:
            target = max(
                self.min_exploration,
                self.exploration_rate * self.exploration_decay
            )
            self.exploration_rate = target
            self._adaptive_mode = f"Decay (Exp={self.exploration_rate:.3f})"
    
    def _end_episode(self) -> None:
        """Process end of episode: update field with experiences."""
        self.total_episodes += 1
        total_reward = sum(self._current_rewards)
        self._recent_rewards.append(total_reward)
        
        if total_reward > self.best_reward:
            self.best_reward = total_reward
        
        # Skip learning during warmup
        if self.total_episodes < self.config.warmup_episodes:
            self._get_percentile_score(total_reward)
            self._clear_buffers()
            return
        
        score = self._get_percentile_score(total_reward)
        
        # Dynamic Quality Gate
        if len(self._recent_rewards) > 1:
            avg_recent = sum(self._recent_rewards) / len(self._recent_rewards)
        else:
            avg_recent = total_reward
        
        dynamic_threshold = max(40.0, avg_recent * 1.25)
        self._last_dynamic_threshold = dynamic_threshold
        
        if total_reward < dynamic_threshold:
            score = min(score, 0.6)  # No lock for mediocre results
        
        # Add experiences to field with temporal weighting
        n = len(self._current_states)
        for i in range(n):
            t_weight = 0.4 + 0.6 * (i / n)  # Later steps weighted higher
            weighted_score = score * t_weight
            final_score = max(0.0, min(1.0, weighted_score))
            
            self.field.add(
                self._current_states[i],
                self._current_actions[i],
                final_score
            )
        
        self._update_adaptive_exploration()
        self._clear_buffers()
    
    def _clear_buffers(self) -> None:
        """Clear episode buffers."""
        self._current_states = []
        self._current_actions = []
        self._current_rewards = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics including field info, best reward, and exploration rate.
        """
        stats = self.field.get_stats()
        stats.update({
            'best_reward': self.best_reward,
            'exploration_rate': self.exploration_rate,
            'mode': self._adaptive_mode,
            'dyn_threshold': f"{self._last_dynamic_threshold:.1f}",
            'total_episodes': self.total_episodes
        })
        return stats
    
    def reset(self) -> None:
        """
        Reset the agent to initial state.
        
        Clears all learned experiences and resets statistics.
        """
        self.field.clear()
        self._current_states = []
        self._current_actions = []
        self._current_rewards = []
        self._reward_history = []
        self._recent_rewards = deque(maxlen=20)
        self._sorted_rewards = []
        self.best_reward = float('-inf')
        self.total_episodes = 0
        self.exploration_rate = self.base_exploration
        self._adaptive_mode = "Normal"
        self._last_dynamic_threshold = 0.0
    
    def save(self, path: str) -> None:
        """
        Save agent state to file.
        
        Parameters
        ----------
        path : str
            File path for saving (pickle format).
        """
        state = {
            'config': self.config,
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'field': self.field,
            'exploration_rate': self.exploration_rate,
            'reward_history': self._reward_history,
            'recent_rewards': list(self._recent_rewards),
            'sorted_rewards': self._sorted_rewards,
            'best_reward': self.best_reward,
            'total_episodes': self.total_episodes,
            'adaptive_mode': self._adaptive_mode,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'NWMAgent':
        """
        Load agent from file.
        
        Parameters
        ----------
        path : str
            File path to load from.
            
        Returns
        -------
        NWMAgent
            Loaded agent instance.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        agent = cls(
            state_dim=state['state_dim'],
            num_actions=state['num_actions'],
            config=state['config']
        )
        agent.field = state['field']
        agent.exploration_rate = state['exploration_rate']
        agent._reward_history = state['reward_history']
        agent._recent_rewards = deque(state['recent_rewards'], maxlen=20)
        agent._sorted_rewards = state['sorted_rewards']
        agent.best_reward = state['best_reward']
        agent.total_episodes = state['total_episodes']
        agent._adaptive_mode = state['adaptive_mode']
        
        return agent
    
    def checkpoint_if_best(self, reward: float) -> None:
        """Placeholder for checkpoint logic (no-op by default)."""
        pass
    
    def restore_best_model(self) -> None:
        """Placeholder for restore logic (no-op by default)."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NWMAgent(state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"episodes={self.total_episodes}, "
            f"best={self.best_reward:.1f})"
        )
