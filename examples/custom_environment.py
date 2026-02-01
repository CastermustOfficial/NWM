#!/usr/bin/env python3
"""
Custom Environment Integration Example
=======================================

Shows how to use NWM with a custom Gymnasium environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from nwm import NWM, NWMConfig


class SimpleGridWorld(gym.Env):
    """
    A simple 5x5 grid world environment.
    
    The agent starts at (0, 0) and must reach (4, 4).
    Actions: 0=up, 1=right, 2=down, 3=left
    """
    
    def __init__(self):
        super().__init__()
        self.size = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.size-1, shape=(2,), dtype=np.float32
        )
        self.goal = np.array([4, 4], dtype=np.float32)
        self.pos = np.array([0, 0], dtype=np.float32)
        self.max_steps = 50
        self.steps = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0], dtype=np.float32)
        self.steps = 0
        return self.pos.copy(), {}
    
    def step(self, action):
        # Move
        moves = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}
        move = np.array(moves[action], dtype=np.float32)
        self.pos = np.clip(self.pos + move, 0, self.size - 1)
        self.steps += 1
        
        # Check goal
        done = np.allclose(self.pos, self.goal)
        truncated = self.steps >= self.max_steps
        
        # Reward
        if done:
            reward = 100
        else:
            dist = np.linalg.norm(self.pos - self.goal)
            reward = -0.1 - dist * 0.01
        
        return self.pos.copy(), reward, done, truncated, {}


def main():
    """Train NWM on the custom grid world."""
    print("Training NWM on SimpleGridWorld")
    print("=" * 40)
    
    env = SimpleGridWorld()
    
    config = NWMConfig(
        max_centroids=200,
        warmup_episodes=20,
        exploration_rate=1.0,
        min_exploration=0.1
    )
    
    agent = NWM(
        state_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=config
    )
    
    solved_count = 0
    
    for episode in range(200):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
        
        if done:
            solved_count += 1
        
        if (episode + 1) % 20 == 0:
            stats = agent.get_stats()
            success_rate = solved_count / (episode + 1) * 100
            print(f"Episode {episode + 1:3d} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Centroids: {stats['num_centroids']}")
    
    print("-" * 40)
    print(f"Final success rate: {solved_count / 200 * 100:.1f}%")


if __name__ == "__main__":
    main()
