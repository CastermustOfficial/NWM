#!/usr/bin/env python3
"""
NWM Quick Start Example
=======================

Minimal example showing how to train an NWM agent on CartPole.
"""

import gymnasium as gym
from nwm import NWM

# Create environment
env = gym.make("CartPole-v1")

# Create agent
agent = NWM(
    state_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n
)

# Training loop
for episode in range(100):
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
    
    if (episode + 1) % 10 == 0:
        stats = agent.get_stats()
        print(f"Episode {episode + 1}: Reward={total_reward:.0f}, "
              f"Centroids={stats['num_centroids']}, "
              f"Locked={stats['locked_centroids']}")

print(f"\nBest reward: {agent.best_reward:.0f}")
env.close()
