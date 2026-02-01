#!/usr/bin/env python3
"""
CartPole Training with Progress Visualization
==============================================

Complete training example with progress tracking and optional plotting.
"""

import gymnasium as gym
import numpy as np
from collections import deque
import argparse
import sys
import os

from nwm import NWM, NWMConfig


def run_episode(env, agent, training=True):
    """Run a single episode and return the total reward."""
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, training=training)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if training:
            agent.step(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    return total_reward


def train(
    num_episodes: int = 500,
    eval_every: int = 10,
    target_reward: float = 475,
    save_path: str = None
):
    """
    Train an NWM agent on CartPole.
    
    Parameters
    ----------
    num_episodes : int
        Total episodes to train.
    eval_every : int
        Evaluate every N episodes.
    target_reward : float
        Early stopping target.
    save_path : str
        Optional path to save the trained agent.
    """
    print("=" * 60)
    print("NWM CartPole Training")
    print("=" * 60)
    
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Configure agent
    config = NWMConfig(
        max_centroids=500,
        warmup_episodes=50,
        exploration_rate=1.0,
        exploration_decay=0.99,
        min_exploration=0.05
    )
    
    agent = NWM(
        state_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=config
    )
    
    print(f"\nState dim: {agent.state_dim}")
    print(f"Actions: {agent.num_actions}")
    print(f"Max centroids: {config.max_centroids}")
    print(f"Warmup episodes: {config.warmup_episodes}")
    print("-" * 60)
    
    # Training
    recent_rewards = deque(maxlen=100)
    best_avg = 0
    
    for episode in range(1, num_episodes + 1):
        reward = run_episode(env, agent, training=True)
        recent_rewards.append(reward)
        
        if episode % eval_every == 0:
            avg_reward = np.mean(recent_rewards)
            stats = agent.get_stats()
            
            print(f"Episode {episode:4d} | "
                  f"Avg100: {avg_reward:6.1f} | "
                  f"Best: {agent.best_reward:5.0f} | "
                  f"Centroids: {stats['num_centroids']:3d} | "
                  f"Locked: {stats['locked_centroids']:3d} | "
                  f"Mode: {stats['mode']}")
            
            if avg_reward > best_avg:
                best_avg = avg_reward
            
            # Early stopping
            if avg_reward >= target_reward:
                print(f"\nTarget reached! Average reward: {avg_reward:.1f}")
                break
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"Best average reward: {best_avg:.1f}")
    print(f"Best single episode: {agent.best_reward:.0f}")
    print(f"Total episodes: {agent.total_episodes}")
    
    # Save if requested
    if save_path:
        agent.save(save_path)
        print(f"Agent saved to: {save_path}")
    
    env.close()
    return agent


def demo(agent_path: str = None):
    """Run a visual demo with a trained agent."""
    env = gym.make("CartPole-v1", render_mode="human")
    
    if agent_path and os.path.exists(agent_path):
        print(f"Loading agent from {agent_path}")
        agent = NWM.load(agent_path)
    else:
        print("Training new agent...")
        agent = train(num_episodes=200, save_path=None)
    
    print("\nRunning demo (5 episodes)...")
    for i in range(5):
        reward = run_episode(env, agent, training=False)
        print(f"Demo episode {i+1}: {reward:.0f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NWM on CartPole")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    parser.add_argument("--demo", action="store_true", help="Run visual demo")
    parser.add_argument("--load", type=str, default=None, help="Load agent path")
    
    args = parser.parse_args()
    
    if args.demo:
        demo(args.load)
    else:
        train(num_episodes=args.episodes, save_path=args.save)
